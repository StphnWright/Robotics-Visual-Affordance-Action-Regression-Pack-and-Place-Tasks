from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from pick_labeler import draw_grasp


class ActionRegressionDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for ActionRegressionModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32
            'target': torch.Tensor (3,), torch.float32
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        Note: target: [x, y, angle] scaled to between 0 and 1.
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        
        # TODO: complete this method
        # ===============================================================================
        
        # Extract info and convert to numpy format
        image_rgb = np.array(data['rgb'])
        center_point = np.array(data['center_point'])
        angle = np.array(data['angle'])
        
        # Normalize the center point
        x_center_norm = center_point[0] / image_rgb.shape[0]
        y_center_norm = center_point[1] / image_rgb.shape[1]
        
        # For the input, permute the axes to (2, 0, 1) and normalize
        image_rgb = np.moveaxis(image_rgb, -1, 0)
        image_rgb -= np.min(image_rgb)
        if np.max(image_rgb) != 0:
            image_rgb = image_rgb / np.max(image_rgb)
        
        # Normalize the angle (0 to 360)
        angle = np.max([np.min([angle, 180]), -180])
        angle = (angle + 180.0) / 360.0      
                
        # Return input and target as torch floating point tensors
        return {'input': torch.from_numpy(image_rgb).type(torch.float32), 
                'target': torch.from_numpy(np.array([x_center_norm, y_center_norm, angle])).type(torch.float32)}
        
        # ===============================================================================


def recover_action(
        action: np.ndarray, 
        shape=(128,128)
        ) -> Tuple[Tuple[int, int], float]:
    """
    :action: np.ndarray([x,y,angle], dtype=np.float32)
    return:
    coord: tuple(x, y) in pixel coordinate between 0 and :shape:
    angle: float in degrees, clockwise
    """
    # TODO: complete this function
    # =============================================================================== 
    # Coordinate (action[0] = x center and action[1] = y center, both normalized between 0 and 1)
    coord = (int(action[0] * shape[0]), int(action[1] * shape[1]))
    # Angle (action[2] = angle from 0 to 360)
    angle = (action[2] * 360.0) - 180.0
    # ===============================================================================
    return coord, angle


class ActionRegressionModel(nn.Module):
    def __init__(self, pretrained=False, out_channels=3, **kwargs):
        super().__init__()
        # load backbone model
        model = mobilenet_v3_small(pretrained=pretrained)
        #model = mobilenet_v3_small(weights='DEFAULT')
        # replace the last linear layer to change output dimention to 3
        ic = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features=ic, out_features=out_channels)
        self.model = model
        # normalize RGB input to zero mean and unit variance
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))

    def predict(self, x):
        """
        Think: Why is this the same as forward 
        (comparing to AffordanceModel.predict)
        """
        return self.forward(x)

    @staticmethod
    def get_criterion():
        """
        Return the Loss object needed for training.
        """
        # TODO: complete this method
        # =============================================================================== 
        return nn.MSELoss() # step 0 training loss 0.46671417355537415; 0.2459, 0.4054
        #return nn.L1Loss() # step 0 training loss 0.5691288709640503; 0.4044, 0.5113
        #return nn.SmoothL1Loss() # step 0 training loss 0.22948019206523895; 0.1162, 0.2203
        #return nn.CrossEntropyLoss() # step 0 training loss 1.6520912647247314; 1.5480, 2.1169
        #return nn.BCEWithLogitsLoss() # step 0 training loss 0.7338048815727234; 0.6884, 0.8005
        # =============================================================================== 

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """        
        vis_img = (np.moveaxis(input,0,-1).copy() * 255).astype(np.uint8)
        # target
        if target is not None:
            coord, angle = recover_action(target, shape=vis_img.shape[:2])
            draw_grasp(vis_img, coord, angle, color=(255,255,255))
        # pred
        coord, angle = recover_action(output, shape=vis_img.shape[:2])
        draw_grasp(vis_img, coord, angle, color=(0,255,0))
        return vis_img

    def predict_grasp(self, rgb_obs: np.ndarray
            ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given a RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Hint: use recover_action
        """
        device = self.device
        # TODO: complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        
        # Format the input (permute axes and normalize)
        input = np.moveaxis(rgb_obs, -1, 0)
        input_torch = input - np.min(input)
        if np.max(input_torch) != 0:
            input_torch = input_torch / np.max(input_torch)
        input_torch = np.expand_dims(input_torch, axis = 0) # TODO MAY NEED TO COMMENT OUT
        input_torch = torch.from_numpy(input_torch).type(torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            action = self.predict(input_torch)[0].cpu().numpy()
        (coord_recover, angle) = recover_action(action)
        
        # Ensure coordinates are within the image
        coord_x = np.minimum(np.maximum(coord_recover[0], 0), 127)
        coord_y = np.minimum(np.maximum(coord_recover[1], 0), 127)
        coord = (coord_x, coord_y)
            
        # ===============================================================================
        # visualization
        vis_img = self.visualize(input, action)
        return coord, angle, vis_img

