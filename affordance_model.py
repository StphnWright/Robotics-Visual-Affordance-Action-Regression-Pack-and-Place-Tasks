from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp
from collections import deque

def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
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
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: (problem 2) complete this method and return the correct input and target as data dict
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        
        # Image in numpy format
        image_rgb = np.array(data['rgb'])
        
        # Make the center point the sole keypoint
        keypts = KeypointsOnImage(
          [Keypoint(x = data['center_point'][0], y = data['center_point'][1]),], 
          shape = image_rgb.shape)
        
        # Rotate
        rotate_img = iaa.Affine(rotate = -data['angle'].item())
        input_img, keypts_aug = rotate_img(image = image_rgb, keypoints = keypts)
        
        # For the input, permute the axes to (2, 0, 1) and normalize
        input_img = np.moveaxis(input_img, -1, 0)
        
        # For the target, take the Gaussian scoremap, add an axis, and normalize
        target = get_gaussian_scoremap((image_rgb.shape[0], image_rgb.shape[1]), 
                                       np.array([keypts_aug[0].x, keypts_aug[0].y]))
        target = np.expand_dims(target, axis = 0)
        
        # Return input and target as torch floating point tensors
        return {'input': torch.from_numpy(input_img).type(torch.float32), 
                'target': torch.from_numpy(target).type(torch.float32)}


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(
        self, 
        rgb_obs: np.ndarray,  
    ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: (problem 2) complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        
        # Number of angle ranges
        n_angle_ranges = 8
        angle_delta = 180 / n_angle_ranges
        
        # Stack the rotated images
        rgb_rot_list = []
        for i in range(n_angle_ranges):
            # Rotate
            rotate_i = iaa.Affine(rotate = (i * -angle_delta))
            rgb_rot = rotate_i(image = rgb_obs)
            
            # Permute axes to (2, 0, 1), i.e. 3 x H x W
            rgb_rot = np.moveaxis(rgb_rot, -1, 0)
            
            # Add to stack
            rgb_rot_list.append(rgb_rot)
        
        # Stack and convert to Torch float tensors on the given device
        rgb_rot_stack = np.stack(rgb_rot_list)
        input_stack = torch.from_numpy(rgb_rot_stack).type(torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            prediction = self.predict(input_stack)

        # Get the index of the max prediction
        max_idx = int(torch.argmax(prediction))
        
        # Index of the predicted angle
        i_pred = max_idx // (128 * 128)
        angle = i_pred * angle_delta
        
        # Predicted center point    
        x_center_pred = max_idx % (128 * 128) % 128
        y_center_pred = max_idx % (128 * 128) // 128
        coord_pred = (x_center_pred, y_center_pred)
                
        # ===============================================================================
        
        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================
        # If there is at least one past action, modify the prediction accordingly
        if self.past_actions:
            # Initial affordance map
            affordance_map = get_gaussian_scoremap(shape = (rgb_obs.shape[0], rgb_obs.shape[1]), 
                                                   keypoint = np.array(coord_pred), sigma = 4)
            
            # Downvote past actions by subtracting their affordance map
            for max_coord in list(self.past_actions):
                supression_map = get_gaussian_scoremap(shape = (rgb_obs.shape[0], rgb_obs.shape[1]), 
                                                       keypoint = np.array(max_coord), sigma = 4)
                affordance_map -= supression_map
            
            # Select the new maximum index 
            max_idx = np.argmax(affordance_map)
            
            # Compute the new predicted center coordinate 
            x_center_pred = max_idx % 128
            y_center_pred = max_idx // 128
            coord_pred = (x_center_pred, y_center_pred)
            
        # Add the current action to the past action list
        self.past_actions.append(coord_pred)
        
        # ===============================================================================
        
        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        
        # Return image stack and prediction to CPU for visualization
        input_stack = np.array(input_stack.cpu())
        prediction = np.array(prediction.cpu())
        
        # Draw the predicted grasp
        draw_grasp(self.visualize(input_stack[i_pred, ...], prediction[i_pred, ...]), coord_pred, 0.0)
                
        # Run through the list of all possible rotation angles
        vis_img = []        
        for i in range(8):
            # Retrieve image
            vis_image_i = self.visualize(input_stack[i, ...], prediction[i, ...])
            
            # Draw a grey line on the bottom row
            vis_image_i[127, :, :] = 127
            
            # Append to list
            vis_img.append(vis_image_i)
            
        # Stack the images
        vis_img = np.concatenate([np.vstack(vis_img[::2]), np.vstack(vis_img[1::2])], axis = 1)
        
        # Rotate the predicted point
        input_pred = np.moveaxis(input_stack[i_pred, ...], 0, -1)
        coord_pred_keypt = KeypointsOnImage([Keypoint(x = coord_pred[0], y = coord_pred[1]),], shape = input_pred.shape)
        rotate_pred = iaa.Affine(rotate = angle)
        coord_pred_keypt_rot = rotate_pred(image = input_pred, keypoints = coord_pred_keypt)[1]
        
        # Ensure coordinates are within the image
        coord_x = np.minimum(np.maximum(int(coord_pred_keypt_rot[0].x), 0), 127)
        coord_y = np.minimum(np.maximum(int(coord_pred_keypt_rot[0].y), 0), 127)
        coord = (coord_x, coord_y)
        
        # ===============================================================================
        return coord, angle, vis_img

