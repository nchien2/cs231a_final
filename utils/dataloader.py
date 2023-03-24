# DATALOADER 
from . import augmentations
import torch
from torch.utils.data import Dataset, DataLoader
import ujson
import os
import pandas as pd


# TODO: Add keep_uncertainty flag
# TODO: Add explicit stratification
class ASLLVDataset(Dataset):
  def __init__(self, df, pose_dir, 
               transform=True, 
               keep_uncertainty=True,
               normalization=True, 
               train=False):
      self.df = df
      self.pose_dir = pose_dir
      self.transform = transform
      self.keep_uncertainty = keep_uncertainty
      self.train=train
      self.normalization = normalization

  def __len__(self):
      return len(self.df)

  def __getitem__(self, idx):     
      label = self.df['label index'].iloc[idx]
      json_idx = self.df['json index'].iloc[idx]
      pose_path = os.path.join(self.pose_dir, f'{json_idx}')

      frame_keypoints = []
      for filename in os.listdir(pose_path):
        # print(f'filename: {filename}')
        json_path = os.path.join(pose_path, filename)
        with open(json_path) as f:
            kp_json = ujson.load(f)
            kp_list = [
                kp_json['pose_keypoints_2d'][3:],
                kp_json['face_keypoints_2d'],
                kp_json['hand_left_keypoints_2d'],
                kp_json['hand_right_keypoints_2d']
            ]
            kp_list = [torch.tensor(x) for x in kp_list]
            keypoint_tensor = torch.cat(kp_list)

        if self.train and self.transform:
          keypoint_tensor = augmentations.augment(keypoint_tensor)


        x, y, uncertainty = augmentations.keypoint_to_coord(keypoint_tensor)

        if not self.keep_uncertainty:
          keypoint_tensor = keypoint_tensor.reshape(-1, 3)[:,:2].reshape(-1)

        frame_out = torch.stack((x, y), dim=1)

        # print(f'keypoint_tensor: {keypoint_tensor.shape}')
        frame_keypoints.append(frame_out)
      full_pose = torch.stack(frame_keypoints, dim=0)
    #   full_pose = augmentations.interpolate_keypoints(full_pose)
      # print(f'full pose: {full_pose.shape}')
      


      return full_pose, torch.tensor(label)

def collate_fn(data):
  poses, labels = zip(*data)
  return poses, labels