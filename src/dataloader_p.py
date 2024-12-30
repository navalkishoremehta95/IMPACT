import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import json
class R_IMU_data(Dataset):
    def __init__(self, root_dir, folders, imu_transform=None, rgb_transform=None, pose_transform=None):
        """
        Args:
            root_dir (str): Root directory containing multiple subfolders for each video.
            folders (list): List of subfolder 
            imu_transform (callable, optional): transform to be applied to the IMU images.
            rgb_transform (callable, optional): transform to be applied to the RGB images.
        """
        self.root_dir = root_dir
        self.folders = folders
        self.imu_transform = imu_transform
        self.rgb_transform = rgb_transform
        self.pose_transform = pose_transform  

        self.data = self.collect_all_data()
        self.filtered_frames = self.filter_frames_with_labels()


    def collect_all_data(self):
   
        data = []
        for folder_name in self.folders:
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                rgb_path = os.path.join(folder_path, 'rgb')
                imu_left_path = os.path.join(folder_path, 'imu_gaf_merged_left')
                imu_right_path = os.path.join(folder_path, 'imu_gaf_merged_right')
                label_file = os.path.join(folder_path, 'frame_metadata_with_status.csv')
                pose_json_folder = os.path.join(folder_path, 'pose_json')   

                if all([os.path.exists(p) for p in [rgb_path, imu_left_path, imu_right_path, label_file, pose_json_folder]]):
                    data.append({
                        'rgb_path': rgb_path,
                        'imu_left_path': imu_left_path,
                        'imu_right_path': imu_right_path,
                        'label_file': label_file,
                        'pose_json_folder': pose_json_folder  
                    })
        return data

    def filter_frames_with_labels(self):
      
        filtered_frames = []
        num_frames_in_window = 50  # 5 seconds at 10 FPS

        for entry in self.data:
            labels = pd.read_csv(entry['label_file'])
            
            for start_idx in range(0, len(labels), num_frames_in_window):
                rgb_frames = labels.iloc[start_idx:start_idx + num_frames_in_window]
                if len(rgb_frames) < num_frames_in_window:
                    continue   

                # Sample 16 RGB frames evenly from the 50 frames in the window
                sampled_rgb_indices = np.linspace(0, num_frames_in_window - 1, 16).astype(int)
                sampled_rgb_frames = rgb_frames.iloc[sampled_rgb_indices]
                
                sampled_rgb_label = sampled_rgb_frames['label'].iloc[-1]
                label_mapping = {'engaged': 0, 'disengaged': 1}

                if sampled_rgb_frames['label'].iloc[0] != sampled_rgb_frames['label'].iloc[-1]:
                    sampled_rgb_label = 'disengaged'
                
                imu_frame_number = int(sampled_rgb_frames.iloc[-1]['frame_filename'].split('_')[1].replace('.jpg', ''))          
                imu_left_frame = f"merged_gaf_frame_{imu_frame_number:06d}.jpg"
                imu_right_frame = f"merged_gaf_frame_{imu_frame_number:06d}.jpg"
                imu_left_exist = os.path.exists(os.path.join(entry['imu_left_path'], imu_left_frame))
                imu_right_exist = os.path.exists(os.path.join(entry['imu_right_path'], imu_right_frame))
                
                frame_number = int(sampled_rgb_frames.iloc[-1]['frame_filename'].split('_')[1].replace('.jpg', ''))
                pose_json_path = os.path.exists(os.path.join(entry['pose_json_folder'], f"keypoints_{frame_number}.json"))
               
                if not imu_left_exist or not imu_right_exist or sampled_rgb_label == 'unknown'or not pose_json_path:
                    continue  

                filtered_frames.append((sampled_rgb_frames, imu_left_frame, imu_right_frame, label_mapping[sampled_rgb_label], entry))
        
        return filtered_frames

    def __len__(self):
        return len(self.filtered_frames)

    def __getitem__(self, idx):
        rgb_frames, imu_left_frame, imu_right_frame, labels, entry = self.filtered_frames[idx]
        frame_number = int(rgb_frames.iloc[-1]['frame_filename'].split('_')[1].replace('.jpg', ''))
        pose_json_path = os.path.join(entry['pose_json_folder'], f"keypoints_{frame_number}.json")
        # print(">>>>>>>>>>>>>>>>>inside dataloader>>>>>>>>>>>>>>>")
        # print(rgb_frames, imu_left_frame, imu_right_frame,pose_json_path, labels)
        # Load RGB images
        rgb_images = []
        for rgb_frame in rgb_frames['frame_filename']:
            rgb_image_path = os.path.join(entry['rgb_path'], rgb_frame)
            rgb_image = cv2.imread(rgb_image_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            if self.rgb_transform:
                rgb_image = self.rgb_transform(Image.fromarray(rgb_image))  # Apply transforms to RGB image
            rgb_images.append(rgb_image)
        rgb_images = torch.stack(rgb_images, dim=1)

        # Load IMU images (left and right)
        imu_left_image_path = os.path.join(entry['imu_left_path'], imu_left_frame)
        imu_right_image_path = os.path.join(entry['imu_right_path'], imu_right_frame)
        imu_left_image = cv2.imread(imu_left_image_path)
        imu_right_image = cv2.imread(imu_right_image_path)
        
        if self.imu_transform:
            imu_left_image = self.imu_transform(Image.fromarray(imu_left_image))
            imu_right_image = self.imu_transform(Image.fromarray(imu_right_image))

        labels = torch.tensor(labels).long()

        with open(pose_json_path, 'r') as f:
            pose_data = json.load(f)
            pose_keypoints = torch.tensor(pose_data['keypoints'])   
            if self.pose_transform:
                pose_keypoints = self.pose_transform(np.asarray(pose_keypoints))   
        # print(rgb_images, imu_left_image, imu_right_image,pose_keypoints,  labels)
        return rgb_images, imu_left_image, imu_right_image,pose_keypoints,  labels
