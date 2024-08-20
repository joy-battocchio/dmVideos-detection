"""
@author: Joy Battocchio
Dataset classes
1. FairFace
2. UTKFace
3. Re-Id
"""

import sys
#sys.path.append('RAFT/core')
import cv2
import os
from glob import glob
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import v2
from torchvision.transforms.functional import center_crop
from torchvision.transforms.functional import resize
import random 
import numpy as np

# configParser = configparser.RawConfigParser()   
# configFilePath = "cuda.config"
# configParser.read(configFilePath)
# DEVICE = configParser.get("cuda-config", "device")
DEVICE = "cpu"

SEED = 1234
NFRAMES = 5

random.seed(SEED)

def remove_mp4(videos):
    return [v for v in videos if '.mp4' not in v]

class Video_dataset(Dataset):
    def __init__(self, split = 'train', dataset = '', small = None, augment = True, techniques = ['*']):
        super().__init__()

        self.videos = []
        self.split = split
        if isinstance(techniques, list):
            with open(f'media/{self.split}.txt') as file:
                for line in file:
                    for tec in techniques:
                        v_name = line.rstrip()
                        self.videos += glob(f'/media/mmlab/Datasets_4TB/videodiffusion/{tec}/{dataset}/{v_name}')
                    self.videos += glob(f'/media/mmlab/Datasets_4TB/videodiffusion/clips_original/{dataset.replace("_gen", "")}/{v_name}')
        else:
            self.videos += glob(f"/media/mmlab/Datasets_4TB/{techniques}/frames/*")
        self.augment = augment
        self.data = self.videos
  

        self.transform = T.Compose(
            [
                T.RandomRotation(30),
                T.RandomVerticalFlip(p=0.3),
                T.RandomHorizontalFlip(p=0.3),
                T.RandomCrop((200,200))
                #T.CenterCrop((200,200))
            ]
        )
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):
        
        folder_path = self.data[index]
        image_files = [os.path.join(folder_path, file) for file in sorted(os.listdir(folder_path)) if file.endswith('.png')]  # Adjust the file extension as needed

        if not image_files:
            raise ValueError("No image files found in the folder.")

        frame = cv2.imread(image_files[0])
        height, width, _ = frame.shape

        # video_tensor = torch.empty(0,dtype=torch.uint8, device=DEVICE)
        video_list = []

        if self.split == "train":
            n_frames = 16
            start_frame = random.randint(0,max(0,len(image_files)-n_frames-1))
            end_frame = min(start_frame+n_frames, len(image_files)-1)
        if self.split == "val":
            start_frame = 0
            end_frame = min(32, len(image_files))

        for image_file in image_files[start_frame:end_frame]:
            frame = cv2.imread(image_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            # Ensure all frames have the same dimensions
            if frame.shape != (height, width, 3):
                raise ValueError("All frames must have the same dimensions.")
            video_list.append(frame) 
            #video_tensor = torch.cat((video_tensor, frame.to(DEVICE).unsqueeze(0)))

        video_tensor = torch.tensor(np.array(video_list)).permute(0,3,1,2) # (frames, channels, height, width)
        quality = random.randint(70,100)
        if self.augment:
            video_tensor = v2.functional.jpeg(video_tensor, quality)
            # if random.random() > 0.7:
            #     video_tensor = T.functional.gaussian_blur(video_tensor, 5)

        #video_tensor = torch.tensor(video_frames, device = DEVICE)  # Convert list of frames to PyTorch tensor
        if self.split == 'train':
            x = self.transform(video_tensor)
        else:
            x = video_tensor

        x = x.permute(1, 0, 2, 3)  # permute as wanted by the network
        # Label
        y = 1 if 'clips_original' in folder_path else 0
        #y = 1 if 'real' in folder_path else 0
        return (x / 127.5 - 1.0), torch.tensor(y, device=DEVICE, dtype=torch.float32)

class BL_dataset(Dataset):
    def __init__(self, split = 'train', dataset = '', small = None, techniques = ['*']):
        super().__init__()
        # self.videos = []
        self.split = split
        self.data = []
        with open(f'media/{self.split}.txt') as file:
            for line in file:
                for tec in techniques:
                    v_name = line.rstrip()
                    fake_frames = glob(f'/media/mmlab/Datasets_4TB/videodiffusion/{tec}/{dataset}/{v_name}/*.png')
                    self.data += fake_frames
                real_frames = glob(f'/media/mmlab/Datasets_4TB/videodiffusion/clips_original/{dataset.replace("_gen", "")}/{v_name}/*.png')
                real_frames = sorted(real_frames)
                self.data += real_frames[:len(fake_frames)]
        # for v in self.videos:
        #     self.data.extend(glob(f'{v}/*.png'))
        random.shuffle(self.data)
 
        self.transform = T.Compose([
            T.RandomCrop((200,200))
        ])
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):
        
        filename = self.data[index]
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.uint8)
        x = torch.tensor(img, device=DEVICE)
        x = x.permute(2,0,1)
        if self.split == 'train':
            x = self.transform(x)


        # Label
        y = 1 if 'clips_original' in filename else 0

        return x/127.5 - 1.0 ,torch.tensor(y, device=DEVICE, dtype=torch.float32)
    