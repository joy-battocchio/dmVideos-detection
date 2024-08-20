import sys
# sys.path.append('RAFT/core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from torchvision.models import resnet18, resnet34, resnet50
from PIL import Image
from torchvision.models.video import r3d_18, R3D_18_Weights

from torchvision import transforms as T

import head
import configparser

configParser = configparser.RawConfigParser()   
configFilePath = "cuda.config"
configParser.read(configFilePath)
DEVICE = configParser.get("cuda-config", "device")
TWINDOW = int(configParser.get("cuda-config", "temporal_window"))

class DetectorFromRGB(torch.nn.Module):

    def __init__(self, args) -> None:
        super(DetectorFromRGB, self).__init__()
        self.of_extractor = torch.nn.DataParallel(RAFT(args))
        self.of_extractor.load_state_dict(torch.load(args.model))
        self.encoder = resnet50()
        self.encoder.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #self.avg = torch.nn.AdaptiveAvgPool1d(1000)
        self.head = head.Head(1000,1)
        #self.freeze(self.of_extractor)
        self.args = args
        #self.transform = T.Resize(360,360)

    def freeze(self, part):
        part.eval()
        for param in part.parameters():
            param.requires_grad = False

    def rescale(self, of):
        mins = of.view(of.shape[0], of.shape[1], -1).min(dim=2).values
        maxs = of.view(of.shape[0], of.shape[1], -1).max(dim=2).values
        mins = mins[:,:,None,None].repeat(1,1,of.shape[2],of.shape[3])
        maxs = maxs[:,:,None,None].repeat(1,1,of.shape[2],of.shape[3])
        return of/(maxs-mins)

    def forward(self,x):
        x = x.squeeze(0)
        of = torch.empty(0,device=DEVICE)
        for frame1, frame2 in zip(x[:-1], x[1:]):
            of = torch.cat((of,self.of_extractor(frame1.unsqueeze(0), frame2.unsqueeze(0), iters=20, test_mode=True)[1]))

        feats = self.encoder(of)
        out = self.head(feats)
        return torch.mean(out).unsqueeze(0)

class DetectorFromOF(torch.nn.Module):

    def __init__(self, args) -> None:
        super(DetectorFromOF, self).__init__()
        self.encoder = resnet18()
        self.head = head.Head(1000,1)
        self.args = args

    def forward(self,x):
        x = x.squeeze(0)
        feats = self.encoder(x)
        out = self.head(feats)
        return torch.mean(out).unsqueeze(0)

class Baseline(torch.nn.Module):

    def __init__(self, args) -> None:
        super(Baseline, self).__init__()
        self.encoder = resnet50()
        self.head = head.Head(1000,1)

    def forward(self,x):
        x = x.squeeze()
        feats = self.encoder(x)
        out = self.head(feats)
        return out.squeeze()

class Resnet3d(torch.nn.Module):
    def __init__(self) -> None:
        super(Resnet3d,self).__init__()
        self.encoder = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        self.head = head.Head(400,1)

    def forward(self,x):
        x = x.squeeze(0) if len(x.shape) > 5 else x
        feats = self.encoder(x)
        out = self.head(feats)
        return out.squeeze(), feats

class Resnet3d_18(torch.nn.Module):
    def __init__(self) -> None:
        super(Resnet3d_18, self).__init__()
        self.encoder = r3d_18(weights = R3D_18_Weights)
        weights = self.encoder.stem[0].weight.clone()
        self.encoder.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=(TWINDOW, 7, 7), stride=(1, 2, 2), padding=(TWINDOW//2, 3, 3), bias=False)
        self.encoder.stem[0].weight = torch.nn.Parameter(weights)
        self.head = head.Head(400,1)

    def forward(self,x):
        x = x.squeeze(0) if len(x.shape) > 5 else x
        feats = self.encoder(x)
        out = self.head(feats)
        return out.squeeze(), feats

def prepare_video(path):
    
    frames = glob.glob(os.path.join(path, '*.png')) + \
            glob.glob(os.path.join(path, '*.jpg'))

    frames = sorted(frames)
    frame0 = load_frame(frames[0])
    padder = InputPadder(frame0.shape)
    frame0 = padder.pad(frame0)[0]
    x = torch.tensor(frame0, device=DEVICE)

    for file in frames[1:]:

        frame = load_frame(file)
        frame = padder.pad(frame)[0]
        x = torch.cat((x,torch.tensor(frame, device=DEVICE)))

    return x

def load_frame(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    x = prepare_video('/home/fbk/test_DM/RAFT/demo-frames')
    detector = Detector(args)
    detector.to(DEVICE)
    out = detector(x)