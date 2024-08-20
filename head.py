"""
@author: Joy Battocchio  
Classification head for attribute estimation 
"""

import sys
sys.path.append('RAFT/core')

import torch
import torch.utils.data as data
import numpy as np
import torch.nn.functional as ff
import torchvision.transforms as t
from torch import nn

class Head(torch.nn.Module):
    
    def __init__(self, in_size, out_size):
        '''__init__ _summary_

        initialization of classification head model

        Parameters
        ----------
        in_size : int
            feature vector input size
        out_size : int
            number of output classes
        '''
        super(Head,self).__init__()
        self.fc1 = nn.Linear(in_size, in_size//2)
        self.bn = nn.BatchNorm1d(in_size//2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_size//2, out_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        '''forward _summary_

        forward pass

        Parameters
        ----------
        x : torch tensor
            network input

        Returns
        -------
        torch tensor
            prediction probabilities
        '''
        out = self.fc1(x)
        #out = self.bn(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out
    
if __name__ == '__main__':
    
    pass