import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, output_dim=40, batch_size = 1):
        super().__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)    
        self.fc1 = nn.Linear(8*16*16, 512)
        self.fc2 = nn.Linear(512, 2) # label 0 means fake, 1 means real

    def forward(self, input_batch):
        step1 = F.relu(self.conv1(input_batch)) 
        step2 = self.pool(step1)
        step3 = F.relu(self.conv2(step2)) 
        step4 = self.pool(step3).reshape(self.batch_size, -1)
        step5 = F.relu(self.fc1(step4))
        ret = self.fc2(step5)
        
        return ret