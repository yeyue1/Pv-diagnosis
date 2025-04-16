# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleCNN, self).__init__()
        
        # 简单卷积层结构 - 输入维度为[B, 6, 80]
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 计算全连接层的输入维度: 80/2/2 = 20, 32*20 = 640
        self.fc1 = nn.Linear(32 * 20, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 确保输入格式正确 [B, 6, 80]
        if x.size(1) != 6:
            # 如果输入格式是[B, 80, 6]，转换为[B, 6, 80]
            x = x.transpose(1, 2)
            
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
