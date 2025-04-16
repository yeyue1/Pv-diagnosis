# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU6(inplace=True)    )

def ConvBNReLUFactorization(in_channels,out_channels,kernel_sizes,paddings):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1,padding=paddings),
        nn.BatchNorm1d(out_channels),
        nn.ReLU6(inplace=True)    )

class InceptionV3ModuleA(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleA, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=5, padding=2),        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1),        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV3ModuleB(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleB, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=7,paddings=3),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=7,paddings=3),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels2, kernel_sizes=7,paddings=3),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV3ModuleC(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleC, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=3,paddings=1)

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3,stride=1,padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=1,paddings=0)

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),        )

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = self.branch2_conv2a(x2)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = self.branch3_conv3a(x3)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV3ModuleD(nn.Module):
    def __init__(self, in_channels,out_channels1reduce,out_channels1,out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3,stride=2)        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2, kernel_size=3, stride=2),        )

        self.branch3 = nn.MaxPool1d(kernel_size=3,stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionV3ModuleE(nn.Module):
    def __init__(self, in_channels, out_channels1reduce,out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleE, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2),        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce,kernel_sizes=7, paddings=3),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=2),        )

        self.branch3 = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

class InceptionAux(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionAux, self).__init__()

        self.auxiliary_avgpool = nn.AvgPool1d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_conv2 = nn.Conv1d(in_channels=128, out_channels=768, kernel_size=5,stride=1)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear1 = nn.Linear(in_features=768, out_features=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = self.auxiliary_conv2(x)
        x = x.view(x.size(0), -1)
        out = self.auxiliary_linear1(self.auxiliary_dropout(x))
        return out

class InceptionV3(nn.Module):
    def __init__(self, num_classes=8, stage='train'):
        super(InceptionV3, self).__init__()
        self.stage = stage

        self.mb = nn.Sequential(
            ConvBNReLU(in_channels=6, out_channels=32, kernel_size=3, stride=2),
            ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2),
            ConvBNReLU(in_channels=64, out_channels=80, kernel_size=1, stride=1),
            ConvBNReLU(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2),
            InceptionV3ModuleA(in_channels=192, out_channels1=64, out_channels2reduce=48, out_channels2=64,
                               out_channels3reduce=64, out_channels3=96, out_channels4=32),
            InceptionV3ModuleA(in_channels=256, out_channels1=64, out_channels2reduce=48, out_channels2=64,
                               out_channels3reduce=64, out_channels3=96, out_channels4=64),
            InceptionV3ModuleA(in_channels=288, out_channels1=64, out_channels2reduce=48, out_channels2=64,
                               out_channels3reduce=64, out_channels3=96, out_channels4=64),
            InceptionV3ModuleD(in_channels=288, out_channels1reduce=384, out_channels1=384, out_channels2reduce=64,
                               out_channels2=96),
            InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=128, out_channels2=192,
                               out_channels3reduce=128, out_channels3=192, out_channels4=192),
            InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160, out_channels2=192,
                               out_channels3reduce=160, out_channels3=192, out_channels4=192),
            InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160, out_channels2=192,
                               out_channels3reduce=160, out_channels3=192, out_channels4=192),
            InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=192, out_channels2=192,
                               out_channels3reduce=192, out_channels3=192, out_channels4=192),
            InceptionV3ModuleE(in_channels=768, out_channels1reduce=192, out_channels1=320, out_channels2reduce=192,
                               out_channels2=192),
            InceptionV3ModuleC(in_channels=1280, out_channels1=320, out_channels2reduce=384, out_channels2=384,
                               out_channels3reduce=448, out_channels3=384, out_channels4=192),
            InceptionV3ModuleC(in_channels=1280, out_channels1=320, out_channels2reduce=384, out_channels2=384,
                               out_channels3reduce=448, out_channels3=384, out_channels4=192),
        )

        self.drop = nn.Dropout(p=0.5)
        self.linear = nn.Sequential(nn.Linear(1280, 256),
                                    nn.Linear(256, 64),
                                    nn.Linear(64, num_classes),
                                    )

    def forward(self, x):
        x = self.mb(x)
        x = self.drop(x)
        x1 = x.view(x.size(0),-1)
        out = self.linear(x1)

        return x1,out



class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.mb = nn.Sequential(
            nn.Conv1d(6, 64, 3),  # 64 * 222 * 222
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),   # 64 * 222* 222
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),  # pooling 64 * 112 * 112

            nn.Conv1d(64, 128, 3),  # 128 * 110 * 110
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),  # 128 * 110 * 110
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),  # pooling 128 * 56 * 56

            nn.Conv1d(128, 256, 3),  # 256 * 54 * 54
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),  # 256 * 54 * 54
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),  # 256 * 54 * 54
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),  # pooling 256 * 28 * 28

            nn.Conv1d(256, 512, 3),  # 512 * 26 * 26
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),  # 512 * 26 * 26
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),  # 512 * 26 * 26
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),  # pooling 512 * 14 * 14

            nn.Conv1d(512, 512, 3),  # 512 * 12 * 12
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),  # 512 * 12 * 12
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),  # 512 * 12 * 12
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),  # pooling 512 * 7 * 7
        )

        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.mb(x)  # 222

        out_1 = out.view(in_size, -1)

        out_1 = self.drop( out_1 )
        out_2 = self.fc1( out_1 )
        out_2 = F.relu(out_2)
        out_2 = self.fc2(out_2)
        out_2 = F.relu(out_2)
        out_2 = self.fc3(out_2)

        label = F.log_softmax(out_2)

        return out_1,label



"""
    input = torch.randn(1, 3, 299, 299)
    aux,out = model(input)
    print(aux.shape)
    print(out.shape)

"""