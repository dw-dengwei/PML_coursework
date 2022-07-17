from pydoc import ispackage
import torch.nn as nn
import torch
import torch.nn.functional as F

class NephNet2D(nn.Module):
    def make_layer(self, 
                   in_channels, 
                   out_channels,
                   is_pooling=False,
                   kernel_size=3, 
                   stride=1, 
                   padding=1):
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if is_pooling:
            layers.add_module('pooling', nn.AvgPool2d(2, 2))
        return layers

    def __init__(self, num_feature=8, num_classes=8):
        super().__init__()
        self.num_feature=num_feature
        self.layer = nn.Sequential(
            self.make_layer(1, self.num_feature, is_pooling=False),
            self.make_layer(self.num_feature, self.num_feature),
            self.make_layer(self.num_feature, self.num_feature),
            self.make_layer(self.num_feature, self.num_feature * 2),
            self.make_layer(self.num_feature * 2, self.num_feature * 2),
            self.make_layer(self.num_feature * 2, self.num_feature * 2),
            self.make_layer(self.num_feature * 2, self.num_feature * 4, True),
            self.make_layer(self.num_feature * 4, self.num_feature * 4, True),
            self.make_layer(self.num_feature * 4, self.num_feature * 4, True),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature * 4 * 3 * 3, self.num_feature * 4),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(self.num_feature * 4, num_classes)
        )    
                
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        x = self.fc_layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, 
                      out_channel, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, 
                      out_channel, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, 
                          out_channel, 
                          kernel_size=1, 
                          stride=stride, 
                          bias=False),
                nn.BatchNorm2d(out_channel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, input_chanel=1, num_classes=8):
        super(ResNet, self).__init__()
        self.in_channel = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_chanel, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 4, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 8, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 16, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 32, 2, stride=2)        
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(32, num_classes),
        )
        self._initialize()

    def make_layer(self, Block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)    # normal: mean=0, std=1
    
    def forward(self, x):
        # print(x.size())
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out