from pydoc import ispackage
import torch.nn as nn
import torch
import torch.nn.functional as F

class NephNet2D(nn.Module):
    def make_layer(self, 
                   in_channels, 
                   out_channels,
                   is_pooling=True,
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

    def __init__(self, num_feature, num_classes=8):
        super().__init__()
        self.num_feature=num_feature
        self.layer = nn.Sequential(
            self.make_layer(1, self.num_feature, is_pooling=False),
            self.make_layer(self.num_feature, self.num_feature, is_pooling=False),
            self.make_layer(self.num_feature, self.num_feature, is_pooling=False),

            self.make_layer(self.num_feature, self.num_feature * 2),
            self.make_layer(self.num_feature * 2, self.num_feature * 2, is_pooling=False),

            self.make_layer(self.num_feature * 2, self.num_feature * 4),

            self.make_layer(self.num_feature * 4, self.num_feature * 8),

            self.make_layer(self.num_feature * 8, self.num_feature * 16, is_pooling=False),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature * 16 * 3 * 3, self.num_feature * 16),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_feature * 16, num_classes)
        )    
                
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        x = self.fc_layer(x)
        return x