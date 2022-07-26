import torch
import cv2
import numpy as np
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from os import path
from PIL import Image

# model
out_dim = 8
num_features = 8

# environment
device = 'cuda' # if don't have GPU, turn 'cpu'
random_seed = 0

# dataset
label_csv_path = '/home/dw-dengwei/dataset/tissue/fake_test.csv'
image_root = '/home/dw-dengwei/dataset/tissue/fake_test/'

# experiment
test_batch_size = 16
save_path = "save/checkpoint128.pth"
TEST = True

# fix random seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# load label and id into memory
label_dataframe = pd.read_csv(label_csv_path)
label_db = {}
ids = []
for index, rows in label_dataframe.iterrows():
    idx, label = rows['Id'], rows['Cell type']
    label_db[str(idx)] = label
    ids.append(str(idx))

test_ids = ids


# Dataset class
class Cell(Dataset):
    def __init__(self, 
                 ids: list, 
                 label_db: dict, 
                 image_root: str, 
        ) -> None:
        '''
        ids: indexes of the training/validating data
        label_db: dict structure of train.csv file
        image_root: image root path
        '''

        self.num = len(ids)
        self.label = []
        self.image = []

        # These operations may cause large memory in runtime and time use at start.
        # Using database file is recommended.
        for idx in ids:
            target = label_db[idx]
            img_cv = cv2.imread(path.join(image_root, str(idx).rjust(6, '0') + ".jpg"))
            img = self.transform(img_cv)
            self.label.append(target)
            self.image.append(img)

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, index):
        print(index, self.image[index], self.label[index])
        return self.image[index], self.label[index]

    def transform(self, img_cv) -> torch.Tensor:
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[0:1, :, :]),
            ])
        return tf(img_pil)


# model structure
class Toy(nn.Module):
    def __init__(self, num_feature, num_classes):
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = x.view(x.size()[0], -1)
        x = self.fc_layer(x)
        return x


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device):
    model.eval()
    y_hat = []
    for iter, batch in enumerate(dataloader):
        image, label = batch['image'], batch['label']
        image = image.to(torch.device(device))
        label = label.to(torch.device(device))
        predict = model.forward(image)
        y_hat.append(predict)
    
    return y_hat


# dataset and dataloader
print('dataset and dataloader')
dataset_test = Cell(test_ids, label_db, image_root,)
dataloader_test = DataLoader(dataset_test, test_batch_size, shuffle=True), 
# model
print('model')
model = Toy(num_feature=num_features, num_classes=out_dim)
# load checkpoint before test
print('load checkpoint')
model.load_state_dict(torch.load(save_path))
model = model.to(device=device)

y_hat = evaluate(model, dataloader_test, device)