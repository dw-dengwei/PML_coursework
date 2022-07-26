import torch
import cv2
import numpy as np
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as lr_scheduler
import random
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from os import path
from PIL import Image

# model
out_dim = 8
num_features = 8
use_apex = True # If you haven't install apex, turn false, see more details on 
                # https://github.com/NVIDIA/apex

# environment
device = 'cuda' # if don't have GPU, turn 'cpu'
random_seed = 0

# dataset
label_csv_path = '/home/dw-dengwei/dataset/tissue/train.csv'
image_root = '/home/dw-dengwei/dataset/tissue/train/'

# experiment
train_batch_size = 128
valid_batch_size = 128
epoch_num = 50
learning_rate = 0.025336627337952815
weight_decay =  0.0046684446208345
training_ratio = 0.8
save_path = "save/checkpoint.pth"
TEST = False

# SGD optimizer
momentum =  0.8
nesterov = True

# scheduler
patience = 5
factor = 0.29

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

# split training set and validation set
n = len(ids)
train_ids = random.sample(ids, k=int(n * training_ratio))
valid_ids = list(set(ids) - set(train_ids))

ids = ids[:100]

# import apex
if use_apex:
    from apex import amp

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


def train_epoch(model: nn.Module, epoch: int, dataloader: DataLoader, optimizer, device):
    model.train()
    pred_true = 0
    pred_tot = 0
    loss_sum = 0
    verbose = 10

    batch_acc = []
    batch_loss = []

    # train batch
    for iter, batch in enumerate(dataloader):
        image, label = batch
        image = image.to(torch.device(device))
        label = label.to(torch.device(device))
        predict = model.forward(image)
        loss = F.cross_entropy(predict, label)

        optimizer.zero_grad()
        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        _, pred_class = predict.max(1)
        pred_true += (label == pred_class).sum()
        pred_tot += label.size(0)
        loss_sum += loss.item()
        acc = (label == pred_class).sum() / label.size(0)
        batch_acc.append(acc)
        batch_loss.append(loss)
        if (iter + 1) % verbose == 0:
            print("[Train:{}-{}/{}]\tAcc:{}\tLoss:{}".format( 
                  epoch, 
                  iter, 
                  len(dataloader), 
                  acc, 
                  loss))

    epoch_acc = pred_true / pred_tot
    epoch_loss = loss_sum / len(dataloader)
    return epoch_acc, batch_acc, epoch_loss, batch_loss


@torch.no_grad()
def evaluate(model: nn.Module, epoch: int, dataloader: DataLoader, device):
    model.eval()
    pred_true = 0
    pred_tot = 0
    loss_sum = 0
    verbose = 10

    batch_acc = []
    batch_loss = []

    for iter, batch in enumerate(dataloader):
        image, label = batch
        image = image.to(torch.device(device))
        label = label.to(torch.device(device))
        predict = model.forward(image)

        loss = F.cross_entropy(predict, label)

        _, pred_class = predict.max(1)
        pred_true += (label == pred_class).sum()
        pred_tot += label.size(0)
        loss_sum += loss.item()
        acc = (label == pred_class).sum() / label.size(0)
        batch_acc.append(acc)
        batch_loss.append(loss)
        if (iter + 1) % verbose == 0:
            print("[Valid:{}-{}/{}]\tAcc:{}\tLoss:{}".format( 
                  epoch, 
                  iter, 
                  len(dataloader), 
                  acc, 
                  loss))


    epoch_acc = pred_true / pred_tot
    epoch_loss = loss_sum / len(dataloader)
    return epoch_acc, batch_acc, epoch_loss, batch_loss


# dataset and dataloader
print('dataset and dataloader')
dataset_train, dataset_valid = \
    Cell(train_ids, label_db, image_root), \
    Cell(valid_ids, label_db, image_root,)
dataloader_train, dataloader_valid = \
    DataLoader(dataset_train, train_batch_size, shuffle=True), \
    DataLoader(dataset_valid, valid_batch_size, shuffle=True), 
# model
print('model')
model = Toy(num_feature=num_features, num_classes=out_dim)
# load checkpoint before test
if TEST:
    print('load checkpoint')
    model.load_state_dict(torch.load(save_path))
model = model.to(device=device)

# using SGD optimizer
print('using SGB optimizer')
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    momentum=momentum,
    nesterov=nesterov
)
# use apex mix precision training
if use_apex:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1') 

# reduce learning rate on plateau
print('reduce learning rate on plateau')
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=patience, factor=factor
)


# start training
print("\033[0;31;40mStart Training\033[0m")
train_loss_per_batch = []
train_loss_per_epoch = []
train_acc_per_batch = []
train_acc_per_epoch = []

valid_loss_per_batch = []
valid_loss_per_epoch = []
valid_acc_per_batch = []
valid_acc_per_epoch = []
if not TEST:
    for epoch in range(epoch_num):
        train_acc, train_batch_acc, train_loss, train_batch_loss = \
            train_epoch(model, epoch, dataloader_train, optimizer, device)
        valid_acc, valid_batch_acc, valid_loss, valid_batch_loss = \
            evaluate(model, epoch, dataloader_valid, device)

        train_loss_per_batch.append({epoch: train_batch_loss})
        train_loss_per_epoch.append(train_loss)
        train_acc_per_batch.append({epoch: train_batch_acc})
        train_acc_per_epoch.append(train_acc)

        valid_loss_per_batch.append({epoch: valid_batch_loss})
        valid_loss_per_epoch.append(valid_loss)
        valid_acc_per_batch.append({epoch: valid_batch_acc})
        valid_acc_per_epoch.append(valid_acc)

        scheduler.step(valid_loss)
        print("[Train:{}]\tAcc:{}\tLoss:{}"\
            .format(epoch, train_acc, train_loss))
        print("\033[0;31;40m[Valid:{}]\tAcc:{}\tLoss:{}\033[0m"\
            .format(epoch, valid_acc, valid_loss))
else:
    valid_acc, valid_batch_acc, valid_loss, valid_batch_loss = \
        evaluate(model, dataloader_valid, device)


if not TEST:
    torch.save(model.state_dict(), save_path)


import pickle
def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()

save_variable(train_acc_per_epoch,  'save/train_acc_per_epoch.bin')
save_variable(train_loss_per_epoch, 'save/train_loss_per_epoch.bin')

save_variable(valid_acc_per_epoch,  'save/valid_acc_per_epoch.bin')
save_variable(valid_loss_per_epoch, 'save/valid_loss_per_epoch.bin')


def load_variavle(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

train_loss_per_epoch = load_variavle('save/train_loss_per_epoch.bin')
valid_loss_per_epoch = load_variavle('save/valid_loss_per_epoch.bin')

train_acc_per_epoch  = load_variavle('save/train_acc_per_epoch.bin')
valid_acc_per_epoch  = load_variavle('save/valid_acc_per_epoch.bin')


def remove_torch(tensor_list: list) -> list:
    ret = tensor_list.copy()
    for i, _ in enumerate(ret):
        ret[i] = ret[i].item()
    
    return ret

pd_train_loss_per_epoch = pd.Series(train_loss_per_epoch)
pd_valid_loss_per_epoch = pd.Series(valid_loss_per_epoch)
pd_train_acc_per_epoch  = pd.Series(remove_torch(train_acc_per_epoch))
pd_valid_acc_per_epoch  = pd.Series(remove_torch(valid_acc_per_epoch))
loss = pd.concat([pd_train_loss_per_epoch, pd_valid_loss_per_epoch], axis=1)
acc  = pd.concat([pd_train_acc_per_epoch, pd_valid_acc_per_epoch], axis=1)

loss.columns = ['Training', 'Validation']
acc.columns = ['Training', 'Validation']

plt.figure(1)
plt.plot(acc, label=acc.columns)
plt.title('Accuracy')
plt.legend(loc='lower right')
plt.savefig("save/acc.png", dpi=700, format='png')
plt.savefig("save/acc.svg", format='svg')

plt.figure(2)
plt.plot(loss, label=loss.columns)
plt.title('Loss')
plt.legend(loc='upper right')
plt.savefig("save/loss.png", dpi=700, format='png')
plt.savefig("save/loss.svg", format='svg')