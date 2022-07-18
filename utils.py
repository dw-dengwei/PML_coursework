from random import random
import torchvision.transforms as transforms
import pandas as pd
import random
import torch
import cv2
from PIL import Image


def get_ids_and_label_db(label_csv_path: str):
    label_dataframe = pd.read_csv(label_csv_path)
    label_db = {}
    ids = []
    for index, rows in label_dataframe.iterrows():
        idx, label = rows['Id'], rows['Cell type']
        label_db[idx] = label
        ids.append(idx)

    ids = ids[:10000] 
    return ids, label_db


def split_train_valid(ids: list, ratio: float=0.8):
    n = len(ids)
    train = random.sample(ids, k=int(n * ratio))
    valid = list(set(ids) - set(train))

    return train, valid


def transform(img_cv) -> torch.Tensor:
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    tf = transforms.Compose([
            # transforms.CenterCrop(15),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[0:1, :, :]),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # transforms.Normalize(
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225])
            # transforms.Lambda(lambda x: (x - 64.92524191326531) / 57.59275746686797)
        ])
    return tf(img_pil) 


class Config(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict2obj(d):
    if not isinstance(d, dict):
        return d

    obj = Config()
    for k, v in d.items():
        obj[k] = dict2obj(v)
    return obj

# Test code below
if __name__ == '__main__':
    label_csv_path = '/home/dw-dengwei/dataset/tissue/train.csv'
    ids, label_db = get_ids_and_label_db(label_csv_path)