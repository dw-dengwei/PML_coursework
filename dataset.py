from torch.utils.data.dataset import Dataset
from PIL import Image
from utils import transform
import pandas as pd
import cv2
import os.path as path


class Tissue(Dataset):
    def __init__(self, ids: list, label_db: dict, image_root: str) -> None:
        '''
        ids: indexes of the training/validating data
        label_db: train.csv file to dict
        image_root: image root path
        '''

        self.num = len(ids)
        self.label = []
        self.image = []

        # These operations may cause large memory and time use.
        # Using database file is recommended.
        for idx in ids:
            target = label_db[idx]
            img_cv = cv2.imread(path.join(image_root, str(idx).rjust(6, '0') + ".jpg"))
            img = transform(img_cv)
            self.label.append(target)
            self.image.append(img)

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, index):
        return self.image[index], self.label[index]