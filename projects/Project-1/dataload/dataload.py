from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .data_transforms import get_transforms

class UTKdataset:
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df

def load_data():
    aug_p = 0.4
    df = pd.read_csv('data/UTKFace/data.csv')
    train_df = df[df['split_type'] == 'train']
    valid_df = df[df['split_type'] == 'valid']
    test_df = df[df['split_type'] == 'test']
    train_transforms, valid_transforms, test_transforms = get_transforms(aug_p=aug_p)
    return None, None, None