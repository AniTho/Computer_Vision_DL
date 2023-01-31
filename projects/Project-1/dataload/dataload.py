from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .data_transforms import get_transforms

class UTKdataset:
    def __init__(self, df, transforms, base_image_dir):
        self.df = df
        self.transforms = transforms
        self.images_dir = base_image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item_row = self.df.iloc[idx]
        img_name = item_row['fname']
        age = (item_row['age'] - 1.0 )/(115.0 - 1.0) # Min age: 1, Max age: 115
        gender = item_row['gender']
        img_path = f'{self.images_dir}/{img_name}'
        img = plt.imread(img_path) / 255.0
        if self.transforms:
            img = self.transforms(image = img)['image']
        return img, age, gender
        

def load_data(base_data_dir, images_dir, aug_p = 0.4, batch_size = 64):
    '''
    Creates Dataloaders for training
    Args:
    - base_data_dir(str): Path where entire data is stored (Possibly containing csv file)
    - images_dir (str): Path where all images are stored
    - aug_p (float): Probability of applying a particular augmentation to the image, has to be in [0, 1] range
    - batch_size (int): Size of batches to be formed

    Returns:
    - (Pytorch Dataloader object): Dataloader for train set 
    - (Pytorch Dataloader object): Dataloader for valid set
    - (Pytorch Dataloader object): Dataloader for test set
    '''
    df = pd.read_csv(f'{base_data_dir}/data.csv')
    train_df = df[df['split_type'] == 'train']
    valid_df = df[df['split_type'] == 'valid']
    test_df = df[df['split_type'] == 'test']
    train_transforms, valid_transforms, test_transforms = get_transforms(aug_p=aug_p)
    train_dataset = UTKdataset(train_df, train_transforms, images_dir)
    valid_dataset = UTKdataset(valid_df, valid_transforms, images_dir)
    test_dataset = UTKdataset(test_df, test_transforms, images_dir)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle = True) 
    validloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle = False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle = False)
    return trainloader, validloader, testloader