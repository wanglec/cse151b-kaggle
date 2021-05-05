import os
import pickle
from glob import glob
import matplotlib.pyplot as plt

# pytorch imports
from torch.utils.data import Dataset, DataLoader

# config variable
train_path = "/Users/kailing/Desktop/UCSD academics/CSE 151B/kaggle/new_train/new_train"
val_path = "/Users/kailing/Desktop/UCSD academics/CSE 151B/kaggle/new_val_in/new_val_in"
submission_path = "/Users/kailing/Desktop/UCSD academics/CSE 151B/kaggle/sample_submission.csv"
submission_dir = "/Users/kailing/Desktop/UCSD academics/CSE 151B/kaggle/submissions"


class ArgoverseDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform

        self.pkl_list = glob(os.path.join(self.data_path, '*'))
        self.pkl_list.sort()
        
    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):

        pkl_path = self.pkl_list[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if self.transform:
            data = self.transform(data)

        return data

def plot_one_scene(scene: dict):
    pass