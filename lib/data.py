import numpy as np
from sklearn.externals import joblib
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class Mimic2(Dataset):

    def __init__(self, mode='dead'):
        '''
        mode in [dead, survivor, total]: ways to impute missingness
        '''
        self.path = 'data/mimic/' + mode + '_mean_finalset.csv'
        self.data = pd.read_csv(self.path)
        self.x = self.data.ix[:,2:]
        self.y = self.data['In-hospital_death']
        self.xtrain, self.xval, self.ytrain, self.yval= train_test_split(self.x,
                                                                         self.y,
                                                                         test_size=0.25,
                                                                         random_state=42,
                                                                         stratify=self.y)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.xtrain[idx], self.ytrain[idx]

