import numpy as np
# from sklearn.externals import joblib
import joblib
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Mimic2(Dataset):

    def __init__(self, mode='dead', random_risk=False, expert_feature_only=False):
        '''
        mode in [dead, survivor, total]: ways to impute missingness
        '''
        self.path = 'data/mimic/' + mode + '_mean_finalset.csv'
        self.data = pd.read_csv(self.path)
        self.x = self.data.iloc[:,2:] # ix deprecated
        if expert_feature_only:
            self.x = self.x.iloc[:,-16:] # ix deprecated
        self.y = self.data['In-hospital_death']

        self.xtr, self.xte, self.ytr, self.yte = train_test_split(self.x,
                                                                  self.y,
                                                                  test_size=0.25,
                                                                  random_state=42,
                                                                  stratify=self.y)

        self.xtrain,self.xval,self.ytrain,self.yval = train_test_split(self.xtr,
                                                                       self.ytr,
                                                                       test_size=0.25,
                                                                       random_state=42,
                                                                       stratify=self.ytr)

        # standardize model
        scaler = StandardScaler()
        scaler.fit(self.xtrain)
        self.xtrain = scaler.transform(self.xtrain)
        self.xval = scaler.transform(self.xval)
        self.xte = scaler.transform(self.xte)  
        # self.ytrain = self.ytrain.as_matrix()
        # self.yval = self.yval.as_matrix()
        # self.yte = self.yte.as_matrix()        
        self.ytrain = np.array(self.ytrain)
        self.yval = np.array(self.yval)
        self.yte = np.array(self.yte)

        # risk factors to use
        self.r = Variable(torch.FloatTensor(list(map(lambda name: 1 \
                                                     if 'worst' in name else 0,
                                                     self.x.columns))))
        if random_risk:
            np.random.seed(42)
            r = np.random.permutation(self.r.data.numpy())            
            self.r = Variable(torch.from_numpy(r))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.xtrain[idx], self.ytrain[idx]

