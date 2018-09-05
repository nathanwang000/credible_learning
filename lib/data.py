import numpy as np
from sklearn.externals import joblib
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Mimic2(Dataset):

    def __init__(self, mode='total', random_risk=False,
                 expert_feature_only=False, duplicate=0):
        '''
        mode in [dead, survivor, total]: ways to impute missingness
        '''
        self.path = 'data/mimic/' + mode + '_mean_finalset.csv'
        self.data = pd.read_csv(self.path)
        self.x = self.data.ix[:,2:]
        if expert_feature_only:
            self.x = self.x.ix[:,-16:]
        self.y = self.data['In-hospital_death']

        xtrain, self.xte, ytrain, self.yte = train_test_split(self.x,
                                                              self.y,
                                                              test_size=0.25,
                                                              random_state=42,
                                                              stratify=self.y)

        self.xtrain,self.xval,self.ytrain,self.yval = train_test_split(xtrain,
                                                                       ytrain,
                                                                       test_size=0.25,
                                                                       random_state=42,
                                                                       stratify=ytrain)


        # standardize model
        scaler = StandardScaler()
        scaler.fit(self.xtrain)
        self.xtrain = scaler.transform(self.xtrain)
        self.xval = scaler.transform(self.xval)
        self.xte = scaler.transform(self.xte)  
        self.ytrain = self.ytrain.values
        self.yval = self.yval.values
        self.yte = self.yte.values        

        # risk factors to use
        self.r = Variable(torch.FloatTensor(list(map(lambda name: 1 \
                                                     if 'worst' in name else 0,
                                                     self.x.columns))))
        if random_risk:
            np.random.seed(42)
            r = np.random.permutation(self.r.data.numpy())            
            self.r = Variable(torch.from_numpy(r))

        for i in range(duplicate):
            self.xtrain = np.hstack([self.xtrain, self.xtrain])
            self.xval = np.hstack([self.xval, self.xval])
            self.xte = np.hstack([self.xte, self.xte])
            self.r = torch.cat([self.r, self.r])
                    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.xtrain[idx], self.ytrain[idx]

