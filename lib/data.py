import numpy as np
import copy
import joblib
import pathlib
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Mimic2(Dataset):

    def __init__(self, mode='total', random_risk=False,
                 expert_feature_only=False, duplicate=0,
                 dupr=0, # whether to duplicate risk factors
                 two_stage=False, threshold=None):
        '''
        mode in [dead, survivor, total]: ways to impute missingness
        '''
        pwd = pathlib.Path(__file__).parent.absolute()
        self.path = f'{pwd}/../data/mimic/' + mode + '_mean_finalset.csv'
        self.data = pd.read_csv(self.path)
        self.x = self.data.iloc[:,2:]
        
        if expert_feature_only:
            self.r = Variable(torch.FloatTensor(list(map(lambda name: 1 \
                                                         if 'worst' in name else 0,
                                                         self.x.columns))))
            
            self.x = self.x.iloc[:,self.r.nonzero().view(-1)]

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

        # handle duplicate experiment
        for i in range(duplicate):
            self.xtrain = np.hstack([self.xtrain, self.xtrain])
            self.xval = np.hstack([self.xval, self.xval])
            self.xte = np.hstack([self.xte, self.xte])
            self.r = torch.cat([self.r, self.r])

        # use training data to delete variables for two stage approach
        risk_set = set(self.r.nonzero().view(-1).data.numpy())
        if two_stage:
            kept = set(self.r.nonzero().view(-1).data.numpy())
            corr = np.corrcoef(self.xtrain.T)
            d = self.r.numel()
            for i in range(d):
                if i not in kept: # unknown variable
                    # determine if we should keep it
                    keep = True
                    for j in range(len(corr[i])):
                        if j in kept and corr[i, j] >= threshold:
                            keep = False
                            break
                    if keep:
                        kept.add(i) # later unknown correlated with this will not keep

            # organize the data and risk
            kept = list(kept)
            self.r = Variable(torch.FloatTensor([(1 if i in risk_set else 0)\
                                                 for i in kept]))
            self.xtrain = self.xtrain[:, kept]
            self.xval = self.xval[:, kept]
            self.xte = self.xte[:, kept]

        # save for later manipulation of the input
        self.xtrain_ = copy.deepcopy(self.xtrain)
        self.xval_ = copy.deepcopy(self.xval)
        self.xte_ = copy.deepcopy(self.xte)
        self.r_ = copy.deepcopy(self.r)

        # duplicate risk factors after the original data are saved
        self.dupr = dupr
        for i in range(self.dupr):
            self._dupr()

    def related_to_c(self):
        '''return a binary vector that is related to c in train'''
        def relate(x, c, threshold=0.8, dropout=False):
            '''x is a numpy array of features (n, 1), c:(n,sum(r))'''
            if dropout: return np.random.uniform(0,1) < threshold
            
            a = np.hstack([x, c]) # (n, 1+sum(r))
            corr = np.corrcoef(a.T) # (1+sum(r), 1+sum(r))
            for i in range(1, len(a[0])):
                if corr[0, i] > threshold:
                    return True
            return False

        if not hasattr(self, 'remove_indices'):
            print('constructing remove indices')

            remove_indices = []
            c = self.xtrain_[:, self.r_==1]
            for idx in range(len(self.r_)):
                if self.r_[idx] == 1: continue # don't remove c
                # remove if correlation too high with c
                if relate(self.xtrain_[:, [idx]], c):
                    remove_indices.append(idx)

            self.remove_indices = remove_indices

        print(f"{len(self.remove_indices)} features to be removed")
        return self.remove_indices
            
        # return self.r_ == 0
    
    def clean(self):
        '''applies to the shortcut experiment, set shortcut to 0 b/c standardized'''
        # remove variables in which C can predict with high accuracy
        def set_x(x):
            x = copy.deepcopy(x)
            x[:, self.related_to_c()] = 0
            return x

        self.xtrain = set_x(self.xtrain_)
        self.xval = set_x(self.xval_)
        self.xte = set_x(self.xte_)        
        self.r = self.r_
        
        # dr = int(self.r_.sum().item())        
        # self.xtrain = np.hstack([self.xtrain_, torch.zeros(len(self.xtrain_), dr)])
        # self.xval = np.hstack([self.xval_, torch.zeros(len(self.xval_), dr)])
        # self.xte = np.hstack([self.xte_, torch.zeros(len(self.xte_), dr)])
        # self.r = torch.cat([self.r_, torch.zeros(dr)])
        
        for i in range(self.dupr):
            self._dupr()

    def bias(self):
        '''applies to the shortcut experiment'''
        self.xtrain = self.xtrain_
        self.xval = self.xval_
        self.xte = self.xte_
        self.r = self.r_
        
        # dr = int(self.r_.sum().item())        
        # self.xtrain = np.hstack([self.xtrain_, self.xtrain_[:, self.r_==1]])
        # self.xval = np.hstack([self.xval_, self.xval_[:, self.r_==1]])
        # self.xte = np.hstack([self.xte_, self.xte_[:, self.r_==1]])
        # self.r = torch.cat([self.r_, torch.zeros(dr)])
        
        for i in range(self.dupr):
            self._dupr()

    def _dupr(self):
        '''duplicate risk factors for STD(C, X) experiments'''
        dr = int(self.r.sum().item())        
        self.xtrain = np.hstack([self.xtrain, self.xtrain[:, self.r==1]])
        self.xval = np.hstack([self.xval, self.xval[:, self.r==1]])
        self.xte = np.hstack([self.xte, self.xte[:, self.r==1]])
        self.r = torch.cat([self.r, torch.ones(dr)])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.xtrain[idx], self.ytrain[idx]

