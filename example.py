''' 
example usage of this package
'''
import numpy as np
from lib.model import LR
from lib.train import Trainer, prepareData
from lib.utility import to_var, to_np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.regularization import eye_loss
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable
from torch.optim import SGD
import os

def model_acc(model, x, y):                                          
    x, _ = prepareData(x, y)                                                   
    yhat = np.argmax(to_np(model(x)), 1)                                       
    return accuracy_score(y, yhat)

n, d = 1000, 2  

def gendata(): 
    a = np.random.randn(n)
    x = np.vstack([a, a]).T # perfectly correlated data
    y = (x.sum(1) > 0).astype(int)
    return x, y

xtr, ytr = gendata()
xte, yte = gendata()

r = to_var(torch.FloatTensor([0, 1]))
train_data = TensorDataset(*map(lambda x: x.data, prepareData(xtr, ytr)))
data = DataLoader(train_data, batch_size=100, shuffle=True)

n_output = 2 # binary classification task         
model = LR(d, n_output)            
learning_rate = 0.01
alpha = 0.1  # regularization strength                                  

reg_parameters = model.i2o.weight
t = Trainer(model, optimizer=SGD(model.parameters(), lr=learning_rate),
            lr=learning_rate, risk_factors=r, alpha=alpha,
            regularization=eye_loss, reg_parameters=reg_parameters)  
t.fit(data, n_epochs=100, print_every=100)

print('done fitting model')
print("train accuracy", model_acc(model, xtr, ytr))
print("test accuracy", model_acc(model, xte, yte))
print('r', list(to_np(r)), ', weight', list(to_np(reg_parameters[1] - reg_parameters[0])))
