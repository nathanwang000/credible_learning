''' 
example usage of this package
'''
import numpy as np
from lib.model import LR
from lib.train import Trainer, prepareData
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.regularization import eye_loss
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable

def model_acc(model, x, y):                                                                                                                                                        
    x, _ = prepareData(x, y)                                                                                                                                                       
    yhat = np.argmax(model(x).data.numpy(), 1)                                                                                                                                     
    return accuracy_score(y, yhat)                                                                                                                                                 

n, d = 1000, 2                                                                                                                                                                     

def gendata():                                                                                                                                                                     
    x = np.random.randn(n, d)                                                                                                                                                      
    y = (x.sum(1) > 0).astype(np.int)                                                                                                                                              
    return x, y                                                                                                                                                                    

xtr, ytr = gendata()                                                                                                                                                               
xte, yte = gendata()                                                                                                                                                               

r = Variable(torch.FloatTensor([0, 1]))
train_data = TensorDataset(*map(lambda x: x.data, prepareData(xtr, ytr)))
data = DataLoader(train_data, batch_size=100, shuffle=True)

n_output = 2 # binary classification task                                                                                                                                          
model = LR(d, n_output)                                                                                                                                                            
learning_rate = 0.01                                                                                                                                                               

reg_parameters = model.i2o.weight
t = Trainer(model, lr=learning_rate, risk_factors=r, alpha=0.08,
            regularization=eye_loss, reg_parameters=reg_parameters)  
t.fit(data, n_epochs=60, print_every=50)

print('done fitting model')
print("train accuracy", model_acc(model, xtr, ytr))
print("test accuracy", model_acc(model, xte, yte))
print('r', list(r.data.numpy()), ', weight', list((reg_parameters[1] - reg_parameters[0]).data.numpy()))
