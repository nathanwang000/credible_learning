import numpy as np
import sys
import torch
from torch import nn
from torch.autograd import Variable
from lib.model import LR
from torch.utils.data import DataLoader
import time, math
from lib.utility import timeSince, data_shuffle, model_auc, calc_loss, model_acc
from sklearn.metrics import accuracy_score

def np2tensor(x, y):
    return torch.from_numpy(x).float(), torch.from_numpy(y).long()

def prepareData(x, y):
    ''' 
    convert x, y from numpy to tensor
    '''
    return Variable(torch.from_numpy(x).float()), Variable(torch.from_numpy(y).long())

class Trainer(object):
    def __init__(self, model, optimizer=None,
                 loss=None, name="m",
                 lr=0.001, alpha=0.001,
                 risk_factors=None,
                 regularization=None,
                 reg_parameters=None):
        '''
        optimizer: optimization method, default to adam
        reg_parameters: parameters to regularize on
        '''
        self.model = model
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)   
        self.optimizer = optimizer
        if loss is None:
            loss = nn.NLLLoss()
        self.loss = loss
        self.name = name
        if regularization is not None and\
           alpha is not None and\
           reg_parameters is not None: # e.g., eye_loss
            self.loss = regularization(loss, alpha, reg_parameters, risk_factors)

    def step(self, x, y):
        '''
        one step of training
        return yhat, regret
        '''
        self.optimizer.zero_grad()
        yhat = self.model(x)
        regret = self.loss(yhat, y)
        regret.backward()
        self.optimizer.step()
        return yhat, regret.data[0]

    def fitData(self, data, batch_size=100, n_epochs=10, print_every=10,
                valdata=None):
        '''
        fit a model to x, y data by batch
        '''
        time_start = time.time()
        losses = []
        vallosses = []
        n = len(data.dataset)
        cost = 0 
        
        for epoch in range(n_epochs):
            for k, (x_batch, y_batch) in enumerate(data):
                x_batch, y_batch = Variable(x_batch), Variable(y_batch)
                y_hat, regret = self.step(x_batch, y_batch)
                m = x_batch.size(0)                
                cost += 1 / (k+1) * (regret/m - cost)
                
                if k % print_every == 0:
                    losses.append(cost)
                    # progress, time, avg loss, auc
                    to_print = ('%.2f%% (%s) %.4f %.4f' % ((epoch * n + (k+1) * m) /
                                                           (n_epochs * n) * 100,
                                                           timeSince(time_start),
                                                           cost,
                                                           model_auc(self.model,
                                                                     data)))
                    if valdata is not None:
                        vallosses.append(calc_loss(self.model, valdata, self.loss))
                        to_print += "%.4f" % model_auc(self.model, valdata)
                        
                    print(to_print)
                    torch.save(self.model, 'models/%s.pt' % self.name)
                    np.save('models/%s.loss' % self.name, losses)
                    cost = 0
        return losses, vallosses

    def fitXy(self, x, y, batch_size=100, n_epochs=10, print_every=10,
              valdata=None):
        '''
        fit a model to x, y data by batch
        '''
        n, d = x.shape
        time_start = time.time()
        losses = []
        cost = 0
        
        for epoch in range(n_epochs):
            x, y = data_shuffle(x, y)

            num_batches = math.ceil(n / batch_size)

            for k in range(num_batches):
                start, end = k * batch_size, min((k + 1) * batch_size, n)
                x_batch, y_batch = prepareData(x[start:end], y[start:end])
                y_hat, regret = self.step(x_batch, y_batch)
                m = end-start
                cost += 1 / (k+1) * (regret/m - cost)
                
                if k % print_every == 0:
                    losses.append(cost)
                    print('%.2f%% (%s) %.4f' % ((epoch * n + (k+1) * (end-start)) /
                                                (n_epochs * n) * 100, # progress
                                                timeSince(time_start), # time 
                                                cost)) # cost
                    torch.save(self.model, 'models/%s.pt' % self.name)
                    np.save('models/%s.loss' % self.name, losses)
        return losses
                    
    def fit(self, x, y=None, batch_size=100, n_epochs=10, print_every=10,
            valdata=None):
        if y is None:
            return self.fitData(x, batch_size, n_epochs, print_every, valdata)
        else:
            return self.fitXy(x, y, batch_size, n_epochs, print_every, valdata)
            
