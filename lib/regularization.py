import numpy as np
import sys
import torch
from torch import nn
from torch.autograd import Variable

def ridge(loss, alpha, theta, r=None):
    def reg(x):
        nonlocal r # default to all unknown
        r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        weight = 2 - r
        return 0.5 * (x).dot(x)

    def ret(yhat, y):
        return loss(yhat, y) + reg(theta[1] - theta[0]) * alpha
    
    return ret

def wridge(loss, alpha, theta, r=None):
    def reg(x): # weighted ridge, weight 2 for unknown, 1 for known
        nonlocal r # default to all unknown
        r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        weight = 2 - r
        return 0.5 * (x * weight).dot(x * weight)

    def ret(yhat, y):
        return loss(yhat, y) + reg(theta[1] - theta[0]) * alpha
    
    return ret

def lasso(loss, alpha, theta, r=None):
    def reg(x):
        nonlocal r # default to all unknown
        r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        return torch.abs(x).sum()

    def ret(yhat, y):
        return loss(yhat, y) + reg(theta[1] - theta[0]) * alpha
    
    return ret

def wlasso(loss, alpha, theta, r=None):
    def reg(x):  # weighted lasso, weight 2 for unknown, 1 for known
        nonlocal r # default to all unknown
        r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        weight = 2 - r
        return torch.abs(x * weight).sum()

    def ret(yhat, y):
        return loss(yhat, y) + reg(theta[1] - theta[0]) * alpha
    
    return ret

def owl(loss, alpha, theta, r=None):
    # the infinity norm formulation    
    weight = Variable(torch.zeros(theta[1].numel()))
    weight.data[-1] = 1 # because order is sorted ascending

    def reg(x):
        nonlocal r # default to all unknown
        r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        order = torch.from_numpy(np.argsort(x.abs().data.numpy())).long()
        return (weight * x.abs()[order]).sum()

    def ret(yhat, y):
        return loss(yhat, y) + reg(theta[1] - theta[0]) * alpha
    
    return ret

def enet(loss, alpha, theta, r=None, l1_ratio=0.5):
    def reg(x): # weighted ridge, weight 2 for unknown, 1 for known
        nonlocal r # default to all unknown
        r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        return l1_ratio * torch.abs(x).sum() + (1-l1_ratio) * 0.5 * x.dot(x)

    def ret(yhat, y):
        return loss(yhat, y) + reg(theta[1] - theta[0]) * alpha
    
    return ret

def eye_loss(loss, alpha, theta, r=None): # loss is the data loss

    def eye(x):
        nonlocal r # default to all unknown
        r = r or Variable(torch.zeros(x.numel()), requires_grad=False)
        l1 = torch.abs(x * (1-r)).sum()
        l2sq = (r * x).dot(r * x)
        return  l1 + torch.sqrt(l1**2 + l2sq)

    def ret(yhat, y):
        return loss(yhat, y) + eye(theta[1] - theta[0]) * alpha
    
    return ret
