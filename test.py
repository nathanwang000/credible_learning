import torch.multiprocessing as mp
mp.set_start_method('spawn')
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.train import InterpretableTrainer, Trainer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.model import Switch, Weight, apply_linear
from lib.utility import logit_elementwise_loss
from lib.utility import plotDecisionSurface, to_var, to_np, check_nan, onehotize, to_cuda
from lib.utility import genCovX
from lib.model import WeightIndependent
from itertools import product
import os
from sklearn.externals import joblib
from lib.parallel_run import map_parallel

dataname = 'data/100d-9-tasks'
#X, Y, Theta, Xtest, Ytest, test_theta, ndim, n_islands = joblib.load(dataname)
X, Y, Theta, Xtest, Ytest, test_theta, ndim, n_islands = joblib.load('%s.pkl' % dataname)
print('train shape:', X.shape, 'n_islands:', n_islands, 'ndim:', ndim)

#os.environ['GPU_NUM'] = '2'

D = ndim # 2d input
K = 20 # can only output 1 lines
num_workers = 0
pin_memory = True #True

switch = Switch(D, K)
weight = Weight(K, D+1) # +1 for b in linear model

max_grad = None
alpha = -0.01 # -0.5; -0.2 also works
beta = -alpha # 0.7
lr = 0.001
log_name = 'heter/%s/a%.2f/lr%.3f' % (dataname, alpha, lr)
os.system('mkdir -p nonlinear_models/%s' % log_name)
silence = True
t = InterpretableTrainer(switch, weight, apply_linear, alpha=alpha, beta=beta,
                         log_name=log_name, max_grad=max_grad, silence=silence, 
                         lr=lr, max_time=60, print_every=100, plot=False)

# fit a model here:
batch_size = 1000
x = torch.from_numpy(X).float()
y = torch.from_numpy(Y).float()

train_data = TensorDataset(x, y)
data = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                  num_workers=num_workers, pin_memory=pin_memory)

xtest = torch.from_numpy(Xtest).float()
ytest = torch.from_numpy(Ytest).float()

test_data = TensorDataset(xtest, ytest)
test_data = DataLoader(test_data, batch_size=batch_size,
                       num_workers=num_workers, pin_memory=pin_memory)

t.fit(data, n_epochs=15000, valdata=test_data, test_theta=test_theta)
