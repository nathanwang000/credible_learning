import torch.multiprocessing as mp
mp.set_start_method('spawn')
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.train import InterpretableTrainer, Trainer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.model import Switch, Weight, apply_linear, MLP, AutoEncoder
from lib.utility import logit_elementwise_loss
from lib.utility import plotDecisionSurface, to_var, to_np, check_nan, onehotize, to_cuda
from lib.utility import genCovX
from itertools import product
import os
from sklearn.externals import joblib
from lib.parallel_run import map_parallel
from lib.train import KmeansTrainer, WeightNetTrainer, CombineTrainer, MLPTrainer
from lib.train import AutoEncoderTrainer
import argparse

def parse_main_args():
    parser = argparse.ArgumentParser(description="dlln training")
    parser.add_argument('-d',help='dataname', default='100d-9-tasks')
    parser.add_argument('-k',help='num clusters', type=int, default=20)
    parser.add_argument('-b', help='baseline number', type=int, default=0)
    return parser
                                
def recover_subtasks(X, Y, Theta):
    '''Theta comes from build training data'''
    tasks = []
    i = 0
    for _, _, _, n_per_island, _, _ in Theta:
        x = X[i: i+n_per_island]
        y = Y[i: i+n_per_island]
        i = i + n_per_island
        tasks.append((x, y))
    return tasks

def loadData(dataname):
    # note: data are generated in heterogeneous groups ipython notebook
    X, Y, Theta, Xtest, Ytest, test_theta, ndim, n_islands = joblib.load('data/%s.pkl' % dataname)
    num_workers = 0
    pin_memory = True 
    batch_size = 1000

    x = torch.from_numpy(X).float()
    y = torch.from_numpy(Y).float()

    train_data = TensorDataset(x, y)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory)

    xtest = torch.from_numpy(Xtest).float()
    ytest = torch.from_numpy(Ytest).float()

    test_data = TensorDataset(xtest, ytest)
    test_data = DataLoader(test_data, batch_size=batch_size,
                           num_workers=num_workers, pin_memory=pin_memory)

    print('train shape:', X.shape, 'n_islands:', n_islands, 'ndim:', ndim)
    print('done loading data')
    return train_data, test_data, ndim, n_islands, test_theta

###################### methods ##################
def deep_locally_linear_trainer(dataname, D, K, alpha=-0.01, lr=0.001, max_time=60):
    ''' D is data dimension, K is number of clusters '''
    switch = Switch(D, K)
    weight = Weight(K, D+1) # +1 for b in linear model

    max_grad = None
    beta = -alpha
    log_name = 'heter/dlln/%s/a%.2f/lr%.3f' % (dataname, alpha, lr)
    os.system('mkdir -p nonlinear_models/%s' % log_name)
    silence = True
    t = InterpretableTrainer(switch, weight, apply_linear, alpha=alpha, beta=beta,
                             log_name=log_name, max_grad=max_grad, silence=silence, 
                             lr=lr, max_time=max_time, print_every=100, plot=False)
    return t

def kmeans_plus_weightnet_trainer(dataname, D, K, lr=0.001, max_time=60):
    ''' D is data dimension, K is number of clusters '''
    weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/kmeans_weight/%s/lr%.3f' % (dataname, lr)
    os.system('mkdir -p nonlinear_models/%s' % log_name)

    ### trainers
    kmeans_trainer = KmeansTrainer(K)
    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time)

    combined_trainer = CombineTrainer(log_name=log_name)
    combined_trainer.add(kmeans_trainer)
    combined_trainer.add(weightnet_trainer)
    return combined_trainer

def weightnet_trainer(dataname, D, K=None, lr=0.001, max_time=60):
    ''' D is data dimension, K is number of clusters '''
    K = D
    weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/weight/%s/lr%.3f' % (dataname, lr)
    os.system('mkdir -p nonlinear_models/%s' % log_name)

    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time,
                                         log_name=log_name)

    return weightnet_trainer

def mlp_trainer(dataname, D, K=None, lr=0.001, max_time=60):
    ''' D is data dimension, K is number of clusters '''
    model = MLP(D)

    log_name = 'heter/mlp/%s/lr%.3f' % (dataname, lr)
    os.system('mkdir -p nonlinear_models/%s' % log_name)

    t = MLPTrainer(model, lr=lr, max_time=max_time,
                   log_name=log_name)

    return t

def linear_trainer(dataname, D, K=None, lr=0.001, max_time=60):
    ''' D is data dimension, K is number of clusters '''
    model = nn.Linear(D, 1)

    log_name = 'heter/linear/%s/lr%.3f' % (dataname, lr)
    os.system('mkdir -p nonlinear_models/%s' % log_name)

    t = MLPTrainer(model, lr=lr, max_time=max_time,
                   log_name=log_name)

    return t

def autoencoder_plus_weightnet_trainer(dataname, D, K, lr=0.001, max_time=60):
    ''' D is data dimension, K is number of clusters '''
    autoencoder = AutoEncoder(D, K)
    weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/auto_plus_weight/%s/lr%.3f' % (dataname, lr)
    os.system('mkdir -p nonlinear_models/%s' % log_name)

    ### trainers
    auto_trainer = AutoEncoderTrainer(autoencoder, lr=lr, max_time=max_time)
    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time)

    combined_trainer = CombineTrainer(log_name=log_name)
    combined_trainer.add(auto_trainer)
    combined_trainer.add(weightnet_trainer)
    return combined_trainer

def autoencoder_plus_kmeans_plus_weightnet_trainer(dataname, D, K,
                                                   lr=0.001, max_time=60):
    ''' D is data dimension, K is number of clusters '''
    autoencoder = AutoEncoder(D, K)
    weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/auto_plus_kmeans_plus_weight/%s/lr%.3f' % (dataname, lr)
    os.system('mkdir -p nonlinear_models/%s' % log_name)

    ### trainers
    auto_trainer = AutoEncoderTrainer(autoencoder, lr=lr, max_time=max_time)
    kmeans_trainer = KmeansTrainer(K)    
    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time)

    combined_trainer = CombineTrainer(log_name=log_name)
    combined_trainer.add(auto_trainer)
    combined_trainer.add(kmeans_trainer)    
    combined_trainer.add(weightnet_trainer)
    return combined_trainer

if __name__ == '__main__':
    parser = parse_main_args()
    args = parser.parse_args()
    
    dataname = args.d 
    train_data, test_data, ndim, n_islands, test_theta = loadData(dataname)

    D = ndim
    K = args.k

    if args.b == 0:
        t = deep_locally_linear_trainer(dataname, D, K, max_time=180)
    elif args.b == 1:
        # should outperform this in acc
        autoencoder_plus_kmeans_plus_weightnet_trainer(dataname, D, K)
    elif args.b == 2:
        # should outperform this in acc
        autoencoder_plus_weightnet_trainer(dataname, D, K)
    elif args.b == 3:
        # should outperform this in acc        
        kmeans_plus_weightnet_trainer(dataname, D, K)
    elif args.b == 4:
        # should outperform this in cosine distance        
        weightnet_trainer(dataname, D, K)
    elif args.b == 5:
        # should be at least close to this
        mlp_trainer(dataname, D, K)
    elif args.b == 6:
        # should outperform this by a large margin
        linear_trainer(dataname, D, K)

    t.fit(train_data, n_epochs=15000, valdata=test_data, test_theta=test_theta)


