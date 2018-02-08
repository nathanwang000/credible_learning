import torch.multiprocessing as mp
# mp.set_start_method('spawn')
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.train import InterpretableTrainer, Trainer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.model import Switch, Weight, apply_linear, MLP, AutoEncoder, WeightIndependent
from lib.utility import logit_elementwise_loss
from lib.utility import plotDecisionSurface, to_var, to_np, check_nan, onehotize, to_cuda
from lib.utility import genCovX, loadData
from itertools import product
import os
from sklearn.externals import joblib
from lib.parallel_run import map_parallel
from lib.train import KmeansTrainer, WeightNetTrainer, CombineTrainer, MLPTrainer
from lib.train import AutoEncoderTrainer
import argparse

def parse_main_args():
    parser = argparse.ArgumentParser(description="dlln training")
    parser.add_argument('-d',help='dataname', default='100d-8-tasks')
    parser.add_argument('-k',help='num clusters', type=int, default=20)
    parser.add_argument('-b', help='baseline number', type=int, default=0)
    parser.add_argument('-s', help='batch size', type=int, default=1000)    
    parser.add_argument('-t', help='max run time', type=int, default=60)    
    parser.add_argument('-a', help='joint entropy coefficient',
                        type=float, default=-0.01)
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

###################### methods ##################
def deep_locally_linear_trainer(dataname, D, K, alpha=-0.01, lr=0.001,
                                max_time=60, switch=None, weight=None,
                                batch_size=1000):
    ''' D is data dimension, K is number of clusters '''
    if switch is None:
        switch = Switch(D, K)
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model

    max_grad = None
    beta = -alpha
    log_name = 'heter/%s/dlln/k%d/a%.4f/lr%.3f/bs%d' % (dataname, K, alpha, lr,
                                                        batch_size)
    silence = True
    t = InterpretableTrainer(switch, weight, apply_linear, alpha=alpha, beta=beta,
                             log_name=log_name, max_grad=max_grad, silence=silence, 
                             lr=lr, max_time=max_time, print_every=100, plot=False)
    return t

def kmeans_plus_weightnet_trainer(dataname, D, K, lr=0.001, max_time=60,
                                  weight=None, kmeans_clf=None, batch_size=1000):
    ''' D is data dimension, K is number of clusters '''
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model
        
    log_name = 'heter/%s/kmeans_weight/k%d/lr%.3f/bs%d' % (dataname, K, lr,
                                                           batch_size)

    ### trainers
    kmeans_trainer = KmeansTrainer(K, clf=kmeans_clf)    
    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time)

    combined_trainer = CombineTrainer(log_name=log_name)
    combined_trainer.add(kmeans_trainer)
    combined_trainer.add(weightnet_trainer)
    return combined_trainer

def weightnet_trainer(dataname, D, K=None, lr=0.001, max_time=60,
                      weight=None, batch_size=1000):
    ''' D is data dimension, K is number of clusters '''
    K = D
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/%s/weight/lr%.3f/bs%d' % (dataname, lr, batch_size)

    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time,
                                         log_name=log_name)

    return weightnet_trainer

def mlp_trainer(dataname, D, K=None, lr=0.001, max_time=60,
                model=None, batch_size=1000):
    ''' D is data dimension, K is number of clusters '''
    if model is None:
        model = MLP(D)

    log_name = 'heter/%s/mlp/lr%.3f/bs%d' % (dataname, lr, batch_size)

    t = MLPTrainer(model, lr=lr, max_time=max_time,
                   log_name=log_name)

    return t

def linear_trainer(dataname, D, K=None, lr=0.001, max_time=60,
                   model=None, batch_size=1000):
    ''' D is data dimension, K is number of clusters '''
    if model is None:
        model = nn.Linear(D, 1)

    log_name = 'heter/%s/linear/lr%.3f/bs%d' % (dataname, lr, batch_size)

    t = MLPTrainer(model, lr=lr, max_time=max_time,
                   log_name=log_name, islinear=True)

    return t

def autoencoder_plus_weightnet_trainer(dataname, D, K, lr=0.001, max_time=60,
                                       autoencoder=None, weight=None,
                                       batch_size=1000):
    ''' D is data dimension, K is number of clusters '''
    if autoencoder is None:
        autoencoder = AutoEncoder(D, K)
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/%s/auto_plus_weight/k%d/lr%.3f/bs%d' % (dataname, K, lr,
                                                              batch_size)

    ### trainers
    auto_trainer = AutoEncoderTrainer(autoencoder, lr=lr, max_time=max_time)
    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time)

    combined_trainer = CombineTrainer(log_name=log_name)
    combined_trainer.add(auto_trainer)
    combined_trainer.add(weightnet_trainer)
    return combined_trainer

def autoencoder_plus_kmeans_plus_weightnet_trainer(dataname, D, K,
                                                   lr=0.001, max_time=60,
                                                   autoencoder=None,
                                                   weight=None,
                                                   kmeans_clf=None,
                                                   batch_size=1000):
    ''' D is data dimension, K is number of clusters '''
    if autoencoder is None:
        autoencoder = AutoEncoder(D, K)
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/%s/auto_plus_kmeans_plus_weight/k%d/lr%.3f/bs%d' % (dataname,
                                                                          K, lr,
                                                                          batch_size)

    ### trainers
    kmeans_trainer = KmeansTrainer(K, clf=kmeans_clf)            
    auto_trainer = AutoEncoderTrainer(autoencoder, lr=lr, max_time=max_time)
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
    batch_size = args.s   
    train_data, val_data,\
        train_theta, val_theta,\
        ndim, n_islands = loadData(dataname, batch_size=batch_size)

    D = ndim
    K = args.k
    alpha = args.a
    max_time = args.t
    
    if args.b == 0:
        t = deep_locally_linear_trainer(dataname, D, K, alpha=alpha,
                                        max_time=max_time,
                                        batch_size=batch_size)
    elif args.b == 1:
        # should outperform this in acc
        t = autoencoder_plus_kmeans_plus_weightnet_trainer(dataname, D, K,
                                                           max_time=max_time,
                                                           batch_size=batch_size)
    elif args.b == 2:
        # should outperform this in acc
        t = autoencoder_plus_weightnet_trainer(dataname, D, K,
                                               max_time=max_time,
                                               batch_size=batch_size)
    elif args.b == 3:
        # should outperform this in acc        
        t = kmeans_plus_weightnet_trainer(dataname, D, K,
                                          max_time=max_time,
                                          batch_size=batch_size)
    elif args.b == 4:
        # should outperform this in cosine distance        
        t = weightnet_trainer(dataname, D, K,
                              max_time=max_time,
                              batch_size=batch_size)
    elif args.b == 5:
        # should be at least close to this
        t = mlp_trainer(dataname, D, K,
                        max_time=max_time,
                        batch_size=batch_size)
    elif args.b == 6:
        # should outperform this by a large margin
        t = linear_trainer(dataname, D, K, max_time=max_time,
                           batch_size=batch_size)

    t.fit(train_data, n_epochs=15000, valdata=val_data, val_theta=val_theta)


