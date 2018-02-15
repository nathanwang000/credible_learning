import torch.multiprocessing as mp
# mp.set_start_method('spawn')
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.train import InterpretableTrainer, Trainer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.model import Switch, Weight, WeightIndependent, apply_linear, MLP, AutoEncoder
from lib.model import WeightIndependent
from lib.utility import logit_elementwise_loss
from lib.utility import plotDecisionSurface, to_var, to_np, check_nan, onehotize, to_cuda
from lib.utility import genCovX, loadData
from itertools import product
import os
from sklearn.externals import joblib
from lib.parallel_run import map_parallel
from lib.train import KmeansTrainer, PosKmeansTrainer, WeightNetTrainer, CombineTrainer
from lib.train import AutoEncoderTrainer, MLPTrainer
import argparse

def parse_main_args():
    parser = argparse.ArgumentParser(description="dlln training")
    parser.add_argument('-d',help='dataname', default='100d-8-tasks')
    parser.add_argument('-c',help='cuda device', type=int, default=0)
    parser.add_argument('-k',help='num clusters', type=int, default=20)
    parser.add_argument('-b', help='baseline number', type=int, default=0)
    parser.add_argument('-s', help='batch size', type=int, default=1000)    
    parser.add_argument('-t', help='max run time', type=int, default=60)    
    parser.add_argument('-a', help='joint entropy coefficient',
                        type=float, default=-0.01)
    parser.add_argument('-f', help='switch net update frequency',
                        type=int, default=30)
    parser.add_argument('-w', help='switch net update frequency',
                        type=float, default=0)
    
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
                                batch_size=1000, weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    if switch is None:
        switch = Switch(D, K)
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model

    max_grad = None
    beta = -alpha
    log_name = 'heter/%s/dlln/k%d/a%.4f/lr%.3f/bs%d/wd%.5f' % (dataname, K, alpha, lr,
                                                             batch_size, weight_decay)
    silence = True
    t = InterpretableTrainer(switch, weight, apply_linear, alpha=alpha, beta=beta,
                             log_name=log_name, max_grad=max_grad, silence=silence, 
                             lr=lr, max_time=max_time, print_every=100, plot=False,
                             weight_decay=weight_decay)
    return t

def optimal_weight_trainer(dataname, D, K, alpha=-0.01, lr=0.001,
                           max_time=60, switch=None, weight=None,
                           batch_size=1000, switch_update_every=30,
                           weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    if switch is None:
        switch = Switch(D, K)
    if weight is None:
        weight = WeightIndependent(K, D+1) # +1 for b in linear model

    max_grad = None
    beta = -alpha
    log_name = 'heter/%s/optweight/k%d/a%.4f/lr%.3f/bs%d/fq%d/wd%.5f' % \
               (dataname,
                K, alpha, lr,
                batch_size,
                switch_update_every,
                weight_decay)
    silence = True
    t = InterpretableTrainer(switch, weight, apply_linear, alpha=alpha, beta=beta,
                             log_name=log_name, max_grad=max_grad, silence=silence, 
                             lr=lr, max_time=max_time, print_every=100, plot=False,
                             switch_update_every=switch_update_every,
                             n_early_stopping=3000,
                             weight_decay=weight_decay)
    return t

def mixture_of_expert_trainer(dataname, D, K, alpha=-0.01, lr=0.001,
                              max_time=60, switch=None, weight=None,
                              batch_size=1000, weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    if switch is None:
        switch = Switch(D, K)
    if weight is None:
        weight = WeightIndependent(K, D+1) # +1 for b in linear model

    max_grad = None
    beta = -alpha
    log_name = 'heter/%s/moe/k%d/a%.4f/lr%.3f/bs%d/wd%.5f' % (dataname, K, alpha, lr,
                                                              batch_size, weight_decay)
    silence = True
    t = InterpretableTrainer(switch, weight, apply_linear, alpha=alpha, beta=beta,
                             log_name=log_name, max_grad=max_grad, silence=silence, 
                             lr=lr, max_time=max_time, print_every=100, plot=False,
                             weight_decay=weight_decay)
    return t


def kmeans_plus_weightnet_trainer(dataname, D, K, lr=0.001, max_time=60,
                                  weight=None, kmeans_clf=None, batch_size=1000,
                                  weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model
        
    log_name = 'heter/%s/kmeans_weight/k%d/lr%.3f/bs%d/wd%.5f' % (dataname, K, lr,
                                                                  batch_size,
                                                                  weight_decay)

    ### trainers
    kmeans_trainer = KmeansTrainer(K, clf=kmeans_clf)    
    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time,
                                         weight_decay=weight_decay)

    combined_trainer = CombineTrainer(log_name=log_name)
    combined_trainer.add(kmeans_trainer)
    combined_trainer.add(weightnet_trainer)
    return combined_trainer

def poskmeans_plus_weightnet_trainer(dataname, D, K, lr=0.001, max_time=60,
                                     weight=None, kmeans_clf=None, batch_size=1000,
                                     weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model
        
    log_name = 'heter/%s/posk_weight/k%d/lr%.3f/bs%d/wd%.5f' % (dataname, K, lr,
                                                              batch_size, weight_decay)

    ### trainers
    kmeans_trainer = PosKmeansTrainer(K, clf=kmeans_clf)    
    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time,
                                         weight_decay=weight_decay)

    combined_trainer = CombineTrainer(log_name=log_name)
    combined_trainer.add(kmeans_trainer)
    combined_trainer.add(weightnet_trainer)
    return combined_trainer


def weightnet_trainer(dataname, D, K=None, lr=0.001, max_time=60,
                      weight=None, batch_size=1000,
                      weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    K = D
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/%s/weight/lr%.3f/bs%d/wd%.5f' % (dataname, lr, batch_size,
                                                     weight_decay)

    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time,
                                         log_name=log_name,
                                         weight_decay=weight_decay)

    return weightnet_trainer

def mlp_trainer(dataname, D, K=None, lr=0.001, max_time=60,
                model=None, batch_size=1000, weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    if model is None:
        model = MLP(D)

    log_name = 'heter/%s/mlp/lr%.3f/bs%d/wd%.5f' % (dataname, lr, batch_size,
                                                  weight_decay)

    t = MLPTrainer(model, lr=lr, max_time=max_time,
                   log_name=log_name, weight_decay=weight_decay)

    return t

def linear_trainer(dataname, D, K=None, lr=0.001, max_time=60,
                   model=None, batch_size=1000, weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    if model is None:
        model = nn.Linear(D, 1)

    log_name = 'heter/%s/linear/lr%.3f/bs%d/wd%.5f' % (dataname, lr, batch_size,
                                                       weight_decay)

    t = MLPTrainer(model, lr=lr, max_time=max_time,
                   log_name=log_name, islinear=True,
                   weight_decay=weight_decay)

    return t

def autoencoder_plus_weightnet_trainer(dataname, D, K, lr=0.001, max_time=60,
                                       autoencoder=None, weight=None,
                                       batch_size=1000, weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    if autoencoder is None:
        autoencoder = AutoEncoder(D, K)
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/%s/auto_plus_weight/k%d/lr%.3f/bs%d/wd%.5f' % (dataname, K, lr,
                                                                     batch_size,
                                                                     weight_decay)

    ### trainers
    auto_trainer = AutoEncoderTrainer(autoencoder, lr=lr, max_time=max_time,
                                      weight_decay=weight_decay)
    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time,
                                         weight_decay=weight_decay)

    combined_trainer = CombineTrainer(log_name=log_name)
    combined_trainer.add(auto_trainer)
    combined_trainer.add(weightnet_trainer)
    return combined_trainer

def autoencoder_plus_kmeans_plus_weightnet_trainer(dataname, D, K,
                                                   lr=0.001, max_time=60,
                                                   autoencoder=None,
                                                   weight=None,
                                                   kmeans_clf=None,
                                                   batch_size=1000,
                                                   weight_decay=0):
    ''' D is data dimension, K is number of clusters '''
    if autoencoder is None:
        autoencoder = AutoEncoder(D, K)
    if weight is None:
        weight = Weight(K, D+1) # +1 for b in linear model

    log_name = 'heter/%s/auto_plus_kmeans_plus_weight/k%d/lr%.3f/bs%d/wd%.5f'\
               % (dataname,
                  K, lr,
                  batch_size, weight_decay)

    ### trainers
    kmeans_trainer = KmeansTrainer(K, clf=kmeans_clf)            
    auto_trainer = AutoEncoderTrainer(autoencoder, lr=lr, max_time=max_time,
                                      weight_decay=weight_decay)
    weightnet_trainer = WeightNetTrainer(weight, apply_linear,
                                         lr=lr, max_time=max_time,
                                         weight_decay=weight_decay)

    combined_trainer = CombineTrainer(log_name=log_name)
    combined_trainer.add(auto_trainer)
    combined_trainer.add(kmeans_trainer)    
    combined_trainer.add(weightnet_trainer)
    return combined_trainer

if __name__ == '__main__':
    parser = parse_main_args()
    args = parser.parse_args()

    # torch.cuda.set_device(args.c)
    
    dataname = args.d
    batch_size = args.s   
    train_data, val_data,\
        train_theta, val_theta,\
        ndim, n_islands = loadData(dataname, batch_size=batch_size)

    D = ndim
    K = args.k
    alpha = args.a
    max_time = args.t
    weight_decay = args.w
    
    if args.b == 0:
        t = deep_locally_linear_trainer(dataname, D, K, alpha=alpha,
                                        max_time=max_time,
                                        batch_size=batch_size,
                                        weight_decay=weight_decay)
    elif args.b == 1:
        t = kmeans_plus_weightnet_trainer(dataname, D, K,
                                          max_time=max_time,
                                          batch_size=batch_size,
                                          weight_decay=weight_decay)
    elif args.b == 2:
        t = linear_trainer(dataname, D, K, max_time=max_time,
                           batch_size=batch_size,
                           weight_decay=weight_decay)
    elif args.b == 3:
        t = mixture_of_expert_trainer(dataname, D, K, alpha=alpha,
                                      max_time=max_time,
                                      batch_size=batch_size,
                                      weight_decay=weight_decay)
    elif args.b == 4:
        t = mlp_trainer(dataname, D, K,
                        max_time=max_time,
                        batch_size=batch_size,
                        weight_decay=weight_decay)
    elif args.b == 5:
        t = poskmeans_plus_weightnet_trainer(dataname, D, K,
                                             max_time=max_time,
                                             batch_size=batch_size,
                                             weight_decay=weight_decay)
    elif args.b == 6:
        t = optimal_weight_trainer(dataname, D, K, alpha=alpha,
                                   max_time=max_time,
                                   batch_size=batch_size,
                                   switch_update_every=args.f,
                                   weight_decay=weight_decay)
    
    # elif args.b == 7:
    #     t = autoencoder_plus_kmeans_plus_weightnet_trainer(dataname, D, K,
    #                                                        max_time=max_time,
    #                                                        batch_size=batch_size,
    #                                                        weight_decay=weight_decay)
    # elif args.b == 8:
    #     t = autoencoder_plus_weightnet_trainer(dataname, D, K,
    #                                            max_time=max_time,
    #                                            batch_size=batch_size,
    #                                            weight_decay=weight_decay)
    # elif args.b == 9:
    #     t = weightnet_trainer(dataname, D, K,
    #                           max_time=max_time,
    #                           batch_size=batch_size,
    #                           weight_decay=weight_decay)

    if dataname == 'mimic2':
        use_auc = True
    else:
        use_auc = False

    t.fit(train_data, n_epochs=15000, valdata=val_data, val_theta=val_theta,
          use_auc=use_auc)


