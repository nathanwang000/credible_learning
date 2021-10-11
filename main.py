from lib.data import Mimic2
import numpy as np
from lib.model import LR, CCM_res, CBM
from lib.train import Trainer, prepareData
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.regularization import eye_loss, wridge, wlasso, lasso, \
    enet, owl, ridge, eye_loss2, eye_loss_height
from sklearn.metrics import accuracy_score
from lib.utility import get_y_yhat, model_auc, calcAP, sweepS1, sparsity, bootstrap
import torch, os
from scipy.stats import ttest_rel
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import tqdm
import joblib
import sys

def trainData(name, data, regularization=eye_loss, alpha=0.01, model=None,
              n_epochs=300, learning_rate=1e-3, batch_size=4000, r=None, test=False):
    '''
    return validation auc, average precision, score1
    if test is true, combine train and val and report on test performance
    '''
    m = data

    if test:
        name = 'test' + name
        xtrain = np.vstack([m.xtrain, m.xval])
        xval = m.xte
        ytrain = np.hstack([m.ytrain, m.yval])
        yval = m.yte
    else:
        xtrain = m.xtrain
        xval = m.xval
        ytrain = m.ytrain
        yval = m.yval

    # note: for cross validation, just split data into n fold and
    # choose appropriate train_data and valdata from those folds
    # not doing here for simplicity
    d = m.r.size(0)
    train_data = TensorDataset(*map(lambda x: x.data, prepareData(xtrain, ytrain)))
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valdata = TensorDataset(*map(lambda x: x.data, prepareData(xval, yval)))
    valdata = DataLoader(valdata, batch_size=4000, shuffle=True)

    n_output = 2 # binary classification task
    if model is None:
        model = LR(d, n_output)
    reg_parameters = model.i2o.weight

    t = Trainer(model, lr=learning_rate, risk_factors=m.r, alpha=alpha,
                regularization=regularization, reg_parameters=reg_parameters,
                name=name)
    losses, vallosses = t.fit(data, n_epochs=n_epochs, print_every=1, valdata=valdata)

    # report statistics: very messy code below
    ap = calcAP(m.r.data.numpy(), (reg_parameters[1] - reg_parameters[0]).data.numpy())
    val_auc = model_auc(model, valdata)
    t, s1 = sweepS1(model, valdata)

    if 'expert_feature_only' in name:
        joblib.dump((ap, val_auc, s1, val_auc, s1), 'models/' + name + '.pkl')
        return
        
    m.clean() # become clean data
    if test:
        xtrain = np.vstack([m.xtrain, m.xval])
        xval = m.xte
        ytrain = np.hstack([m.ytrain, m.yval])
        yval = m.yte
    else:
        xtrain = m.xtrain
        xval = m.xval
        ytrain = m.ytrain
        yval = m.yval

    # note: for cross validation, just split data into n fold and
    # choose appropriate train_data and valdata from those folds
    # not doing here for simplicity
    d = m.r.size(0)
    train_data = TensorDataset(*map(lambda x: x.data, prepareData(xtrain, ytrain)))
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valdata = TensorDataset(*map(lambda x: x.data, prepareData(xval, yval)))
    valdata = DataLoader(valdata, batch_size=4000, shuffle=True)

    clean_auc = model_auc(model, valdata)
    t, clean_s1 = sweepS1(model, valdata)
    
    joblib.dump((ap, val_auc, s1, clean_auc, clean_s1), 'models/' + name + '.pkl')    

class ParamSearch:
    def __init__(self, data, n_cpus=10):
        self.tasks = []
        self.hyperparams = []
        self.n_cpus = n_cpus
        self.data = data
        valdata = TensorDataset(*map(lambda x: x.data,
                                     prepareData(data.xval, data.yval)))
        self.valdata = DataLoader(valdata, batch_size=4000, shuffle=True)    
        
    def add_param(self, name, reg, alpha, *args):
        if not os.path.exists('models/' + name + '.pkl'):        
            self.tasks.append((name, self.data, reg, alpha, *args))
        self.hyperparams.append((name, reg, alpha))

    def select_on_auc_sp(self, n_bootstrap=100):
        '''
        return the index in self.hyperparams chosen

        hyper parameter selection based on auc and sp
        choose the hyper parameter that has no auc difference with top 
        but is the sparsest model

        This is the criteria used in the learning credible models paper
        '''
        print('hyperparam select using auc and sparsity')        
        aucs = []
        models = []
        sparsities = []
        for name, reg, alpha in self.hyperparams:
            # load the model        
            model = torch.load('models/' + name + '.pt')
            reg_parameters = model.i2o.weight
            sp = sparsity((reg_parameters[1]-reg_parameters[0]).data.numpy())
            models.append(model)
            sparsities.append(sp)

        for _ in range(n_bootstrap):
            test = bootstrap(self.valdata)
            local_aucs = []
            for model in models:
                # bootstrap for CI on auc
                local_aucs.append(model_auc(model, test))
            aucs.append(local_aucs)
        aucs = np.array(aucs)

        # only keep those with high auc
        b = np.argmax(aucs.mean(0))
        discardset = set([])
        for a in range(len(models)):
            diffs = ((aucs[:,a] - aucs[:,b]) >= 0).astype(int)
            if diffs.sum() / diffs.shape[0] <= 0.05:
                discardset.add(a)

        # choose the one with largest sparsity
        chosen, sp = max(filter(lambda x: x[0] not in discardset,
                                enumerate(sparsities)),
                         key=lambda x: x[1])
        return chosen
    
    def select_on_auc(self, *args, **kwargs):
        '''hyperparameter selection based on auc alone
        return index within self.hyperparams that need to retrain
        '''
        print('hyperparam select using auc')
        aucs = []
        sparsities = []
        for name, reg, alpha in self.hyperparams:
            # load the model        
            model = torch.load('models/' + name + '.pt')
            aucs.append(model_auc(model, self.valdata))

        # choose the one with largest auc
        chosen, auc = max(enumerate(aucs),
                          key=lambda x: x[1])
        return chosen

    def select_on_auc_ap(self, n_bootstrap=100):
        '''
        return the index in self.hyperparams chosen

        hyper parameter selection based on auc and ap
        choose the hyper parameter that has no auc difference with top 
        but is the model with highest average precision (align with expert)
        '''
        print('hyperparam select using auc and ap')
        aucs = []
        models = []
        aps = []
        for name, reg, alpha in self.hyperparams:
            # load the model        
            model = torch.load('models/' + name + '.pt')
            reg_parameters = model.i2o.weight
            ap = calcAP(self.data.r.data.numpy(),
                        (reg_parameters[1] - reg_parameters[0]).data.numpy())
            models.append(model)
            aps.append(ap)

        for _ in range(n_bootstrap):
            test = bootstrap(self.valdata)
            local_aucs = []
            for model in models:
                # bootstrap for CI on auc
                local_aucs.append(model_auc(model, test))
            aucs.append(local_aucs)
        aucs = np.array(aucs)

        # only keep those with high auc
        b = np.argmax(aucs.mean(0))
        discardset = set([])
        for a in range(len(models)):
            diffs = ((aucs[:,a] - aucs[:,b]) >= 0).astype(int)
            if diffs.sum() / diffs.shape[0] <= 0.05:
                discardset.add(a)

        # choose the one with largest average precision
        chosen, ap = max(filter(lambda x: x[0] not in discardset,
                                enumerate(aps)),
                         key=lambda x: x[1])
        return chosen
    

    def select_on_auc_alpha(self, n_bootstrap=100):
        '''
        return the index in self.hyperparams chosen

        hyper parameter selection based on auc and alpha
        choose the hyper parameter that has no auc difference with top 
        but is the model with highest average precision (align with expert)
        '''
        print('hyperparam select using auc and alpha')
        aucs = []
        models = []
        alphas = []
        for name, reg, alpha in self.hyperparams:
            # load the model        
            model = torch.load('models/' + name + '.pt')
            reg_parameters = model.i2o.weight
            ap = calcAP(self.data.r.data.numpy(),
                        (reg_parameters[1] - reg_parameters[0]).data.numpy())
            models.append(model)
            alphas.append(alpha)

        for _ in range(n_bootstrap):
            test = bootstrap(self.valdata)
            local_aucs = []
            for model in models:
                # bootstrap for CI on auc
                local_aucs.append(model_auc(model, test))
            aucs.append(local_aucs)
        aucs = np.array(aucs)

        # only keep those with high auc
        b = np.argmax(aucs.mean(0))
        discardset = set([])
        for a in range(len(models)):
            diffs = ((aucs[:,a] - aucs[:,b]) >= 0).astype(int)
            if diffs.sum() / diffs.shape[0] <= 0.05:
                discardset.add(a)

        # choose the one with largest alpha
        chosen, alpha = max(filter(lambda x: x[0] not in discardset,
                                enumerate(alphas)),
                         key=lambda x: x[1])
        return chosen
    
    def run(self, n_bootstrap=100):

        if self.n_cpus is None: n_jobs = 10
        else: n_jobs = self.n_cpus
        Parallel(n_jobs=n_jobs)(delayed(trainData)(*task) for task in self.tasks)
        # for task in self.tasks:
        #     trainData(*task)

        # select a model to run
        # chosen_idx = self.select_on_auc_sp(n_bootstrap=n_bootstrap)
        # chosen_idx = self.select_on_auc() # don't need bootstrap
        # chosen_idx = self.select_on_auc_ap(n_bootstrap=n_bootstrap)
        chosen_idx = self.select_on_auc_alpha(n_bootstrap=n_bootstrap)
        
        # retrian the chosen model
        name, reg, alpha = self.hyperparams[chosen_idx]
        print('name', name)        
        trainData(name, self.data, reg, alpha, test=True)

def two_stage_exp(threshold=0.90, n_cpus=None, n_bootstrap=30):
    '''
    remove features by setting a threshold on correlation, 
    then apply l2 regularization on the remaining features
    '''
    m = Mimic2(mode='total', two_stage=True, threshold=float(threshold))
    ps = ParamSearch(m, n_cpus)

    reg = ridge
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        name = 'two_stage_ridge_' + str(threshold) + '^' + str(alpha)
        ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)
    
def random_risk_exp(regs, n_cpus=None, n_bootstrap=30):
    m = Mimic2(mode='total', random_risk=True)
    ps = ParamSearch(m, n_cpus)

    reg = eye_loss    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for reg in regs:
        for alpha in alphas:
            name = 'random_risk_' + reg.__name__ + '^' + str(alpha)
            ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)

def reg_exp(regs, n_cpus=None, n_bootstrap=30):
    m = Mimic2(mode='total')
    ps = ParamSearch(m, n_cpus)
    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for reg in regs:
        for alpha in alphas:
            name = reg.__name__ + '^' + str(alpha)
            ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)

def eye_height_exp(regs, n_cpus=None, n_bootstrap=30):
    m = Mimic2(mode='total')
    ps = ParamSearch(m, n_cpus)
    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for reg in regs:
        for alpha in alphas:
            name = reg.__name__ + '^' + str(alpha)
            ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)

def expert_feature_only_exp(n_cpus=None, n_bootstrap=30):
    m = Mimic2(mode='total', expert_feature_only=True)
    ps = ParamSearch(m, n_cpus)

    reg = ridge
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for alpha in alphas:
        name = 'expert_only_ridge' + '^' + str(alpha)
        ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)

def duplicate_exp(regs, n_cpus=None, n_bootstrap=30):
    m = Mimic2(mode='total', duplicate=1)
    ps = ParamSearch(m, n_cpus)
    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for reg in regs:
        for alpha in alphas:
            name = reg.__name__ + '_dup_' + '^' + str(alpha)
            ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)
        
def stdcx_shortcut_exp(n_cpus=None, n_bootstrap=30):
    # shortcut experiments for stdcx
    m = Mimic2(mode='total', dupr=True)
    m.bias() # setup the shortcut
    ps = ParamSearch(m, n_cpus)
    
    reg = ridge
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for alpha in alphas:
        name = 'stdcx_shortcut_' + '^' + str(alpha)
        ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)
    
def res_shortcut_exp(expert_only_paths, n_cpus=None, n_bootstrap=30):
    # shortcut experiments for ccm res
    m = Mimic2(mode='total')
    m.bias() # setup the shortcut
    ps = ParamSearch(m, n_cpus)

    cbm = CBM(m.r, torch.load(expert_only_paths[0]).i2o) # the path is to conform my api
    reg = ridge
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for alpha in alphas:
        name = 'res_shortcut_' + '^' + str(alpha)
        ps.add_param(name, reg, alpha, CCM_res(cbm, LR(m.r.size(0), 2).i2o))

    ps.run(n_bootstrap)
    
def shortcut_exp(regs, n_cpus=None, n_bootstrap=30):
    # shortcut experiments for concept credible model
    m = Mimic2(mode='total')
    m.bias() # setup the shortcut
    ps = ParamSearch(m, n_cpus)
    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for reg in regs:
        for alpha in alphas:
            name = reg.__name__ + '_shortcut_' + '^' + str(alpha)
            ps.add_param(name, reg, alpha)

    ps.run(n_bootstrap)
    
#####################################################
def wridge1_5(*args, **kwargs):
    return wridge(*args, **kwargs, w=1.5)

def wridge3(*args, **kwargs):
    return wridge(*args, **kwargs, w=3)

def wlasso1_5(*args, **kwargs):
    return wlasso(*args, **kwargs, w=1.5)

def wlasso3(*args, **kwargs):
    return wlasso(*args, **kwargs, w=3)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please specify your function and argument to run')
    else:
        print(sys.argv[1:])
        f = eval(sys.argv[1])
        if len(sys.argv) >= 3:
            args = eval(sys.argv[2])
            f(args)
        else:
            f()
