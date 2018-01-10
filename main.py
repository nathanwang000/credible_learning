from lib.data import Mimic2
from lib.parallel_run import map_parallel
import numpy as np
from lib.model import LR
from lib.train import Trainer, prepareData
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.regularization import eye_loss, wridge, wlasso, lasso, enet, owl, ridge
from sklearn.metrics import accuracy_score
from lib.utility import get_y_yhat, model_auc, calcAP, sweepS1, sparsity, bootstrap
import torch, os
from scipy.stats import ttest_rel
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def trainData(name, data, regularization=eye_loss, alpha=0.01, n_epochs=300, 
              learning_rate=1e-3, batch_size=4000, r=None, test=False):
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
    model = LR(d, n_output)
    reg_parameters = model.i2o.weight
    t = Trainer(model, lr=learning_rate, risk_factors=m.r, alpha=alpha,
                regularization=regularization, reg_parameters=reg_parameters,
                name=name)
    losses, vallosses = t.fit(data, n_epochs=n_epochs, print_every=1, valdata=valdata)

    # report statistics
    val_auc = model_auc(model, valdata)
    ap = calcAP(m.r.data.numpy(), (reg_parameters[1] - reg_parameters[0]).data.numpy())
    t, s1 = sweepS1(model, valdata)
    sp = sparsity((reg_parameters[1]-reg_parameters[0]).data.numpy())
    joblib.dump((val_auc, ap, s1, sp), 'models/' + name + '.pkl')    
    return val_auc, ap, s1, sp

def random_risk_exp(n_cpus=None, n_bootstrap=30):
    # parameter search
    m = Mimic2(mode='total', random_risk=True)
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    n_epochs = 300
    reg = eye_loss
    tasks = []
    hyperparams = []
    for alpha in alphas:
        name = 'random_risk_eye' + '^' + str(alpha)
        if not os.path.exists('models/' + name + '.pkl'):
            tasks.append((name, m, reg, alpha))
        hyperparams.append((name, m, reg, alpha))
    map_parallel(trainData, tasks, n_cpus)

    # select a model to run: split on auc and sparsity
    valdata = TensorDataset(*map(lambda x: x.data, prepareData(m.xval, m.yval)))
    valdata = DataLoader(valdata, batch_size=4000, shuffle=True)    
    aucs = []
    models = []
    sparsities = []
    for alpha in alphas:
        # load the model        
        name = 'random_risk_eye' + '^' + str(alpha)
        model = torch.load('models/' + name + '.pt')
        reg_parameters = model.i2o.weight
        sp = sparsity((reg_parameters[1]-reg_parameters[0]).data.numpy())
        models.append(model)
        sparsities.append(sp)
        
    for _ in range(n_bootstrap):
        test = bootstrap(valdata)
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
        diffs = ((aucs[:,a] - aucs[:,b]) >= 0).astype(np.int)
        if diffs.sum() / diffs.shape[0] <= 0.05:
            discardset.add(a)
        # t, p = ttest_rel(aucs[:,a], aucs[:,b])
        # if t < 0 and p < 0.05:
        #     discardset.add(a)
    #print(discardset)
    # choose the one with largest sparsity
    chosen, sp = max(filter(lambda x: x[0] not in discardset,
                            enumerate(sparsities)),
                     key=lambda x: x[1])

    # retrian the chosen model
    name, m, reg, alpha = hyperparams[chosen]
    trainData(name, m, reg, alpha, test=True)
    
def diff_regs_exp(n_cpus=None, n_bootstrap=30):
    m = Mimic2(mode='total')    
    regs = [eye_loss, wridge, wlasso, lasso, enet, owl]    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    tasks = []
    for reg in regs:
        for alpha in alphas:
            name = reg.__name__ + '^' + str(alpha)
            if not os.path.exists('models/' + name + '.pkl'):
                tasks.append((name, m, reg, alpha))
    map_parallel(trainData, tasks, n_cpus)

    # select a model to run: split on auc and sparsity
    valdata = TensorDataset(*map(lambda x: x.data, prepareData(m.xval, m.yval)))
    valdata = DataLoader(valdata, batch_size=4000, shuffle=True)    

    for reg in regs:
        # select for each regularization used
        aucs = []
        models = []
        hyperparams = []
        sparsities = []
        for alpha in alphas:
            # load the model
            name = reg.__name__ + '^' + str(alpha)
            hyperparams.append((name, m, reg, alpha))            
            model = torch.load('models/' + name + '.pt')
            reg_parameters = model.i2o.weight
            sp = sparsity((reg_parameters[1]-reg_parameters[0]).data.numpy())
            models.append(model)
            sparsities.append(sp)
        
        for _ in range(n_bootstrap):
            test = bootstrap(valdata)
            local_aucs = []
            for model in models:
                # bootstrap for CI on auc
                local_aucs.append(model_auc(model, test))
            aucs.append(local_aucs)
        aucs = np.array(aucs)

        # only keep those with high auc
        argsorted_means = np.argsort(aucs.mean(0))
        index = -1
        b = argsorted_means[index]
        discardset = set([])
        for a in range(len(models)):
            diffs = ((aucs[:,a] - aucs[:,b]) >= 0).astype(np.int)
            if diffs.sum() / diffs.shape[0] <= 0.05:
                discardset.add(a)
            # paired t test too strict
            # t, p = ttest_rel(aucs[:,a], aucs[:,b])
            # if t < 0 and p < 0.05:
            #     discardset.add(a)

        # choose the one with largest sparsity
        chosen, sp = max(filter(lambda x: x[0] not in discardset,
                                enumerate(sparsities)),
                         key=lambda x: x[1])

        # retrian the chosen model
        name, m, reg, alpha = hyperparams[chosen]
        print('alpha chosen', alpha)
        trainData(name, m, reg, alpha, test=True)
    

def expert_feature_only_exp(n_cpus=None, n_bootstrap=30):
    m = Mimic2(mode='total', expert_feature_only=True)
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    n_epochs = 300
    reg = ridge
    tasks = []
    hyperparams = []
    for alpha in alphas:
        name = 'expert_only_ridge' + '^' + str(alpha)
        if not os.path.exists('models/' + name + '.pkl'):        
            tasks.append((name, m, reg, alpha))
        hyperparams.append((name, m, reg, alpha))
    map_parallel(trainData, tasks, n_cpus)

    # select a model to run: split on auc and sparsity
    valdata = TensorDataset(*map(lambda x: x.data, prepareData(m.xval, m.yval)))
    valdata = DataLoader(valdata, batch_size=4000, shuffle=True)    
    aucs = []
    models = []
    sparsities = []
    for alpha in alphas:
        # load the model        
        name = 'expert_only_ridge' + '^' + str(alpha)
        model = torch.load('models/' + name + '.pt')
        reg_parameters = model.i2o.weight
        sp = sparsity((reg_parameters[1]-reg_parameters[0]).data.numpy())
        models.append(model)
        sparsities.append(sp)
        
    for _ in range(n_bootstrap):
        test = bootstrap(valdata)
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
        diffs = ((aucs[:,a] - aucs[:,b]) >= 0).astype(np.int)
        if diffs.sum() / diffs.shape[0] <= 0.05:
            discardset.add(a)
        # t, p = ttest_rel(aucs[:,a], aucs[:,b])
        # if t < 0 and p < 0.05:
        #     discardset.add(a)
    #print(discardset)
    # choose the one with largest sparsity
    chosen, sp = max(filter(lambda x: x[0] not in discardset,
                            enumerate(sparsities)),
                     key=lambda x: x[1])

    # retrian the chosen model
    name, m, reg, alpha = hyperparams[chosen]
    trainData(name, m, reg, alpha, test=True)
    

#####################################################
def run_exp(n_cpus=30, n_bootstrap=100):
    random_risk_exp(n_cpus, n_bootstrap)
    diff_regs_exp(n_cpus, n_bootstrap)    
    expert_feature_only_exp(n_cpus, n_bootstrap)

def main():
    run_exp()
    
if __name__ == '__main__':
    main()
