from lib.data import Mimic2
from lib.parallel_run import map_parallel
import numpy as np
from lib.model import LR
from lib.train import Trainer, prepareData
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.regularization import eye_loss, wridge, wlasso, lasso, enet, owl
from sklearn.metrics import accuracy_score
from lib.utility import get_y_yhat, model_auc, calcAP, sweepS1
import torch, os
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def trainData(name, data, regularization=eye_loss, alpha=0.01, n_epochs=300, 
              learning_rate=1e-3, batch_size=4000, r=None):
    '''
    return validation auc, average precision, score1
    '''
    m = data

    d = m.r.size(0)
    train_data = TensorDataset(*map(lambda x: x.data, prepareData(m.xtrain, m.ytrain)))
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valdata = TensorDataset(*map(lambda x: x.data, prepareData(m.xval, m.yval)))
    valdata = DataLoader(valdata, batch_size=4000, shuffle=True)

    n_output = 2 # binary classification task 
    model = LR(d, n_output)
    reg_parameters = model.i2o.weight
    t = Trainer(model, lr=learning_rate, risk_factors=m.r, alpha=alpha,
                regularization=regularization, reg_parameters=reg_parameters,
                name=name)
    losses, vallosses = t.fit(data, n_epochs=n_epochs, print_every=1, valdata=valdata)

    # report statistics
    # train_auc = model_auc(model, data)
    val_auc = model_auc(model, valdata)
    ap = calcAP(m.r.data.numpy(), (reg_parameters[1] - reg_parameters[0]).data.numpy())
    t, s1 = sweepS1(model, valdata)
    joblib.dump((val_auc, ap, s1), 'models/' + name + '.pkl')    
    return val_auc, ap, s1

def main():

    # random_risk = True
    # m = Mimic2(mode='total', random_risk=random_risk)
    # n_epochs = 300
    # regularization = eye_loss
    # name = 'random_risk_eye'
    # auc, ap, s1 = trainData(name, m, n_epochs=n_epochs,
    #                         regularization=regularization)

    m = Mimic2(mode='total', random_risk=False)    
    regs = [eye_loss, wridge, wlasso, lasso, enet, owl]    
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    tasks = []
    for reg in regs:
        for alpha in alphas:
            name = reg.__name__ + str(alpha)
            tasks.append((name, m, reg, alpha))
    map_parallel(trainData, tasks, n_cpus=30)
    
if __name__ == '__main__':
    main()
