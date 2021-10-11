from pandas import DataFrame
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
import glob

def evaluate(model, data, batch_size=4000):
    m = data
    xtrain = np.vstack([m.xtrain, m.xval])
    xval = m.xte
    ytrain = np.hstack([m.ytrain, m.yval])
    yval = m.yte

    # note: for cross validation, just split data into n fold and
    # choose appropriate train_data and valdata from those folds
    # not doing here for simplicity
    d = m.r.size(0)
    train_data = TensorDataset(*map(lambda x: x.data, prepareData(xtrain, ytrain)))
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valdata = TensorDataset(*map(lambda x: x.data, prepareData(xval, yval)))
    valdata = DataLoader(valdata, batch_size=4000, shuffle=True)

    auc = model_auc(model, valdata)
    t, s1 = sweepS1(model, valdata)
    return auc, s1
    
if __name__ == '__main__':

    if len(sys.argv) == 2:
        model_dir = sys.argv[1]
    else:
        model_dir = 'models'
    print('looking at {}/'.format(model_dir))

    raw_data = []
    for fn in glob.glob('{}/test*.pt'.format(model_dir)):
        name = fn.split('/')[-1].split('.pt')[0]
        name, alpha = name.split('^')
        if 'test' in name: name = name[4:]+'*'

        print(name)
        if 'expert_only' in name: continue
        if 'stdcx' in name:
            data = Mimic2(mode='total', dupr=True)
        else:
            data = Mimic2(mode='total')            

        model = torch.load(fn)        
        data.bias()
        auc_b, s1_b = evaluate(model, data)
        data.clean()
        auc_c, s1_c = evaluate(model, data)        
        raw_data.append([name, alpha, auc_b, s1_b, auc_c, s1_c])
        
    df = DataFrame(data=raw_data, columns=['method name', 'alpha', 'auc (biased)', 'min(recall, precision) (biased)', 'auc (clean)', 'min(recall, precision) (clean)'])
    print(df.sort_values(['auc (clean)'], ascending=False))

        
    
