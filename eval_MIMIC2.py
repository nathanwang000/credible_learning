from lib.data import Mimic2
import numpy as np
from lib.model import LR
from lib.train import Trainer, prepareData
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.regularization import eye_loss, wridge, wlasso, lasso, ridge, owl
from sklearn.metrics import accuracy_score
from lib.utility import get_y_yhat, model_auc
import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from main import random_risk_exp, expert_feature_only_exp, reg_exp
import glob
# from sklearn.externals import joblib
import joblib
from pandas import DataFrame
import sys

if len(sys.argv) == 2:
    model_dir = sys.argv[1]
else:
    model_dir = 'models'
print('looking at {}/'.format(model_dir))

raw_data = []
for fn in glob.glob('{}/test*.pkl'.format(model_dir)):
    name = fn.split('/')[-1].split('.pkl')[0]
    name, alpha = name.split('^')
    if 'test' in name: name = name[4:]+'*'
    ap, auc, s1, clean_auc, clean_s1 = joblib.load(fn)
    raw_data.append([name, alpha, ap, auc, s1, clean_auc, clean_s1])
df = DataFrame(data=raw_data, columns=['method name', 'alpha', 'AP', 'auc (biased)', 'min(recall, precision) (biased)', 'auc (clean)', 'min(recall, precision) (clean)'])
print(df.sort_values(['auc (clean)'], ascending=False))
