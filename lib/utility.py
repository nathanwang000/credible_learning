import time
import numpy as np
import math
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, roc_auc_score

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def data_shuffle(x, y):
    p = np.random.permutation(y.shape[0])
    return x[p], y[p]

def model_acc(model, data, mode='auc'):
    yhat = []
    ys = []
    for x, y in data:
        ys.append(y.numpy())
        x, y = Variable(x), Variable(y)
        yhat.append(np.argmax(model(x).data.numpy(), 1))

    return {'auc': roc_auc_score, 'acc': accuracy_score
    }[mode](np.hstack(ys), np.hstack(yhat))

def calc_loss(model, data, loss):
    cost = 0
    denom = 0
    for x, y in data:
        x, y = Variable(x), Variable(y)
        regret = loss(model(x), y).data[0]
        m = x.size(0)
        cost += regret * m
        denom += m
    return cost / denom

        
