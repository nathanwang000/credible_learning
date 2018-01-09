import time
import numpy as np
import math
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import average_precision_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def data_shuffle(x, y):
    p = np.random.permutation(y.shape[0])
    return x[p], y[p]

def get_y_yhat(model, data):
    yhat = []
    ys = []
    for x, y in data:
        ys.append(y.numpy())
        x, y = Variable(x), Variable(y)
        yhat.append(model(x).data.numpy())

    return np.hstack(ys), np.vstack(yhat)

def model_auc(model, data):
    yhat = []
    ys = []
    for x, y in data:
        ys.append(y.numpy())
        x, y = Variable(x), Variable(y)
        yhat.append(model(x).data.numpy())
    y, yhat = np.hstack(ys), np.vstack(yhat)[:,1]
    return roc_auc_score(y, yhat)    

def model_acc(model, data):
    yhat = []
    ys = []
    for x, y in data:
        ys.append(y.numpy())
        x, y = Variable(x), Variable(y)
        yhat.append(np.argmax(model(x).data.numpy(), 1))
    y, yhat = np.hstack(ys), np.vstack(yhat)

    return accuracy_score(y, yhat)

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

def calcAP(ytrue, ypred):
    return average_precision_score(ytrue, np.abs(ypred))
    
def plotAUC(m, model, valdata):
    # m is a dataset in data.py
    X_test = m.xval
    y_test = m.yval
    X_train = m.xtrain
    y_train = m.ytrain
    y_test, y_score = get_y_yhat(model, valdata)
    y_score = y_score[:, 1]
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

# sweep for accuracy
def score1(cm):
    tn, fp, fn, tp = cm.ravel()
    se = tp / (tp + fn)
    precision = tp / (tp + fp) 
    return min(se, precision)

def acc(cm):
    tn, fp, fn, tp = cm.ravel()
    return (tp + tn) / cm.sum()

def sweepS1(model, valdata, plot=False, mode='s1'):
    # mode are s1: score1, acc: accuracy
    y_test, y_score = get_y_yhat(model, valdata)
    y_score = np.exp(y_score[:,1])
    thresholds = np.unique(sorted(y_score))
    
    def getS1(t):
        yhat = (y_score >= t).astype(np.int)
        cm = confusion_matrix(y_test, yhat)
        return {'s1': score1, 'acc': acc}[mode](cm)

    scores = [getS1(t) for t in thresholds]
    max_ind = np.argmax(scores)
    t = thresholds[max_ind]

    if plot:
        plt.plot(thresholds, scores, color='green', lw=2)
        plt.xlabel('threshold')
        plt.ylabel('score1 (min(sensitivity, precision))')

        plt.show()
        
    return t, getS1(t)
