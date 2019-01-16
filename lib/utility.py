import time, random, string
import numpy as np
import math
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import average_precision_score, confusion_matrix
from lib.data import Mimic2
import numpy as np
import time
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import torch
from torch import nn
from PIL import Image
from torch.autograd import Function
from scipy.linalg import block_diag
import os
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lib.settings import USE_GPU

def to_cuda(x):
    if USE_GPU and torch.cuda.is_available():
        x = x.cuda()
    return x

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x, *args, **kwargs):
    x = to_cuda(x)
    return Variable(x, *args, **kwargs)

def np2tensor(x, y):
    return torch.from_numpy(x).float(), torch.from_numpy(y).long()

def check_nan(v):
    return np.isnan(to_np(v)).sum() > 0

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
        x, y = to_var(x), to_var(y)
        yhat.append(model(x).data.numpy())

    return np.hstack(ys), np.vstack(yhat)

def model_auc(model, data):
    # note that this is for 2 classes, for 1 output, use report auc instead
    yhat = []
    ys = []
    for x, y in data:
        ys.append(y.numpy())
        x, y = to_var(x), to_var(y)
        yhat.append(to_np(model.forward(x)))

    y, yhat = np.hstack(ys), np.vstack(yhat)[:,1]
    return roc_auc_score(y, yhat)    

def model_acc(model, data):
    yhat = []
    ys = []
    for x, y in data:
        ys.append(y.numpy())
        x, y = to_var(x), to_var(y)
        yhat.append(np.argmax(to_np(model(x)), 1))
    y, yhat = np.hstack(ys), np.vstack(yhat)

    return accuracy_score(y, yhat)

def calc_loss(model, data, loss):
    cost = 0
    denom = 0
    for x, y in data:
        x, y = to_var(x), to_var(y)
        regret = loss(model(x), y).item()
        m = x.size(0)
        cost += regret * m
        denom += m
    return cost / denom

def calcAP(ytrue, ypred):
    return average_precision_score(ytrue, np.abs(ypred))
    
def plotAUC(m, model, valdata):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # m is a dataset in data.py
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
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.plot(thresholds, scores, color='green', lw=2)
        plt.xlabel('threshold')
        plt.ylabel('score1 (min(sensitivity, precision))')

        plt.show()
        
    return t, getS1(t)

def sparsity(param, threshold=0.01):
    # percentagge of near-zero features <= 0.01 of the largest absolute feature weight
    # param is a numpy array
    absparam = np.abs(param)
    return (absparam <= threshold * absparam.max()).sum() / np.ones_like(absparam).sum()

def bootstrap(valdata):
    num_samples = len(valdata.dataset)
    sampler = WeightedRandomSampler(np.ones(num_samples), num_samples)
    return DataLoader(valdata.dataset, batch_size=valdata.batch_size,
                      sampler=sampler)

def var2constvar(v):
    return to_var(v.data)

class ElementaryLogitLoss(Function):

    @staticmethod
    def forward(ctx, o, y):
        ctx.save_for_backward(o, y)
        a = -y*o
        res = torch.log(1+torch.exp(-y*o))
        ind = a > 50
        res[ind] = a[ind]
        return res
    
    @staticmethod
    def backward(ctx, grad_output):
        o, y = ctx.saved_variables
        grad_o = -y / (1 + torch.exp(y*o))
        grad_y = -o / (1 + torch.exp(y*o))
        return grad_o * grad_output, grad_y * grad_output

logit_elementwise_loss = ElementaryLogitLoss().apply

def prepareX(x):
    '''
    convert x from numpy to tensor
    '''
    return to_var(torch.from_numpy(x).float(), requires_grad=True)

def plotDecisionSurface(model, xmin, xmax, ymin, ymax, nsteps=30,
                        multioutput=True, colors=None):
    '''
    plot decision surface of a pytorch model
    assumes model output likelihood
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, nsteps),
                         np.linspace(ymin, ymax, nsteps))
    # note here assumes model gives log likelihood
    model_input = prepareX(np.c_[xx.ravel(), yy.ravel()])

    Z = model(model_input)
    # assert not check_nan(Z)    
    Z = to_np(Z)
    if multioutput:
        Z = np.argmax(Z, axis=1)
        # Z = Z.reshape(xx.shape)        
        # plt.contourf(xx, yy, Z)
    else:
        Z = (Z > 0).astype(np.int)

    if colors:
        colors = list(map(lambda i: colors[int(i) % len(colors)], Z))
    else:
        colors = Z
    
    plt.scatter(xx.ravel(), yy.ravel(), c=colors, s=50)        
    return model_input

def gradNorm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        total_norm += p.grad.data.norm(norm_type)**norm_type
    return total_norm**(1/norm_type)

def valueNorm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        total_norm += p.data.norm(norm_type)**norm_type
    return total_norm**(1/norm_type)
    
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
    
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h, 4 )
    
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def reportAuc(model, data):
    # note that this is for 1 classes, for 2 output, use modelAuc instead
    yhat = []
    ys = []
    for x, y in data:
        localy = y.numpy()
        localy[np.nonzero(localy == -1)] = 0 # binarize
        ys.append(localy)
        x, y = to_var(x), to_var(y)
        yhat.append(to_np(model.forward(x)))

    y, yhat = np.hstack(ys), np.vstack(yhat).ravel()
    return roc_auc_score(y, yhat)    

def reportAcc(model, test_data):
    accuracy = 0
    for k, (x, y) in enumerate(test_data):
        x, y = to_var(x).float(), to_var(y).float()
        yhat = to_np(model.forward(x) >= 0).ravel()
        yhat = yhat.astype(int)*2-1
        y = to_np(y)
        acc = (yhat==y).sum() / y.shape[0]
        accuracy += 1 / (k+1) * (acc - accuracy)
    return accuracy

def reportMSE(model, test_data, is_autoencoder=False):
    f = nn.MSELoss()
    mse_cum = 0
    for k, (x, y) in enumerate(test_data):
        x, y = to_var(x).float(), to_var(y).float()
        yhat = model.forward(x)
        if is_autoencoder:
            mse = f(yhat, x)
        else:
            mse = f(yhat, y)
        mse_cum += 1 / (k+1) * (mse - mse_cum)
    return float(to_np(mse_cum)[0])


def onehotize(z, ntasks):
    '''
    z is n x 1 long tensor variable
    '''
    m, d = z.size()
    x_onehot = to_cuda(torch.FloatTensor(m, int(ntasks)))
    x_onehot.zero_()
    x_onehot.scatter_(1, z.data.long(), 1)
    return to_var(x_onehot)

def chain_functions(funcs):
    # chain functions together
    def ret(x):
        for f in funcs:
            x = f(x)
        return x
    return ret

def gen_random_string(n=5):
    # first character has to be digit, it is n+1 characters
    d = random.choice(string.digits)
    return d + ''.join(random.choice(string.ascii_uppercase + \
                                     string.ascii_lowercase +\
                                     string.digits) for _ in range(n))
    
### data generation helpers ###
### from my code on https://gist.github.com/anonymous/1ba9a828e814bfea6c5df4d97b443ade
def genCovX(C, n): # helper function to create N(0, C)
    ''' C is the covariance matrice (assume to be psd)
    n is number of examples'''
    A = np.linalg.cholesky(C)
    d, _ = C.shape
    Z = np.random.randn(n, d)
    X = Z.dot(A.T)
    return X.astype(np.float32)

def loadData(dataname, get_test=False,
             pin_memory=True, batch_size=1000,
             num_workers=0):

    if dataname == 'mimic2':
        m = Mimic2(mode='total')
        ndim = m.xtrain.shape[1]
        if get_test:
            xtrain = np.vstack([m.xtrain, m.xval])
            xval = m.xte
            ytrain = np.hstack([m.ytrain, m.yval])
            yval = m.yte
        else:
            xtrain = m.xtrain
            xval = m.xval
            ytrain = m.ytrain
            yval = m.yval

        # make y in {-1, 1}
        ytrain = ytrain * 2 - 1
        yval = yval * 2 - 1
        
        d = m.r.size(0)
        train_data = TensorDataset(*np2tensor(xtrain, ytrain))
        data = DataLoader(train_data, shuffle=True,
                          batch_size=batch_size,
                          num_workers=num_workers, pin_memory=pin_memory)

        valdata = TensorDataset(*np2tensor(xval, yval))
        valdata = DataLoader(valdata, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory)

        if get_test:
            return valdata, None, ndim, None
        else:
            print('train shape:', xtrain.shape, 'n_islands:', None, 'ndim:', ndim)
            print('done loading data')
            return data, valdata, None, None, ndim, None

    # note: data are generated in heterogeneous groups ipython notebook
    X, Y, Theta, \
    Xval, Yval, val_theta,\
    Xtest, Ytest, test_theta,\
    ndim, n_islands = joblib.load('data/%s.pkl' % dataname)
    
    if get_test:
        xtest = torch.from_numpy(Xtest).float()
        ytest = torch.from_numpy(Ytest).float()

        test_data = TensorDataset(xtest, ytest)
        test_data = DataLoader(test_data, batch_size=batch_size,
                               num_workers=num_workers, pin_memory=pin_memory)
        return test_data, test_theta, ndim, n_islands

    x = torch.from_numpy(X).float()
    y = torch.from_numpy(Y).float()

    train_data = TensorDataset(x, y)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory)

    xval = torch.from_numpy(Xval).float()
    yval = torch.from_numpy(Yval).float()

    val_data = TensorDataset(xval, yval)
    val_data = DataLoader(val_data, batch_size=batch_size,
                           num_workers=num_workers, pin_memory=pin_memory)

    print('train shape:', X.shape, 'n_islands:', n_islands, 'ndim:', ndim)
    print('done loading data')
    return train_data, val_data,\
        Theta, val_theta,\
        ndim, n_islands
