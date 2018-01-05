import numpy as np
import sys
import torch
from torch import nn
from torch.autograd import Variable
from model import LR
import time
import math
from regularization import eye_loss, wridge, wlasso, lasso, ridge, owl, enet

def prepareTrainingInput(X, y):
    # X is an epsiode, y is the ytrue where the last label should be trusted
    # return variablized version
    return Variable(episode2tensor(X)), Variable(torch.from_numpy(y).long())

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def unison_shuffled_copies(X, y):
    assert X.shape[0] == y.shape[0]
    p = np.random.permutation(y.shape[0])
    return X[p], y[p]

class Trainer(object):
    def __init__(self, model, name, n_epochs=10, print_every=3, eval_every=100, 
                 learning_rate=0.005, criterion=nn.NLLLoss(), weight_decay=0,
                 # eye specific parameters
                 alpha=None, risk_factors=None, regularization=None,
                 parameters=None):
        # actual training
        self.name = name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.eval_every = eval_every
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
        self.alpha = alpha
        self.criterion = criterion
        if regularization and alpha: # eg. eye_loss
            self.criterion = regularization(criterion, alpha=alpha,
                                            theta=parameters, r=risk_factors)

    # train for one step
    def step(self, ytrue, episode):
        # episode tensorfied episode # for MLP episode is the minibatch
        self.optimizer.zero_grad()

        output = self.model(episode)

        loss = self.criterion(output, ytrue)
        loss.backward()

        self.optimizer.step()
        return output, loss.data[0] # 0 because only 1 loss
        
    def fit(self, X, y):
        # X are 3 dimensional array (list of episodes), y is 2d array
        # also no batchsize because just 1 at a time
        # Keep track of losses for plotting
        n_examples = X.shape[0]
        all_losses = []
        snapshots = [] # tr, tval, te snapshots
            
        time_start = time.time()
            
        cost = 0
        batch_size = 1000
        num_prints = 0
        for epoch in range(self.n_epochs):
            X, y = unison_shuffled_copies(X, y)

            num_batches = int(np.floor(n_examples / batch_size))
            for k in range(num_batches):
                start, end = k * batch_size, min((k + 1) * batch_size, n_examples)
                episode, ytrue = prepareTrainingInput(X[start:end], y[start:end])
                ouptut, loss = self.step(ytrue, episode)
                cost += loss

                # Print epoch number, loss, name and guess
                if k % self.print_every == 0:
                    num_prints += 1
                    print('%.2f%% (%s) %.4f' % ((epoch * n_examples + k * batch_size) /
                                                (self.n_epochs * n_examples) * 100,
                                                timeSince(time_start),
                                                cost))
                    all_losses.append(cost / self.print_every)
                    if self.weight_decay:
                        name_base = "%s_%.2e" % (self.name, self.weight_decay)
                    elif self.alpha:
                        name_base = "%s_%.2e" % (self.name, self.alpha)       
                    else:
                        name_base = "%s_%.2e" % (self.name, self.learning_rate)    
                    torch.save(self.model, '../models/%s.pt' % name_base)
                    np.save('../models/%s.loss' % name_base, all_losses)
                    cost = 0

                    if num_prints % self.eval_every == 0:
                        self.model.eval()
                        tr, tval, te = run_mlp(self.model)
                        self.model.train()
                        snapshots.append((tr, tval, te))
                        np.save('models/%s.snapshots' % name_base, snapshots)
                    
        # todo: early stopping
        print(all_losses)


if __name__ == '__main__':
    kwargs = sys.argv[1:]
    for i in kwargs:
        print(i)
        exec(i)

    # now grid search
    nlr = numprocess
    lr_max = 0.1
    lr_eps = 1e-6
    lrs = np.logspace(np.log10(lr_max*lr_eps), np.log10(lr_max), nlr)

    nwd = numprocess
    wd_max = 1e-2 # was 1e-2
    wd_eps = 1e-6 # was 1e-6
    wds = np.logspace(np.log10(wd_max*wd_eps), np.log10(wd_max), nwd)    

    _, d = xtrain.shape
    n_hidden = 128
    numlayers = 10
    n_output = 2
    model = LR(d, n_output)
    
    random_risk = np.random.permutation(risk)
    t  = Trainer(model=model, name="lreye-random-wd", learning_rate=3.56e-3,
                 alpha=wds[theindex-1], regularization=eye_loss,
                 # using random risk factors
                 risk_factors=Variable(torch.from_numpy(random_risk).float()), 
                 parameters=model.i2o.weight)

    t.fit(xtrain, ytrain)


