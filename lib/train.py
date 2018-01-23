import numpy as np
import sys
import torch
from torch import nn
from torch.autograd import Variable
from lib.model import LR
from torch.utils.data import DataLoader
import time, math
from lib.utility import timeSince, data_shuffle, model_auc, calc_loss, model_acc
from lib.utility import var2constvar, logit_elementwise_loss, plotDecisionSurface
from lib.utility import to_np, to_var, gradNorm, check_nan, fig2data, fig2img
from lib.utility import to_cuda, valueNorm
from sklearn.metrics import accuracy_score
from lib.settings import DISCRETE_COLORS
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm
from torchvision.transforms import ToTensor

def np2tensor(x, y):
    return torch.from_numpy(x).float(), torch.from_numpy(y).long()

def prepareData(x, y):
    ''' 
    convert x, y from numpy to tensor
    '''
    return to_var(torch.from_numpy(x).float()), to_var(torch.from_numpy(y).long())

class Trainer(object):
    def __init__(self, model, optimizer=None,
                 loss=None, name="m",
                 lr=0.001, alpha=0.001,
                 risk_factors=None,
                 regularization=None,
                 reg_parameters=None):
        '''
        optimizer: optimization method, default to adam
        reg_parameters: parameters to regularize on
        '''
        self.model = model
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)   
        self.optimizer = optimizer
        if loss is None:
            loss = nn.NLLLoss()
        self.loss = loss
        self.name = name
        if regularization is not None and\
           alpha is not None and\
           reg_parameters is not None: # e.g., eye_loss
            self.loss = regularization(loss, alpha, reg_parameters, risk_factors)

    def step(self, x, y):
        '''
        one step of training
        return yhat, regret
        '''
        self.optimizer.zero_grad()
        yhat = self.model(x)
        regret = self.loss(yhat, y)
        regret.backward()
        self.optimizer.step()
        return yhat, regret.data[0]

    def fitData(self, data, batch_size=100, n_epochs=10, print_every=10,
                valdata=None):
        '''
        fit a model to x, y data by batch
        print_every is 0 if do not wish to print
        '''
        time_start = time.time()
        losses = []
        vallosses = []
        n = len(data.dataset)
        cost = 0 
        
        for epoch in range(n_epochs):

            for k, (x_batch, y_batch) in enumerate(data):
                x_batch, y_batch = to_var(x_batch), to_var(y_batch)
                y_hat, regret = self.step(x_batch, y_batch)
                m = x_batch.size(0)                
                cost += 1 / (k+1) * (regret/m - cost)

                if print_every != 0 and k % print_every == 0:
                    
                    losses.append(cost)
                    # progress, time, avg loss, auc
                    to_print = ('%.2f%% (%s) %.4f %.4f' % ((epoch * n + (k+1) * m) /
                                                           (n_epochs * n) * 100,
                                                           timeSince(time_start),
                                                           cost,
                                                           model_auc(self.model,
                                                                     data)))
                    if valdata is not None:
                        vallosses.append(calc_loss(self.model, valdata, self.loss))
                        to_print += "%.4f" % model_auc(self.model, valdata)
                        
                    print(to_print)
                    torch.save(self.model, 'models/%s.pt' % self.name)
                    np.save('models/%s.loss' % self.name, losses)
                    cost = 0
        return losses, vallosses

    def fitXy(self, x, y, batch_size=100, n_epochs=10, print_every=10,
              valdata=None):
        '''
        fit a model to x, y data by batch
        '''
        n, d = x.shape
        time_start = time.time()
        losses = []
        cost = 0
        
        for epoch in range(n_epochs):
            x, y = data_shuffle(x, y)

            num_batches = math.ceil(n / batch_size)

            for k in range(num_batches):
                start, end = k * batch_size, min((k + 1) * batch_size, n)
                x_batch, y_batch = prepareData(x[start:end], y[start:end])
                y_hat, regret = self.step(x_batch, y_batch)
                m = end-start
                cost += 1 / (k+1) * (regret/m - cost)
                
                if print_every != 0 and k % print_every == 0:
                    losses.append(cost)
                    print('%.2f%% (%s) %.4f' % ((epoch * n + (k+1) * (end-start)) /
                                                (n_epochs * n) * 100, # progress
                                                timeSince(time_start), # time 
                                                cost)) # cost
                    torch.save(self.model, 'models/%s.pt' % self.name)
                    np.save('models/%s.loss' % self.name, losses)
        return losses
                    
    def fit(self, x, y=None, batch_size=100, n_epochs=10, print_every=10,
            valdata=None):
        if y is None:
            return self.fitData(x, batch_size, n_epochs, print_every, valdata)
        else:
            return self.fitXy(x, y, batch_size, n_epochs, print_every, valdata)
            
class InterpretableTrainer(Trainer):
    def __init__(self, switchNet, weightNet, apply_f,
                 lr=0.001,
                 alpha=0.001,
                 beta=0.001,
                 max_grad=0.1,
                 log_name=None,
                 silence=True):
        '''
        optimizer: optimization method, default to adam
        alpha: z entropy weight
        beta: y entropy weight
        max_grad: gradient clipping max
        silence: don't output graph and statement
        '''
        to_cuda(switchNet)
        to_cuda(weightNet)
        
        self.switchNet = switchNet
        self.switch_size = switchNet.switch_size
        self.weightNet = weightNet
        self.apply_f = apply_f

        self.log_dir = 'logs/' + ("" if log_name is None else log_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.name = "name" if log_name is None else log_name
        self.silence = silence
        self.count = 0

        self.optSwitch = torch.optim.Adam(self.switchNet.parameters(), lr=lr)
        self.optWeight = torch.optim.Adam(self.weightNet.parameters(), lr=lr)        
        self.loss = nn.SoftMarginLoss() # logit loss
        self.elementwise_loss = logit_elementwise_loss
        self.max_grad = 0.1
        self.alpha = alpha
        self.beta  = beta

    def sampleZ(self, x):
        n = x.size(0) # minibatch size        
        # determine which line to use        
        prob = to_np(torch.exp(self.switchNet(x)))
        # sample the used line
        z_index = [np.random.choice(self.switch_size, p=prob[i]) for i in range(n)]
        z = np.zeros((n, self.switch_size))
        z[np.arange(n), z_index] = 1
        return to_var(torch.from_numpy(z)).float()
        
    def forward(self, x):
        x = to_var(x.data, volatile=True).float()
        
        # form an explanation
        z = self.sampleZ(x)
        # assert not check_nan(z), to_np(z)
        f = self.weightNet(z)

        # apply f on x
        o = self.apply_f(f, x)
        # assert not check_nan(o)
        return o

    def p_z(self, x, const=False):
        p_z_x = torch.exp(self.switchNet(x))
        res = p_z_x.sum(0) / p_z_x.size(0)
        if const:
            return var2constvar(res) 
        return res

    def entropy(px):
        return px * torch.log(torch.clamp(px, 1e-10, 1))
    
    def L(self, x, y, z, const=False):
        f = self.weightNet(z)
        o = self.apply_f(f, x)
        res = self.elementwise_loss(o, y)
        if const:
            return var2constvar(res) 
        return res
        
    def backward(self, x, y, sample=False, n_samples=30):

        n = x.size(0)

        log_p_z = torch.log(torch.clamp(self.p_z(x, const=True), 1e-10, 1))
        log_p_z = log_p_z.expand(n, log_p_z.size(0))
        # p_z_x = self.switchNet(x)
        log_p_z_x = self.switchNet(x)
        # log_p_z_x = torch.log(torch.clamp(p_z_x, 1e-10, 1))

        # for y_entropy_loss
        p_y_z = torch.ones((2, self.switch_size))        
        zs = to_np(self.sampleZ(x)).argmax(1)
        for z in range(self.switch_size):
            y_given_z = to_np(y)[zs == z]
            for i, label in enumerate([-1, 1]):
                p_y_z[i, z] = float((y_given_z == label).sum())
                if y_given_z.shape[0] > 0:
                    p_y_z[i, z] /= y_given_z.shape[0]

        switch_cost = 0
        weight_cost = 0

        if sample:
            for i in range(n_samples):
                z = self.sampleZ(x)

                # switch net: E_z|x (L(x, y, z)
                # - a * log p(z) - a
                # + b * sum_y p(y|z) log p(y|z))
                # * d log p(z|x) / d theta
                data_loss = self.L(x, y, z)
                z_entropy_loss = - (log_p_z*z).sum(1) - 1

                # assume binary problem
                y_entropy_loss = 0
                for y_query in [0, 1]:
                    pyz = p_y_z[y_query].expand(n, self.switch_size)
                    pyz = (to_var(pyz) * z).sum(1)
                    y_entropy_loss += pyz * torch.log(torch.clamp(pyz, 1e-10, 1))
                
                c =  to_var(data_loss.data) + \
                     self.alpha * z_entropy_loss + \
                     self.beta * y_entropy_loss
                derivative = (log_p_z_x * z).sum(1)
                switch_cost += c * derivative

                # weight net: E_z|x d L(x, y, z) / d theta
                weight_cost += data_loss

            switch_cost /= n_samples
            switch_cost.mean().backward()

            weight_cost /= n_samples
            weight_cost.mean().backward()
        else:
            _data_loss = 0
            _z_entropy_loss = 0
            _y_entropy_loss = 0
            
            p_z_x = to_var(torch.exp(log_p_z_x).data)
            for i in range(self.switch_size):
                z = np.zeros(self.switch_size)
                z[i] = 1
                z = to_var(torch.from_numpy(z).float()).expand(n, self.switch_size)

                # switch net: E_z|x (L(x, y, z)
                # - a * log p(z) - a
                # + b * sum_y p(y|z) log p(y|z))
                # * d log p(z|x) / d theta
                data_loss = self.L(x, y, z)
                z_entropy_loss = - (log_p_z*z).sum(1) - 1
                # assume binary problem
                y_entropy_loss = 0
                for y_query in [0, 1]:
                    pyz = p_y_z[y_query].expand(n, self.switch_size)
                    pyz = (to_var(pyz) * z).sum(1)
                    y_entropy_loss -= pyz * torch.log(torch.clamp(pyz, 1e-10, 1))

                c =  to_var(data_loss.data) + \
                     self.alpha * z_entropy_loss - \
                     self.beta * y_entropy_loss
                derivative = (log_p_z_x * z).sum(1)
                switch_cost += p_z_x[:, i] * c * derivative

                # weight net: E_z|x d L(x, y, z) / d theta
                weight_cost += p_z_x[:, i] * data_loss

                # collect statistics: +1 for transform derivative back to entropy
                _data_loss += p_z_x[:, i] * data_loss
                _z_entropy_loss += p_z_x[:, i] * (z_entropy_loss + 1)
                _y_entropy_loss += p_z_x[:, i] * y_entropy_loss 

            switch_cost.mean().backward()
            weight_cost.mean().backward()

            hz = _z_entropy_loss.mean().data[0]
            hyz = _y_entropy_loss.mean().data[0]

            self.writer.add_scalar('loss/data', _data_loss.mean().data[0],
                                   self.count)
            self.writer.add_scalar('loss/z_entropy',
                                   hz,
                                   self.count)
            self.writer.add_scalar('loss/y_given_z_entropy',
                                   hyz,
                                   self.count)
            self.writer.add_scalar('loss/y_z_entropy',
                                   hz + hyz,
                                   self.count)
            

    def step(self, x, y):
        '''
        one step of training
        return yhat, regret
        '''
        self.optSwitch.zero_grad()
        self.optWeight.zero_grad()        
        yhat = self.forward(x)
        regret = self.loss(yhat, y)
        self.backward(x, y)

        try:
            assert np.isfinite(gradNorm(self.switchNet))
        except:
            print('inf gradient switchNet')

        try:
            assert np.isfinite(gradNorm(self.weightNet))
        except:
            print('inf gradient weightNet')
        
        # clip gradient here
        clip_grad_norm(self.switchNet.parameters(), self.max_grad)
        # per parameter clip
        # for p in self.switchNet.parameters():
        #     if p.grad is None:
        #         continue
        #     p.grad.data = p.grad.data.clamp(-self.max_grad, self.max_grad)
        # for p in self.weightNet.parameters():
        #     if p.grad is None:
        #         continue
        #     p.grad.data = p.grad.data.clamp(-self.max_grad, self.max_grad)
        
        self.optSwitch.step()
        self.optWeight.step()        
        return yhat, regret.data[0]

    def fit(self, data, batch_size=100, n_epochs=10, print_every=10):
        '''
        fit a model to x, y data by batch
        print_every is 0 if do not wish to print
        '''
        time_start = time.time()
        losses = []
        n = len(data.dataset)
        cost = 0
        self.count = 0
        
        for epoch in range(n_epochs):

            for k, (x_batch, y_batch) in enumerate(data):

                x_batch, y_batch = to_var(x_batch).float(), to_var(y_batch).float()
                y_hat, regret = self.step(x_batch, y_batch)
                m = x_batch.size(0)                
                cost += 1 / (k+1) * (regret - cost)

                if print_every != 0 and self.count % print_every == 0:

                    losses.append(cost)
                    
                    # progress, time, avg loss, auc
                    to_print = ('%.2f%% (%s) %.4f' % ((epoch * n + (k+1) * m) /
                                                      (n_epochs * n) * 100,
                                                      timeSince(time_start),
                                                      cost))
                    
                    print(to_print)
                    self.plot(x_batch, y_batch, silence=self.silence)
                    torch.save(self.switchNet,
                               'nonlinear_models/%s^switch.pt' % self.name)
                    torch.save(self.weightNet,
                               'nonlinear_models/%s^weight.pt' % self.name)
                    np.save('nonlinear_models/%s.loss' % self.name, losses)


                    self.writer.add_scalar('switch/grad_norm', gradNorm(self.switchNet),
                                           self.count)
                    self.writer.add_scalar('weight/grad_norm', gradNorm(self.weightNet),
                                           self.count)
                    self.writer.add_scalar('data/train_loss', cost, self.count)
                    
                    for tag, value in self.switchNet.named_parameters():
                        tag = tag.replace('.', '/')
                        self.writer.add_histogram(tag, to_np(value), self.count)
                        self.writer.add_histogram(tag+'/grad', to_np(value.grad),
                                                  self.count)

                    cost = 0                        
                    
                self.count += 1

        self.plot(x_batch, y_batch, inrange=True, silence=self.silence)

        return losses

    def plot(self, x, y, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5,
             inrange=False, silence=False):
        '''
        inrange: if true fix y axis from ymin to ymax
        '''
        
        plt.figure(figsize=(10,10))
        _ = plotDecisionSurface(self.forward, xmin, xmax, ymin, ymax,
                                multioutput=False, colors=['white', 'black'])
        
        plt.xlim([xmin, xmax])
        if inrange:
            plt.ylim([ymin, ymax])

        output = to_np(self.switchNet(x)) # p(z) = E_x p(z|x)
        choices = output.argmax(1)
        probs =  np.zeros(self.switch_size)
        for i in range(self.switch_size):
            probs[i] = (choices == i).sum() / len(choices)
            
        c = DISCRETE_COLORS
        for i, (t1, t2, b) in enumerate(self.weightNet.explain()):
            # t1 x + t2 y + b = 0
            # y = - (t1 x + b) / t2
            def l(x):
                return - (t1 * x + b) / t2
            alpha = probs[i] / max(probs)
            plt.plot([xmin, xmax], [l(xmin), l(xmax)], c=c[i], alpha=alpha)
            
            xmid = (xmin + xmax) / 2
            direction_norm = np.sqrt((np.array([t1, t2])**2).sum())
            plt.plot([xmid, xmid+t1/direction_norm*0.1],
                     [l(xmid), l(xmid)+t2/direction_norm*0.1],
                     c=c[i], alpha=alpha)

        im = ToTensor()(fig2img(plt.gcf()))
        self.writer.add_image('weight', im,
                              self.count)
        if silence:
            plt.close()
        else:
            plt.show()

        if not silence:
            for i in range(self.switch_size):
                print('probability of choosing', c[i], 'is', probs[i])
        

            p_y_z = torch.ones((2, self.switch_size))        
            zs = to_np(self.sampleZ(x)).argmax(1)
            for z in range(self.switch_size):
                y_given_z = to_np(y)[zs == z]
                for i, label in enumerate([-1, 1]):
                    p_y_z[i, z] = float((y_given_z == label).sum())
                    if y_given_z.shape[0] > 0:
                        p_y_z[i, z] /= y_given_z.shape[0]
            for i in range(self.switch_size):
                print('p(y=-1|z="%s")' % c[i], 'is', p_y_z[0, i])

        # plot cluster assignment
        x = plotDecisionSurface(self.switchNet.forward, xmin, xmax, ymin, ymax,
                                colors=c)
        im = ToTensor()(fig2img(plt.gcf()))
        self.writer.add_image('switch', im,
                              self.count)

        if silence:
            plt.close()
        else:
            plt.show()
