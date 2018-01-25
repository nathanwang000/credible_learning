import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.utility import to_np, to_var, check_nan

###### non linear credibility start #############
# softmax version
class Switch(nn.Module):
    def __init__(self, input_size, switch_size, mtl=False):
        '''
        mtl: is multi-task-learning or not
             if yes, assume the last dimension is task number
        '''
        super().__init__()
        self.i2o = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, switch_size),            
        )
        self.logsoftmax = nn.LogSoftmax()
        self.switch_size = switch_size
        self.mtl = mtl
        self.input_size = input_size
        
    def forward(self, x):
        if self.mtl: # the last one is task number
            x = x[:,-1:]
            m, d = x.size()
            # turn into onehot
            if self.input_size > 1:
                x_onehot = torch.FloatTensor(m, self.input_size)
                x_onehot.zero_()
                x_onehot.scatter_(1, x.cpu().data.long(), 1)
                x = to_var(x_onehot)

        o = self.i2o(x)
        # assert not check_nan(o)
        return self.logsoftmax(o)

class Weight(nn.Module):
    def __init__(self, switch_size, param_size):
        '''
        param_size: number of parameters for the interpretable model
        '''
        super().__init__()
        self.switch_size = switch_size
        self.i2o = nn.Sequential(
            nn.Linear(switch_size, 32),
            nn.ReLU(),
            nn.Linear(32, param_size), 
            # nn.Linear(switch_size, 128),
            # nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 32),
            # nn.ReLU(),            
            # nn.Linear(32, param_size),            
        )
        
    def forward(self, x):
        return self.i2o(x)

    def explain(self):
        explanations = []
        for i in range(self.switch_size):
            x = np.zeros(self.switch_size)
            x[i] = 1
            x = to_var(torch.from_numpy(x)).float()
            explanations.append(list(to_np(self.forward(x))))
        return explanations

def apply_linear(f, x): # for linear model
    return (f[:,:-1] * x).sum(1) + f[:,-1]    

class WeightIndependent(nn.Module):
    def __init__(self, switch_size, param_size):
        '''
        param_size: number of parameters for the interpretable model
        independent switch_size number of lines
        '''
        super().__init__()
        self.switch_size = switch_size
        self.classifiers = []
        for i in range(switch_size):
            self.classifiers.append(nn.Linear(param_size, 1))
        
    def forward(self, x):
        return self.i2o(x)

    def explain(self):
        explanations = []
        for i in range(self.switch_size):
            x = np.zeros(self.switch_size)
            x[i] = 1
            x = to_var(torch.from_numpy(x)).float()
            explanations.append(list(to_np(self.forward(x))))
        return explanations

###### non linear credibility end ###############

class LR(nn.Module): # logistic regression with 2 neurons
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.i2o = nn.Linear(input_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.output_size = output_size
        self.input_size = input_size
        
    def forward(self, input):
        output = self.softmax(self.i2o(input))
        return output

class MLP(nn.Module):
    # so this essentially shares weights for each layer, may not be what  I want
    def __init__(self, input_size, hidden_size, output_size, numlayers=5):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.nl  = nn.LeakyReLU()
        self.do  = nn.Dropout()
        self.softmax = nn.LogSoftmax()
        
        self.numlayers = numlayers
        self.output_size = output_size
        self.input_size = input_size
        
    def forward(self, input):
        hidden = self.i2h(input)
        for _ in range(self.numlayers-1):
            neuron = self.nl(hidden)
            hidden = self.h2h(self.do(neuron))
        output = self.softmax(self.h2o(hidden))
        return output

class RNN(nn.Module): # simple 1 layer rnn
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = F.tanh(self.i2h(combined))
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return to_var(torch.zeros(1, self.hidden_size))

class RNN2(nn.Module): # canonical RNN pytorch with added output layer
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN2, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNNCell(input_size, hidden_size)

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()


    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.rnn(input, hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return to_var(torch.zeros(1, self.hidden_size))
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTMCell(input_size, hidden_size)

        # note: I should really just use input size
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()


    def forward(self, input, hidden):
        h_n, c_n  = hidden
        combined = torch.cat((input, h_n),1)
        hidden = self.rnn(input, hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (to_var(torch.zeros(1, self.hidden_size)),
                to_var(torch.zeros(1, self.hidden_size)))
        
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = torch.nn.GRUCell(input_size, hidden_size)

        # note: I should really just use input size
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()


    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.rnn(input, hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return to_var(torch.zeros(1, self.hidden_size))
        
