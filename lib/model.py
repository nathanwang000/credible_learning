import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

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


class MLP2(nn.Module): # non shared version, no dropout
    def __init__(self, input_size, hidden_size, output_size, numlayers=3, with_do=False):
        super(MLP2, self).__init__()
        m = nn.Sequential()
        m.add_module('i2h', nn.Linear(input_size, hidden_size))
        m.add_module('nl0', nn.LeakyReLU())
        if with_do: m.add_module('do0', nn.Dropout())
        
        for i in range(1,numlayers-1):
            m.add_module('h2h%d'% i, nn.Linear(hidden_size, hidden_size))
            m.add_module('nl%d' % i, nn.LeakyReLU())
            if with_do: m.add_module('do%d' % i, nn.Dropout())
            
        m.add_module('h2o', nn.Linear(hidden_size, output_size))

        self.m = m
        self.softmax = nn.LogSoftmax()
        
        self.numlayers = numlayers
        self.output_size = output_size
        self.input_size = input_size
        self.with_do = with_do
        
    def forward(self, input):
        output = self.softmax(self.m(input))
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
        return Variable(torch.zeros(1, self.hidden_size))

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
        return Variable(torch.zeros(1, self.hidden_size))
    
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
        return (Variable(torch.zeros(1, self.hidden_size)),
                Variable(torch.zeros(1, self.hidden_size)))
        
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
        return Variable(torch.zeros(1, self.hidden_size))
        
############### old implementation without using cell ####################
class RNN_old(nn.Module): # simple 1 layer rnn
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_old, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

class RNN2_old(nn.Module): # canonical RNN pytorch with added output layer
    def __init__(self, input_size, hidden_size, output_size, numlayer=1):
        super(RNN2_old, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size, hidden_size, numlayer)
        self.numlayer = numlayer

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()


    def forward(self, input, hidden):
        # input: seq_len, batch, hiddensize
        # hidden: num_layer, batch, hiddensize
        combined = torch.cat((input, hidden[self.numlayer-1]), 1)
        _input = input.unsqueeze(0)  # add a fake squence dimension
        _, hidden = self.rnn(_input, hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(self.numlayer, 1, self.hidden_size))

class LSTM_old(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, numlayer=1):
        super(LSTM_old, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(input_size, hidden_size, numlayer)
        self.numlayer = numlayer

        # note: I should really just use input size
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()


    def forward(self, input, hidden):
        # input: seq_len, batch, hiddensize
        # hidden: num_layer, batch, hiddensize
        h_n, c_n  = hidden
        combined = torch.cat((input, h_n[self.numlayer-1]), 1)
        _input = input.unsqueeze(0)  # add a fake squence dimension
        _, hidden = self.rnn(_input, hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (Variable(torch.zeros(self.numlayer, 1, self.hidden_size)),
                Variable(torch.zeros(self.numlayer, 1, self.hidden_size)))
