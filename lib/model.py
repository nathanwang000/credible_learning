import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LR(nn.Module): # logistic regression with 2 neurons
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.i2o = nn.Linear(input_size, output_size)
        self.softmax = nn.LogSoftmax()
        
        self.output_size = output_size
        self.input_size = input_size
        
    def forward(self, input):
        output = self.softmax(self.i2o(input))
        return output
