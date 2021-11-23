# -*- encoding: utf-8 -*-
''' 
The RNN base model for time series forcasting and uncertainty prediction. 
'''
# Author: Peng Cui 
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.autograd import Variable
from tqdm import tqdm

__all__ = ['rnn_base']
class rnn_base(nn.Module):
    """RNN base model for forcasting and uncertainty prediction. 

    Args:
        kernel_type (str, optional): Type of recurrent net (RNN,
            LSTM, GRU). Defaults to 'LSTM'.
        input_size (int, optional): Size of rnn input features. Defaults to 128.
        hidden_sizes (list, optional): Number of hidden units per layer. Defaults to [128,64].
        prediction_window_size (int, optional): Prediction window size. Defaults to 10.
        activation (str,optional): The activation func to use for rnn kernel. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        dropout_rate (float, optional): Defaults to 0.2.
        variance (bool, optional): Whether to add a variance item at the last layer to indicate uncertainty. Default to True.

    Attributes:
        model: rnn model.
    """
    def __init__(
            self,
            kernel_type='LSTM',
            input_size=128,
            hidden_sizes=[128, 64],
            prediction_window_size=1,
            activation='tanh',
            dropout_rate=0.2,
            variance = True):
        super(rnn_base, self).__init__()

        self.kernel_type = kernel_type
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.variance = variance
        self.dropout_rate = dropout_rate
        self.prediction_window_size = prediction_window_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        #print(getattr(nn, kernel_type))
        if kernel_type in ['LSTM', 'GRU']:
            self.rnn_input_layer = getattr(nn, kernel_type)(\
                input_size, hidden_sizes[0], num_layers=1, batch_first=True)
            self.rnn_hidden_layers = nn.ModuleList([getattr(nn, kernel_type)(\
                hidden_sizes[i], hidden_sizes[i+1], num_layers=1, batch_first=True) for i in range(len(hidden_sizes)-1)])
        else:
            self.rnn_input_layer = nn.RNN(input_size, hidden_sizes[0], \
                num_layers=1, batch_first=True, nonlinearity=activation)
            self.rnn_hidden_layers = nn.ModuleList([nn.RNN(\
                hidden_sizes[i], hidden_sizes[i+1], num_layers=1, batch_first=True) for i in range(len(hidden_sizes)-1)])

        self.pre_mean = nn.Linear(hidden_sizes[-1], prediction_window_size)
        self.get_var = nn.Linear(hidden_sizes[-1], prediction_window_size)

    
    def forward(self, X):
        in_size, seq_len, _ = X.size()
        state = self.init_hidden_state(in_size, self.hidden_sizes[0])
        output, state_hidden = self.rnn_input_layer(X, state)
        output = F.dropout(output, p=self.dropout_rate, training=True)
        for i in range(len(self.hidden_sizes)-1):
            state = self.init_hidden_state(in_size, self.hidden_sizes[i+1])
            output, state_hidden = self.rnn_hidden_layers[i](output, state)
            output = F.dropout(output, p=self.dropout_rate, training=True)
        #print(type(state_hidden), state_hidden[0].shape,state_hidden[0].shape,output.shape)
        pre_mean = self.pre_mean(state_hidden[0].squeeze(0))
        if self.variance:
            pre_var = self.get_var(state_hidden[0].squeeze(0))
            return pre_mean.reshape(len(pre_mean),self.prediction_window_size), pre_var.reshape(len(pre_var),self.prediction_window_size)
        else:
            return pre_mean.reshape(len(pre_mean),self.prediction_window_size)
    
    def init_hidden_state(self, in_size, out_size):
        hidden_state = Variable(torch.zeros(1, in_size, out_size))
        cell_state = Variable(torch.zeros(1, in_size, out_size))
        if self.kernel_type in ['RNN','GRU']:
            return hidden_state.to(self.device)
        else:
            return hidden_state.to(self.device), cell_state.to(self.device)


        
