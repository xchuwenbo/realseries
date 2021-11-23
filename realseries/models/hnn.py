# -*- encoding: utf-8 -*-
''' 
The models(HNN, Deep-ensemble, MC-dropout, CRMMD...) for time series forcasting and uncertainty prediction. 
'''
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
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import scipy.stats as st
from ..utils.preprocess import normalization
from ..utils.utility import EarlyStopping, aleatoric_loss
from .base_rnn import rnn_base
from .base import BaseModel


class HNN(BaseModel):
    """HNN forecaster for uncertainty prediction. 

    Args:
        kernel_type (str, optional): Type of recurrent net (RNN,
            LSTM, GRU). Defaults to 'LSTM'.
        input_size (int, optional): Size of rnn input features. Defaults to 128.
        hidden_sizes (list, optional): Number of hidden units per layer. Defaults to [128,64].
        prediction_window_size (int, optional): Prediction window size. Defaults to 1.
        activation (str,optional): The activation func to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'relu'``
        dropout_rate (float, optional): Defaults to 0.2.
        variance (bool, optional): Whether to add a variance item at the last layer to indicate uncertainty. Default to True
        lr (float, optional): Learning rate. Defaults to 0.0002.
        weight_decay (float, optional): Weight decay. Defaults to 1e-4.
        grad_clip (int, optional): Gradient clipping. Defaults to 10.
        epochs (int, optional): Upper epoch limit. Defaults to 200.
        batch_size (int, optional): Batch size. Defaults to 1024.
        window_size (int, optional): LSTM input sequence length. Defaults to 15.
        model_path (str, optional): The path to save or load model. Defaults to './model'.
        seed (int, optional): Seed. Defaults to 1111.

    Attributes:
        model: HNN model.
    """
    def __init__(
            self,
            kernel_type='LSTM',
            input_size=128,
            hidden_sizes=[128, 64],
            prediction_window_size=1,
            activation='tanh',
            dropout_rate=0.2,
            variance = True,
            lr=0.0002,
            weight_decay=1e-3,
            grad_clip=10,
            epochs=200,
            batch_size=1024,
            window_size=15,
            model_path='./model',
            seed=1111,):
        super(HNN, self).__init__()

        self.kernel_type = kernel_type
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.variance = variance
        self.dropout_rate = dropout_rate
        self.prediction_window_size = prediction_window_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.rnn_model = rnn_base(kernel_type,input_size,\
            hidden_sizes,prediction_window_size,activation,dropout_rate,True).to(self.device)
        self.model = None
    
    def fit(self,
            train_data,
            train_label,
            val_data,
            val_label,
            patience=50,
            delta=0,
            verbose=True):
        """Train the LSTM model.

        Args:
            train_data (numpy array): The 3-D input sequence (n_samples,window_size,n_features)
            train_label (numpy array): The 2-D input sequence (n_samples,prediction_window_size)
            val_data (numpy array): The 3-D input sequence (n_samples,window_size,n_features)
            val_label (numpy array): The 2-D input sequence (n_samples,prediction_window_size)
            patience (int, optional): Patience argument represents the number of
                epochs before stopping once your loss starts to increase (stops
                improving). Defaults to 10.
            delta (int, optional): A threshold to whether quantify a loss at
                some epoch as improvement or not. If the difference of loss is
                below delta, it is quantified as no improvement. Better to leave
                it as 0 since we're interested in when loss becomes worse.
                Defaults to 0.
            verbose (bool, optional): Verbose decides what to print. Defaults to
                True.
        Attributes:
            model: a trained model to save to model_path/checkpoint_hnn.pt.
        """
        os.makedirs(self.model_path, exist_ok=True)
        early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=verbose)
        train_data, train_label = torch.from_numpy(train_data).float(), torch.from_numpy(train_label).float()
        val_data, val_label = torch.from_numpy(val_data).float(), torch.from_numpy(val_label).float()
        train_dataset = TensorDataset(train_data,train_label)
        val_dataset = TensorDataset(val_data,val_label)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True)
        valid_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True)
        optimizer = torch.optim.Adam(self.rnn_model.parameters(), \
        lr=self.lr, weight_decay=self.weight_decay)
        self.train_losses = []
        self.valid_losses = []

        for epoch in range(1, self.epochs + 1):
            train_batch_losses = []
            valid_batch_losses = []
            self.rnn_model.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output,log_var = self.rnn_model(data)
                # loss = nn.MSELoss()(output, target)
                aleatoric_loss_func = aleatoric_loss()
                loss = aleatoric_loss_func(target, output, log_var)
                self.rnn_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnn_model.parameters(),self.grad_clip)
                optimizer.step()
                train_batch_losses.append(loss.item())

            self.rnn_model.eval()  # prep model for evaluation
            for data, target in valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                # forward pass: compute predicted outputs by passing inputs to
                # the model
                output,log_var = self.rnn_model(data)
                # calculate the loss
                loss = aleatoric_loss_func(target, output, log_var)
                # record validation loss
                valid_batch_losses.append(loss.item())

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_batch_losses)
            valid_loss = np.average(valid_batch_losses)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            if epoch % 20 == 0:
                print(f'epoch: {epoch:} -> train_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}')

            # early_stopping call function
            early_stopping(valid_loss, self.rnn_model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if epoch % 200 == 0:
                self.save_model(Path(self.model_path, f'checkpoint{epoch}.pt'))
        self.save_model(Path(self.model_path, 'checkpoint_hnn.pt'))

        pass
    
    def evaluation_model(self, scaler, test_data, test_label, t=1, confidence=95):
        """Get predictive intervals and evaluation scores.

        Args:
            scaler: receive the scaler of data loader
            test_data (numpy array): The 3-D input sequence (n_samples,window_size,n_features)
            test_label (numpy array): The 2-D input sequence (n_samples,prediction_window_size)
            t (optional, int):  the forecasting horizon, default to 1
            confidence (optional, int): the confidence of predictive intervals. 
                Default to 95, output 95% predictive intervals.
            
        Returns:
            PIs (two numpy arrays): the lower bound and the upper bound arrays of the predictive intervals for test data.
            rmse (float): the rmse score
            calibration error (float): the uncertainty evaluation score for test data.

        """
        test_data, test_label = torch.from_numpy(test_data).float(), torch.from_numpy(test_label).float()
        test_data, test_label = test_data.to(self.device), test_label.to(self.device)
        # self.model = load_model(self.rnn_model,os.path.join(self.model_path, 'checkpoint.pt'))
        self.model.eval()
        pre_mean, logvar = self.model(test_data)
        pre_mean, logvar = pre_mean.cpu().data.numpy(), logvar.cpu().data.numpy()
        test_data, test_label = test_data.cpu().data.numpy(), test_label.cpu().data.numpy()
        pre_std = np.sqrt(np.exp(logvar))
        # recover raw data
        if scaler:
            inversed = scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0],-1),test_label),1))
            test_label = inversed[:,-test_label.shape[1]:]
            inversed = scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0],-1),pre_mean),1))
            pre_mean = inversed[:,-pre_mean.shape[1]:]
            inversed = scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0],-1),pre_std),1))
            pre_std = inversed[:,-pre_std.shape[1]:]
        #print(pre_std.shape,pre_mean.shape)
        rmse = np.sqrt(mean_squared_error(pre_mean, test_label))
        r_square = r2_score(pre_mean,test_label)
        print('R2:',r_square)
        # get confidence interval by normal distribution
        conf = float(confidence)/100.
        interval_low, interval_up = st.norm.interval(conf, loc=pre_mean, scale=pre_std)
        #print(interval_low.shape)
        interval_low = interval_low.reshape(interval_low.shape[0], -1)
        interval_up = interval_up.reshape(interval_up.shape[0], -1)
        k_l = interval_low < test_label
        k_u = interval_up > test_label
        picp = np.mean(k_u * k_l)
        cali_error = np.around(np.abs(picp-conf), decimals=3)
        # mpiw = np.around(np.mean(interval_up - interval_low), decimals=3)
        return interval_low, interval_up, rmse, cali_error
    
    def forecast(self, scaler, test_data, t=1, confidence=95, is_uncertainty=True):
        """Get predictive intervals and evaluation scores.

        Args:
            scaler: receive the scaler of data loader
            test_data (numpy array): The 3-D input sequence (n_samples,window_size,n_features)
            t (optional, int):  the forecasting horizon, default to 1
            confidence (optional, int): the confidence of predictive intervals. 
                Default to 95, output 95% predictive intervals.
            is_uncertainty (optional, bool): whether to get uncertainty, 
                if true, outputing PIs, if false, outputing means. Defaults to True.
            
        Returns:
            PIs (two numpy arrays): the lower bound and the upper bound arrays of the predictive intervals for test data.

        """
        test_data = torch.from_numpy(test_data).float()
        test_data = test_data.to(self.device)
        # self.model = load_model(self.rnn_model,os.path.join(self.model_path, 'checkpoint.pt'))
        self.model.eval()
        pre_mean, logvar = self.model(test_data)
        pre_mean, logvar = pre_mean.cpu().data.numpy(), logvar.cpu().data.numpy()
        test_data = test_data.cpu().data.numpy()
        pre_std = np.sqrt(np.exp(logvar))
        # recover raw data
        if scaler:
            inversed = scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0],-1),pre_mean),1))
            pre_mean = inversed[:,-pre_mean.shape[1]:]
            inversed = scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0],-1),pre_std),1))
            pre_std = inversed[:,-pre_std.shape[1]:]
        # get confidence interval by normal distribution
        conf = float(confidence)/100.
        interval_low, interval_up = st.norm.interval(conf, loc=pre_mean, scale=pre_std)
        if is_uncertainty:
            return interval_low, interval_up
        else:
            return pre_mean

    def save_model(self, model_path=Path('./model', 'checkpoint_hnn.pt')):
        """Save pytorch model.
        
        Args:
            model_path (string or path): Path for saving model.
        """    
        print("saving the model to %s" % model_path)
        try:
            torch.save(self.rnn_model.state_dict(), model_path)
        except:
            torch.save(self.rnn_model, model_path)
    
    def load_model(self, path=Path('./model', 'checkpoint_hnn.pt')):
        """Load Pytorch model.
    
        Args:
            model_path (string or path): Path for loading model.
        
        """    
        print("loading %s" % path)
        model = self.rnn_model
        with open(path, 'rb') as f:
            pretrained = torch.load(f, map_location=lambda storage, loc: storage)
            model_dict = model.state_dict()
            pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
            model_dict.update(pretrained)
            model.load_state_dict(model_dict)
        self.model = model
