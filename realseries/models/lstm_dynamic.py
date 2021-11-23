# -*- encoding: utf-8 -*-
"""The lstm danamic threshold method is the implentation of paper
'Detecting Spacecraft Anomalies Using LSTMs andNonparametric Dynamic Thresholding'
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

from ..utils.errors import get_errors, process_errors
from ..utils.utility import EarlyStopping, load_model, save_model
from .base import BaseModel

__all__ = ['LSTM_dynamic']

class LSTM_dynamic(BaseModel):
    """LSTM Dynamic method.

    Args:
        hidden_size (int, optional): Hidden size of LSTM. Defaults to 128.
        model_path (str, optional): Path for saving and loading model. Defaults
            to './model'.
        dropout (float, optional): Dropout rate. Defaults to 0.3.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        lstm_batch_size (int, optional): Batch size of training LSTM. Defaults
            to 100.
        epochs (int, optional): Epochs of training. Defaults to 50.
        num_layers (int, optional): Number of LSTM layer. Defaults to 2.
        l_s (int, optional): Length of the input sequence for LSTM. Defaults to
            120.
        n_predictions (int, optional): Number of values to predict by input
            sequence. Defaults to 10.
        batch_size (int, optional):  Number of values to evaluate in each batch
            in the prediction stage. Defaults to 32.
        window_size (int, optional): Window_size to use in error calculation.
            Defaults to 50.
        smoothing_perc (float, optional): Percentage of total values used in
            EWMA smoothing. Defaults to 0.2.
        error_buffer (int, optional): Number of values surrounding an error that
            are brought into the sequence. Defaults to 50.
        p (float, optional): Minimum percent decrease between max errors in
            anomalous sequences (used for pruning). Defaults to 0.1.

    Attributes:
        model: The LSTM model.
        y_test: The origin data for calculate error.
        y_hat: The predicted data.
    """

    def __init__(self,
                 hidden_size=128,
                 model_path='./model',
                 dropout=0.3,
                 lr=1e-3,
                 lstm_batch_size=100,
                 epochs=50,
                 num_layers=2,
                 l_s=120,
                 n_predictions=10,
                 batch_size=32,
                 window_size=50,
                 smoothing_perc=0.2,
                 error_buffer=50,
                 p=0.1):

        super(LSTM_dynamic, self).__init__()
        self.batch_size = batch_size
        self.window_size = window_size
        self.smoothing_perc = smoothing_perc
        self.error_buffer = error_buffer
        self.dropout = dropout
        self.lstm_batch_size = lstm_batch_size
        self.epochs = epochs
        self.l_s = l_s
        self.n_predictions = n_predictions
        self.p = p
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.num_layers = num_layers
        self.random_seed = 42

    def fit(self,
            X,
            split=0.25,
            monitor='val_loss',
            patience=10,
            delta=0,
            verbose=True):
        """Train the LSTM model.

        Args:
            X (arrar_like): The 2-D input sequence with shape (n_samples,
                n_features)
            split (float, optional): Fration to split for validation set.
                Defaults to 0.25.
            monitor (str, optional): Monitor the validation loss by setting the
                monitor argument to 'val_loss'. Defaults to 'val_loss'.
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
        """
        X = np.asarray(X)
        os.makedirs(self.model_path, exist_ok=True)
        early_stopping = EarlyStopping(monitor, patience, delta, verbose)
        X, y = _shape_telemetry(X, self.l_s, self.n_predictions)
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        dataset = TensorDataset(X, y)
        val_length = int(split * len(X))
        train_length = len(X) - val_length
        train_dataset, val_dataset = random_split(dataset,
                                                  (train_length, val_length))

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True)
        valid_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True)

        input_shape = X.shape[-1]
        self.model = lstm(
            input_size=input_shape,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=self.n_predictions).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_losses = []
        self.valid_losses = []
        for epoch in trange(1, self.epochs + 1):
            train_batch_losses = []
            valid_batch_losses = []
            self.model.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = nn.MSELoss()(output, target)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                train_batch_losses.append(loss.item())

            self.model.eval()  # prep model for evaluation
            for data, target in valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                # forward pass: compute predicted outputs by passing inputs to
                # the model
                output = self.model(data)
                # calculate the loss
                loss = nn.MSELoss()(output, target)
                # record validation loss
                valid_batch_losses.append(loss.item())

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_batch_losses)
            valid_loss = np.average(valid_batch_losses)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            print(f'train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}')

            # early_stopping call function
            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if epoch % 20 == 0:
                save_model(self.model,
                           Path(self.model_path, f'checkpoint{epoch}.pt'))
        save_model(early_stopping.best_model,
                   Path(self.model_path, 'checkpoint.pt'))

    def predict(self, X):
        """Predict the reconstructed output array y_hat.

        Args:
            X (array_like): The input 2-D array.

        Raises:
            ValueError: Num_batches less than 0.

        Returns:
            ndarray: The predicted array of lstm_encoder_decoder.
        """
        X = np.asarray(X)
        X, y = _shape_telemetry(X, self.l_s, self.n_predictions)
        input_shape = X.shape[-1]
        self.model = lstm(
            input_size=input_shape,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=self.n_predictions).to(self.device)
        self.model = load_model(self.model,
                                os.path.join(self.model_path, 'checkpoint.pt'))
        self.model.eval()
        y_hat = np.array([])
        num_batches = int((X.shape[0] - self.l_s) / self.batch_size)
        if num_batches < 0:
            raise ValueError(
                f'l_s {self.l_s} too large for stream with length {X.shape[0]}.'
            )

        # simulate data arriving in batches
        for i in range(1, num_batches + 2):
            prior_idx = (i - 1) * self.batch_size
            idx = i * self.batch_size
            if i == num_batches + 1:
                idx = X.shape[
                    0]  # remaining values won't necessarily equal batch size
            X_test_period = torch.from_numpy(X[prior_idx:idx])
            X_test_period = X_test_period.to(self.device).float()
            y_hat_period = self.model(X_test_period)
            y_hat_period = y_hat_period.cpu().data.numpy()

            # map predictions n steps ahead to their corresponding timestep
            final_y_hat = []
            for t in range(len(y_hat_period) + self.n_predictions):
                y_hat_t = []
                for j in range(self.n_predictions):
                    if t - j >= 0 and t - j < len(y_hat_period):
                        y_hat_t.append(y_hat_period[t - j][j])
                if t < len(y_hat_period):
                    if y_hat_t.count(0) == len(y_hat_t):
                        final_y_hat.append(0)
                    else:
                        final_y_hat.append(y_hat_t[0])  # first prediction

            y_hat_period = np.array(final_y_hat).reshape(len(final_y_hat), 1)
            y_hat = np.append(y_hat, y_hat_period)

        y_hat = np.reshape(y_hat, (y_hat.size,))
        self.y_test = y
        self.y_hat = y_hat
        print(f'y_hat shape {y_hat.shape},y_test shape {y.shape}')
        return y_hat

    def detect(self, X, smoothed=True):
        """Get anomaly score of input sequence.

        Args:
            X (array_like): Input sequence.
            smoothed (bool, optional): Whether to smooth the errors by EWMA. 
                Defaults to True.

        Returns:
            tuple: (error_seq, error_seq_scores).The error_seq is list that
            stand the anomaly duration. The error_seq_scores is the corresponding
            anomaly score.
        """
        self.predict(X)
        error_seq, error_seq_scores = LSTM_dynamic.obtain_anomaly(
            self.y_test, self.y_hat, self.batch_size, self.window_size,
            self.smoothing_perc, self.p, self.l_s, self.error_buffer, smoothed)
        return error_seq, error_seq_scores
        # self.e_s = get_errors(
        #     self.batch_size,
        #     self.window_size,
        #     self.smoothing_perc,
        #     self.y_test,
        #     self.y_hat,
        #     smoothed=True)
        # self.error_seq, self.error_seq_scores = process_errors(
        #     self.p,
        #     self.l_s,
        #     self.batch_size,
        #     self.window_size,
        #     self.error_buffer,
        #     self.y_test,
        #     self.y_hat,
        #     self.e_s,
        # )
        # print(
        #     f'error_seq: {self.error_seq}, error_seq_scores: {self.error_seq_scores}'
        # )
        # return self.error_seq, self.error_seq_scores

    @staticmethod
    def obtain_anomaly(y_test,
                       y_hat,
                       batch_size,
                       window_size,
                       smoothing_perc,
                       p,
                       l_s,
                       error_buffer,
                       smoothed=True):
        """Obtain anomaly from the origin sequence and reconstructed sequence y_hat.

        Args:
            y_test (ndarray): The origin 1-D signals array of test targets
                corresponding to true values to be predicted at end of each window.
            y_hat (ndarray): The predicted 1-D sequence y_hat for each timestep in y_test
            batch_size (int, optional):  Number of values to evaluate in each
                batch in the prediction stage. Defaults to 32.
            window_size (int, optional): Window_size to use in error calculation.
                 Defaults to 50.
            smoothing_perc (float, optional): Percentage of total values used in
                EWMA smoothing. Defaults to 0.2.
            error_buffer (int, optional): Number of values surrounding an error
                that are brought into the sequence. Defaults to 50.
            p (float, optional): Minimum percent decrease between max errors in
                anomalous sequences (used for pruning). Defaults to 0.1.
            l_s (int, optional): Length of the input sequence for LSTM. Defaults
                to 120.
            smoothed (bool, optional): Whether to smooth the errors by EWMA.
                Defaults to True.
        Returns:
            tuple: (error_seq, error_seq_scores)
        """
        e_s = get_errors(batch_size, window_size, smoothing_perc, y_test, y_hat,
                         smoothed)
        error_seq, error_seq_scores = process_errors(
            p,
            l_s,
            batch_size,
            window_size,
            error_buffer,
            y_test,
            y_hat,
            e_s,
        )
        print(f'error_seq: {error_seq}, error_seq_scores: {error_seq_scores}')
        return error_seq, error_seq_scores


class lstm(nn.Module):

    def __init__(self,
                 input_size=50,
                 hidden_size=64,
                 output_size=10,
                 num_layers=2,
                 dropout=0.5):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)  # shape  (batch, seq, hidden_size)
        x = x[:, -1, :]  # (batch, hidden_size)
        x = self.layer2(x)
        return x


def _shape_telemetry(arr, l_s=100, n_predictions=10):
    """Split the sequence with window length `l_s=100` as train samples.
    The `n_predictions=10` means that the total 10 points after the window
    is assigned as the value to predict.

    Args:
        arr (ndarray): The input 2-D array.
        l_s (int, optional): window size. Defaults to 100.
        n_predictions (int, optional): Number of ahead values to predict.
            Defaults to 10.

    Returns:
        tuple: (X, y)
    """
    dim1 = len(arr) - l_s - n_predictions
    dim2 = l_s + n_predictions
    dim3 = arr.shape[-1]
    data = np.empty((dim1, dim2, dim3), dtype='float')
    for i in range(len(arr) - l_s - n_predictions):
        data[i] = arr[i:i + l_s + n_predictions]
    assert len(data.shape) == 3
    X = data[:, :-n_predictions, :]
    y = data[:, -n_predictions:, 0]  # telemetry value is at position 0
    return X, y
