# -*- encoding: utf-8 -*-
# @Author: Wenkai Li
"""Introduction of seqvl.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from .base import BaseModel

__all__=['SeqVL']
class SeqVL(BaseModel):

    def __init__(self,
                 contamination=0.1,
                 name='SeqVL',
                 num_epochs=250,
                 batch_size=1,
                 lr=1e-3,
                 lr_decay=0.8,
                 lamb=10,
                 clip_norm_value=12.0,
                 data_split_rate=0.5,
                 window_size=30,
                 window_count=300,
                 h_dim=24,
                 z_dim=5,
                 l_h_dim=24):
        super(SeqVL, self).__init__(contamination=contamination)
        self.name = name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.lamb = lamb
        self.clip_norm_value = clip_norm_value
        self.data_split_rate = 0.5
        self.window_size = window_size
        self.window_count = window_count
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.l_h_dim = l_h_dim
        self.seqvl = None
        self.trained = False

    def reshape_for_training(self, X):
        """Reshape the data for training.

        Args:
            X (ndarray): 1-D time series

        Returns:
            tuple: input with shape [-1, window_count, window_size],
                label with shape [-1, window_count]
        """

        samples = np.array([
            X[i:i + self.window_size + 1]
            for i in range(X.shape[0] - self.window_size)
        ],
                           dtype=np.float32)
        seg_seqs = np.array([
            samples[i:i + self.window_count]
            for i in range(0, samples.shape[0] - self.window_count +
                           1, self.window_count)
        ])
        data = seg_seqs[:, :, :-1]
        labels = seg_seqs[:, :, -1]

        return data, labels

    def fit(self, X):
        """Train the model.

        Args:
            X (array_like): Input sequence.
        """

        def loss_fn(x_hat, x, mu, logvar, y_hat, label):
            mse_x = F.mse_loss(x_hat, x)
            KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
            y_hat = y_hat.squeeze(dim=-1)
            mse_y = F.mse_loss(y_hat, label)
            return mse_x + KLD + mse_y

        X = np.asarray(X)
        train_data, labels = self.reshape_for_training(X)

        self.seqvl = SeqVLModule(self.window_size, self.window_count,
                                 self.h_dim, self.z_dim, self.l_h_dim)
        self.seqvl.train()
        optimizer = torch.optim.Adam(self.seqvl.parameters(), self.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=self.lr_decay)

        for epoch in range(self.num_epochs):
            for i in range(0, train_data.shape[0], self.batch_size):
                optimizer.zero_grad()

                x_hat, mu, logvar, y_hat = self.seqvl(
                    train_data[i:i + self.batch_size])
                loss = loss_fn(x_hat, train_data[i:i + self.batch_size], mu,
                               logvar, y_hat, labels[i:i + self.batch_size])
                loss.backward()

                torch.nn.utils.clip_grad_norm(self.seqvl.parameters(),
                                              self.clip_norm_value)
                optimizer.step()
                if epoch % 20 == 0:
                    print("Epoch[{}/{}] Loss: {:.3f}".format(
                        epoch + 1, self.num_epochs,
                        loss.item() / self.batch_size))
            scheduler.step()

        self.trained = True

    def reshape_for_test(self, X):
        """Reshape the data gor test.

        Args:
            X (array_like): Input data.
        Returns:
            ndarray: Reshaped data.
        """
        samples = np.array([
            X[i:i + self.window_size]
            for i in range(X.shape[0] - self.window_size + 1)
        ],
                           dtype=np.float32)
        return samples

    def detect(self, X, thres):
        """Detect the data by trained model.

        Args:
            X (array_like): 1-D time series with length L.
            thres (float): Threshold.

        Returns:
            dict: Dict containing results. 0-1 sequence indicates whether the last point  of a window is
                an anomaly. length: L - window_size + 1
        """

        def reconstruct(model, data):
            data.view(-1, self.window_size)
            h = model.encoder(data)
            mu, logvar = torch.chunk(h, 2, dim=1)
            z = model.reparameterize(mu, logvar)
            recon = model.decoder(z)
            return recon

        def get_anomaly_score(model, test_windows):
            x_hat = reconstruct(model, test_windows)
            ano_score = x_hat.detach().numpy()[:, -1] - test_windows.detach(
            ).numpy()[:, -1]
            ano_score = np.array([x * x for x in ano_score])
            return ano_score

        self.seqvl.eval()
        samples = self.reshape_for_test(X)
        ano_score = get_anomaly_score(self.seqvl, samples)
        res = np.where(ano_score < thres, 0, 1)

        return {
            'origin_series': X[self.window_size - 1:],
            'result': res,
            'score': ano_score
        }


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class SeqVLModule(nn.Module):

    def __init__(self, window_size, window_count, h_dim, z_dim, l_h_dim):
        super(SeqVLModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, window_size))

        self.lstm = nn.LSTM(window_size, l_h_dim, 1, batch_first=True)

        # fc maybe modified, in single step or the total window_count
        # single step version, I think the fc should be shared, so currently I prefer
        # to use single step version
        self.fc = nn.Linear(l_h_dim, 1)

        self.window_size = window_size
        self.window_count = window_count

        # window_count version
        # self.fc_list = [nn.Linear(l_h_dim, 1) for _ in range(window_count)]

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = to_var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z

    # @torchsnooper.snoop()
    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        '''
        x:[batch, window_count, window_size]
        '''
        x = x.view(-1, self.window_size)
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(-1, self.window_count, self.window_size)

        output, (hn, cn) = self.lstm(x_hat)
        # output : [batch_size, seq_len, l_h_dim]
        y_hat = self.fc(output)

        return x_hat, mu, logvar, y_hat
