# @Author: Wenkai Li

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from .base import BaseModel
from ..utils.segment import BatchSegment

__all__ = ['VAE_Dense']

class VAE_Dense(BaseModel):
    """The Donut-VAE version for anomaly detection

    Args:
        window_size (int). Sliding window size.
        channels (int): Channel count of the input signals.
        name (str, optional): Model name. Defaults to 'VAE_Dense'.
        num_epochs (int, optional): Epochs for model training. Defaults to 256.
        batch_size (int, optional): Batch size for model training. Defaults to 256.
        lr ([type], optional): Learning rate. Defaults to 1e-3.
        lr_decay (float, optional): Learning rate decay. Defaults to 0.8.
        clip_norm_value (float, optional): Gradient clip value. Defaults to 12.0.
        weight_decay ([type], optional): L2 regularization. Defaults to 1e-3.
        h_dim (int, optional): Hidden dim between x and z for VAE's encoder and decoder Defaults to 200.
        z_dim (int, optional): Defaults to 20.

    Attributes:
        model: VAE model built by torch.
    """

    def __init__(self,
                 window_size,
                 channels,
                 name='VAE_Dense',
                 num_epochs=256,
                 batch_size=64,
                 lr=1e-3,
                 lr_decay=0.8,
                 clip_norm_value=12.0,
                 weight_decay=1e-3,
                 h_dim=200,
                 z_dim=20):
        super(VAE_Dense, self).__init__()
        self._name = name
        self._window_size = window_size
        self._channels = channels
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._lr = lr
        self._lr_decay = lr_decay
        self._clip_norm_value = clip_norm_value
        self._weight_decay = weight_decay        
        self._h_dim = h_dim
        self._z_dim = z_dim
        self._vae = VAE_Dense_Module(self._window_size * self._channels,
                              self._h_dim,
                              self._z_dim)
        self._trained = False

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def reform(self, x):
        return x.view(x.shape[0], self._window_size, self._channels)

    def fit(self, X):
        """Train the model

        Args:
            X (array_like): The input 2-D array.
                            The first dimension denotes timesteps.
                            The second dimension denotes the signal channels.
        """
        def loss_vae(x_hat, x, mu, logvar):
            # print(x_hat.shape)
            mse_x = F.mse_loss(x_hat, x)
            mse_x = mse_x * x.shape[1]
            # print(mse_x.shape)
            # KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
            KLD = 1 + logvar - mu ** 2 - logvar.exp()
            # print(KLD.shape)
            KLD = torch.sum(KLD, dim=-1)
            # print(KLD.shape)
            KLD *= -0.5
            # print(KLD.shape)
            return torch.mean(mse_x + KLD), torch.mean(mse_x), torch.mean(KLD)

        train_gen = BatchSegment(len(X),
                                 self._window_size,
                                 self._batch_size,
                                 shuffle=True,
                                 discard_last_batch=True)

        self._vae.train()
        optimizer = torch.optim.Adam(self._vae.parameters(),
                                     lr=self._lr,
                                     weight_decay=self._weight_decay)
        scheduler = StepLR(optimizer, step_size=10,
                           gamma=self._lr_decay)

        for epoch in range(self._num_epochs):
            i = 0
            for (batch_x, ) in train_gen.get_iterator([np.asarray(X, dtype=np.float32)]):
                optimizer.zero_grad()

                batch_x = torch.from_numpy(batch_x)
                batch_x = self.flatten(batch_x)
                x_hat, mu, logvar, x_mean, x_logvar = self._vae(batch_x)
                loss, mse, kld = loss_vae(x_hat, batch_x, mu, logvar)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._vae.parameters(),
                                              self._clip_norm_value)
                optimizer.step()

                if i % 90 == 0:
                    print("Epoch[{}/{}] Loss: {:.8f} MSE: {:.8f} KLD: {:.8f}".format(epoch + 1, self._num_epochs, loss.item(), mse.item(), kld.item()))
                
                i += 1
    
            scheduler.step()

        self._trained = True

    def detect(self, X):
        """ Get anomaly score of input sequence.

        Args:
            X (array_like): Input sequence.

        Returns:
            A dict with attributes:
                origin_series: ndarray [timesteps, channels], Origin time series
                recon_series: ndarray [timesteps, channels], Reconstruct time series
                score: ndarray [timesteps, channels], Corresponding anomaly score.
        """

        def data_to_windows(data, window_size, window_step=1):
            windows = np.array([data[i: i + window_size] for i in range(0, len(data) - window_size + 1, window_step)], dtype=np.float32)
            return windows
        
        self._vae.eval()
        test_windows = data_to_windows(X, self._window_size)
        test_windows_input = self.flatten(torch.from_numpy(test_windows))
        recon_x, _, _, recon_mean, recon_logvar = self._vae(test_windows_input)

        origin_signals = test_windows[:, -1]
        recon_signals = self.reform(recon_mean).detach().numpy()[:, -1]
        ano_score = np.square(recon_signals - origin_signals)

        return {'origin': origin_signals,
                'recon': recon_signals,
                'score': ano_score}

    def predict(self, X):
        pass

    def forecast(self, X):
        pass

    def save(self, path):
        torch.save(self._vae.state_dict(), path)

    def load(self, path):
        self._vae.load_state_dict(torch.load(path))


def to_var(x):
    return Variable(x)


class VAE_Dense_Module(nn.Module):
    def __init__(self, window_size, h_dim, z_dim):
        super(VAE_Dense_Module, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.z_mean_net = nn.Sequential(
            nn.Linear(h_dim, z_dim),
        )

        self.z_logvar_net = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.x_mean_net = nn.Sequential(
            nn.Linear(h_dim, window_size),
        )

        self.x_logvar_net = nn.Sequential(
            nn.Linear(h_dim, window_size),
            nn.Softplus(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = to_var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z

    def forward(self, x):
        '''
        x:[batch, window_size * channels]
        '''
        h_x = self.encoder(x)
        z_mean = self.z_mean_net(h_x)
        z_logvar = self.z_logvar_net(h_x)
        z = self.reparameterize(z_mean, z_logvar)

        h_z = self.decoder(z)
        x_mean = self.x_mean_net(h_z)
        x_logvar = self.x_logvar_net(h_z)
        x_hat = self.reparameterize(x_mean, x_logvar)

        return x_hat, z_mean, z_logvar, x_mean, x_logvar