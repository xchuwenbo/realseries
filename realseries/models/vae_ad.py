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

__all__ = ['VAE_AD']
class VAE_AD(BaseModel):
    """The Donut-VAE version for anomaly detection

    Args:
        name (str, optional): Model name. Defaults to 'VAE_AD'.
        num_epochs (int, optional): Epochs for model training. Defaults to 256.
        batch_size (int, optional): Batch size for model training. Defaults to 256.
        lr ([type], optional): Learning rate. Defaults to 1e-3.
        lr_decay (float, optional): Learning rate decay. Defaults to 0.8.
        clip_norm_value (float, optional): Gradient clip value. Defaults to 12.0.
        weight_decay ([type], optional): L2 regularization. Defaults to 1e-3.
        data_split_rate (float, optional): Defaults to 0.5.
        window_size (int, optional): Defaults to 120.
        window_step (int, optional): Defaults to 1.
        h_dim (int, optional): Hidden dim between x and z for VAE's encoder and decoder Defaults to 100.
        z_dim (int, optional): Defaults to 5.

    Attributes:
        model: VAE model built by torch.
    """
    def __init__(self,
                 name='VAE_AD',
                 num_epochs=256,
                 batch_size=256,
                 lr=1e-3,
                 lr_decay=0.8,
                 clip_norm_value=12.0,
                 weight_decay=1e-3,
                 data_split_rate=0.5,
                 window_size=120,
                 window_step=1,
                 h_dim=100,
                 z_dim=5):
        super(VAE_AD, self).__init__()
        self._name = name
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._lr = lr
        self._lr_decay = lr_decay
        self._clip_norm_value = clip_norm_value
        self._weight_decay = weight_decay
        self._data_split_rate = 0.5
        self._window_size = window_size
        self._window_step = window_step
        self._h_dim = h_dim
        self._z_dim = z_dim
        self._vae = VAE_Module(self._window_size,
                              self._h_dim,
                              self._z_dim)
        self._trained = False

    def fit(self, X):
        """Train the model

        Args:
            X (array_like): The input 1-D array.
        """
        def loss_vae(x_hat, x, mu, logvar):
            # print(x_hat.shape)
            mse_x = F.mse_loss(x_hat, x)
            mse_x = mse_x * x.shape[1]
            # KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
            KLD = 1 + logvar - mu ** 2 - logvar.exp()
            # print(KLD.shape)
            KLD = torch.sum(KLD, dim=-1)
            # print(KLD.shape)
            KLD *= -0.5
            # print(KLD.shape)
            return torch.mean(mse_x + KLD)

        train_gen = BatchSegment(len(X),
                                 self._window_size,
                                 self._batch_size,
                                 shuffle=True,
                                 discard_last_batch=True)
        # self._vae = VAE_Module(self._window_size,
        #                       self._h_dim,
        #                       self._z_dim)
        self._vae.train()
        optimizer = torch.optim.Adam(self._vae.parameters(),
                                     lr=self._lr,
                                     weight_decay=self._weight_decay)
        scheduler = StepLR(optimizer, step_size=10,
                           gamma=self._lr_decay)

        for epoch in range(self._num_epochs):
            for (batch_x, ) in train_gen.get_iterator([np.asarray(X, dtype=np.float32)]):
                optimizer.zero_grad()

                batch_x = torch.from_numpy(batch_x)
                x_hat, mu, logvar = self._vae(batch_x)
                loss = loss_vae(x_hat, batch_x, mu, logvar)

                loss.backward()
                torch.nn.utils.clip_grad_norm(self._vae.parameters(),
                                              self._clip_norm_value)
                optimizer.step()

                if epoch % 20 == 0:
                        print("Epoch[{}/{}] Loss: {:.6f}".format(epoch + 1, self._num_epochs, loss.item() / self._batch_size))
            scheduler.step()

        self._trained = True

    def detect(self, X):
        """ Get anomaly score of input sequence.

        Args:
            X (array_like): Input sequence.

        Returns:
            A dict with attributes:
                origin_series: ndarray, Origin time series
                score: ndarray, Corresponding anomaly score.
        """
        def series_to_windows(series, window_size, window_step):
            windows = np.array([series[i: i + window_size] for i in range(0, len(series) - window_size + 1, window_step)], dtype=np.float32)

            return windows

        def get_expectation(model, mean, logvar, sample_count):
            z_list = list()
            for _ in range(sample_count):
                z = model.reparameterize(mean, logvar)
                z_list.append(z.detach().numpy())
            z_list = torch.from_numpy(np.array(z_list, dtype=np.float32))
            z = torch.mean(z_list, dim=0)
            return z

        def get_anomaly_score(model, test_data, n_z, n_x):
            x = series_to_windows(test_data, self._window_size, self._window_step)
            x = torch.from_numpy(x)

            h_x = model.encoder(x)
            z_mean = model.z_mean_net(h_x)
            z_logvar = model.z_logvar_net(h_x)

            z = get_expectation(model, z_mean, z_logvar, n_z)

            h_z = model.decoder(z)
            x_mean = model.x_mean_net(h_z)
            x_logvar = model.x_logvar_net(h_z)
            x_hat = get_expectation(model, x_mean, x_logvar, n_x)
            # x_hat, _, _ = model(x)
            ori_series = x.detach().numpy()[:, -1]
            recon_series = x_hat.detach().numpy()[:, -1]
            ano_score = recon_series - ori_series
            ano_score = np.array([x*x for x in ano_score])
            return ori_series, ano_score

        self._vae.eval()
        ori_series, anomaly_score = get_anomaly_score(self._vae,
                                                      X,
                                                      n_z=1024,
                                                      n_x=1024)
        # res = np.where(anomaly_score < thres, 0, 1)

        return {'origin_series': ori_series,
                'score': anomaly_score}

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


class VAE_Module(nn.Module):
    def __init__(self, window_size, h_dim, z_dim):
        super(VAE_Module, self).__init__()
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
        x:[batch, window_size]
        '''
        h_x = self.encoder(x)
        z_mean = self.z_mean_net(h_x)
        z_logvar = self.z_logvar_net(h_x)
        z = self.reparameterize(z_mean, z_logvar)

        h_z = self.decoder(z)
        x_mean = self.x_mean_net(h_z)
        x_logvar = self.x_logvar_net(h_z)
        x_hat = self.reparameterize(x_mean, x_logvar)

        return x_hat, z_mean, z_logvar


if __name__ == '__main__':

    from ..utils.data import load_Yahoo
    import matplotlib.pyplot as plt

    train_data, test_data = load_Yahoo('A2Benchmark',
                                       'synthetic_3.csv',
                                       use_norm=True)

    vae = VAE_AD()
    vae.fit(train_data.value.values)
    res = vae.detect(test_data.value.values)
    ori_series = res['ori_series']
    anomaly_score = res['score']
    plt.plot(ori_series / np.max(ori_series) + 1, label='origin')
    plt.plot(anomaly_score / np.max(anomaly_score), label='score')
    plt.legend()
    plt.show()
    plt.close()


