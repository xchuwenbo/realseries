# -*- encoding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.utils.data as data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from ..utils.evaluation import point_metrics, adjust_predicts
from ..utils.utility import load_model, save_model
from .base import BaseModel
from .sr import SpectralResidual, average_filter

__all__=['SR_CNN']
class SR_CNN(BaseModel):
    """The sali_map method for anomaly detection.

    Args:
        model_path (str, optional): Path for saving and loading model.
        window (int, optional): Length of each sample for input. Defaults to 128.
        lr (float, optional): Learning rate. Defaults to 1e-6.
        seed (int, optional): Random seed. Defaults to 0.
        epochs (int, optional): Defaults to 20.
        batch_size (int, optional): Defaults to 64.
        dropout (float, optional): Defaults to 0.2.
        num_worker (int, optional): Defaults to 0.

    Attributes:
        model: CNN model built by torch.
    """

    def __init__(self,
                 model_path,
                 window=128,
                 lr=1e-6,
                 seed=0,
                 epochs=20,
                 batch_size=64,
                 dropout=0.2,
                 num_worker=0):

        super(SR_CNN, self).__init__()
        self.model_path = model_path
        self.window = window
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_worker = num_worker
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _build_model(self):
        self.model = CNN(self.window, self.dropout).to(self.device)
        return self.model

    def fit(self, X, step=64, num=10, back_k=0):
        """Train the model

        Args:
            X (array_like): The input 1-D array.
            step (int, optional): Stride of sliding window. Defaults to 64.
            num (int, optional): Number of added anomaly points to each window.
                Defaults to 10.
            back_k (int, optional): Defaults to 0.
        """
        os.makedirs(self.model_path, exist_ok=True)
        X = np.asarray(X)
        train_data = gene_train_data(
            X,
            window=self.window,
            step=step,
            num=num,
            back_k=back_k,
            seed=self.seed)
        self._build_model()
        print(self.model)
        base_lr = self.lr
        # bp_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.SGD(
            self.model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0)
        gen_data = gen_set(self.window, train_data)
        train_loader = data.DataLoader(
            gen_data,
            shuffle=True,
            num_workers=self.num_worker,
            batch_size=self.batch_size,
            pin_memory=True)
        for epoch in range(1, self.epochs + 1):
            self._train(epoch, train_loader, optimizer)
            cur_lr = base_lr * (0.5**((epoch + 10) // 10))
            for param in optimizer.param_groups:
                param['lr'] = cur_lr
            if epoch % 10 == 0:
                save_model(self.model, Path(self.model_path, f'{epoch}.pt'))


    def _train(self, epoch, train_loader, optimizer):
        self.model.train()
        for batch_idx, (inputs, lb) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            valueseq = inputs.to(self.device)
            lb = lb.to(self.device)
            output = self.model(valueseq)
            loss1 = loss_function(output, lb, self.model, self.window)
            loss1.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss1.item() / len(inputs)))

    def detect(self, X, y, back_k=0, backaddnum=5, step=1):
        """ Get anomaly score of input sequence.

        Args:
            X (array_like): Input sequence.
            y : Ignored.
            back_k (int, optional): Not test. Defaults to 0.
            backaddnum (int, optional): Not test. Defaults to 5.
            step (int, optional): Stride of sliding window in detecing stage.
                Defaults to 1.

        Returns:
            ndarray: Anomaly score.
        """
        X, y = np.asarray(X), np.asarray(y)
        self._build_model()
        self.model = load_model(self.model,
                                Path(self.model_path,
                                     f'{self.epochs}.pt')).to(self.device)
        self.model.eval()
        length = len(X)
        if back_k <= 5:
            back = back_k
        else:
            back = 5
        scores = [0] * (self.window - backaddnum)
        ##
        for pt in range(self.window - backaddnum + back + step, length - back,
                        step):
            head = max(0, pt - (self.window - backaddnum))
            tail = min(length, pt)
            wave = np.array(SpectralResidual.extend_series(X[head:tail + back]))
            mag = spectral_residual(wave)
            with torch.no_grad():
                mag = torch.from_numpy(100 * mag).float()
                mag = torch.unsqueeze(mag, 0).to(self.device)
                output = self.model(mag)
            rawout = output.detach().cpu().numpy().reshape(-1)

            ##
            for ipt in range(pt - step - back, pt - back):
                scores.append(rawout[ipt - head].item())
        scores += [0] * (length - len(scores))
        scores = np.array(scores)
        print(f'scores.shape:{scores.shape},X.shape:{X.shape}', scores[:10])
        print('***********************************************')
        return scores


class CNN(nn.Module):

    def __init__(self, window=1024, dropout=0.2):
        self.window = window
        super(CNN, self).__init__()
        self.layer1 = nn.Conv1d(
            window, window, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layer2 = nn.Conv1d(
            window, 2 * window, kernel_size=1, stride=1, padding=0)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(2 * window, 4 * window)
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(4 * window, window)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), self.window, 1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def loss_function(x, lb, net, win_size):
    l2_reg = 0.
    l2_weight = 0.
    for W in net.parameters():
        l2_reg = l2_reg + W.norm(2)
    weight = torch.ones(lb.shape)
    weight[lb == 1] = win_size // 100
    weight = weight.cuda()
    BCE = F.binary_cross_entropy(x, lb, weight=weight, reduction='sum')
    return l2_reg * l2_weight + BCE


class gen_set(Dataset):

    def __init__(self, width, data_array):
        self.genlen = 0
        self.len = self.genlen
        self.width = width
        self.kpinegraw = data_array
        self.negrawlen = len(self.kpinegraw)
        print('length :', len(self.kpinegraw))
        self.len += self.negrawlen
        self.kpineglen = 0
        self.control = 0.

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        idx = index % self.negrawlen
        datas = self.kpinegraw[idx]
        datas = np.array(datas)
        data = datas[0, :]
        lbs = datas[1, :]
        wave = spectral_residual(data)
        waveavg = average_filter(wave)
        for i in range(self.width):
            if wave[i] < 0.001 and waveavg[i] < 0.001:
                lbs[i] = 0
                continue
            ratio = wave[i] / waveavg[i]
            if ratio < 1.0 and lbs[i] == 1:
                lbs[i] = 0
            if ratio > 5.0:
                lbs[i] = 1
        srscore = abs(wave - waveavg) / (waveavg + 0.01)
        sortid = np.argsort(srscore)
        for idx in sortid[-2:]:
            if srscore[idx] > 5:
                lbs[idx] = 1
        resdata = torch.from_numpy(100 * wave).float()
        reslb = torch.from_numpy(lbs).float()
        return resdata, reslb


def gene_train_data(in_value, window=128, step=64, num=10, back_k=0, seed=0):
    np.random.seed(seed)
    print("generating train data")
    generator = gen(window, step, num)
    assert len(in_value) > window,\
        print("value's length < window size", len(in_value), window)
    train_data = generator.generate_train_data(in_value, back_k)
    print('fake data size:', train_data.shape)
    return train_data


class gen():

    def __init__(self, win_siz, step, nums):
        self.control = 0
        self.win_siz = win_siz
        self.step = step
        self.number = nums

    def generate_train_data(self, value, back_k=0):

        def normalize(a):
            amin = np.min(a)
            amax = np.max(a)
            a = (a - amin) / (amax - amin + 1e-5)
            return 3 * a

        if back_k <= 5:
            back = back_k
        else:
            back = 5
        length = len(value)
        tmp = []
        for pt in range(self.win_siz, length - back, self.step):
            head = max(0, pt - self.win_siz)
            tail = min(length - back, pt)
            data = np.array(value[head:tail])
            data = normalize(data)
            num = np.random.randint(1, self.number)
            ids = np.random.choice(self.win_siz, num, replace=False)
            lbs = np.zeros(self.win_siz, dtype=np.int64)
            if (self.win_siz - 6) not in ids:
                self.control += np.random.random()
            else:
                self.control = 0
            if self.control > 100:
                ids[0] = self.win_siz - 6
                self.control = 0
            mean = np.mean(data)
            dataavg = average_filter(data)
            var = np.var(data)
            for id in ids:
                data[id] += (dataavg[id] + mean) * np.random.randn() * min(
                    (1 + var), 10)
                lbs[id] = 1
            tmp.append([data, lbs])
        return np.array(tmp)


def spectral_residual(values, n=3, EPS=1e-8):
    """
    This method transform a time series into spectral residual series

    Args:
        values (list or numpy array): a 1-D array of float values.
        n (int, optional): window of filter. Defaults to 3.
        EPS (float, optional): value less than EPS has no meaning, using 0 to replace it. Defaults to 1e-8.

    Returns:
        numpy array: a list of float values as the spectral residual values
    """
    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real**2 + trans.imag**2)

    maglog = [np.log(item) if abs(item) > EPS else 0 for item in mag]

    spectral = np.exp(maglog - average_filter(maglog, n))

    trans.real = [
        ireal * ispectral / imag if abs(imag) > EPS else 0
        for ireal, ispectral, imag in zip(trans.real, spectral, mag)
    ]
    trans.imag = [
        iimag * ispectral / imag if abs(imag) > EPS else 0
        for iimag, ispectral, imag in zip(trans.imag, spectral, mag)
    ]
    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real**2 + wave_r.imag**2)
    return mag