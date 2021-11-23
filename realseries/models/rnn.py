# -*- encoding: utf-8 -*-
"""RNN encoder decoder model.
Reference 'LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection'
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from ..utils.preprocess import augmentation
from ..utils.utility import EarlyStopping, load_model, save_model
from .base import BaseModel

__all__=['LSTMED']
class LSTMED(BaseModel):
    """RNN(LSTM) encoder decoder model for anomaly detection.

    Args:
        rnn_type (str, optional): Type of recurrent net (RNN_TANH, RNN_RELU,
            LSTM, GRU, SRU). Defaults to 'LSTM'.
        emsize (int, optional): Size of rnn input features. Defaults to 128.
        nhid (int, optional): Number of hidden units per layer. Defaults to 128.
        epochs (int, optional): Upper epoch limit. Defaults to 200.
        nlayers (int, optional): Number of LSTM layers. Defaults to 2.
        batch_size (int, optional): Batch size. Defaults to 64.
        window_size (int, optional): LSTM input sequence length. Defaults to 50.
        dropout (float, optional): Defaults to 0.2.
        lr (float, optional): Learning rate. Defaults to 0.0002.
        weight_decay (float, optional): Weight decay. Defaults to 1e-4.
        clip (int, optional): Gradient clipping. Defaults to 10.
        res_connection (bool, optional): Residual connection. This parameters
            has not been tested when setting `True`. Defaults to False.
        prediction_window_size (int, optional): Prediction window size.
            Defaults to 10.
        model_path (str, optional): The path to save or load model.
            Defaults to None.
        seed (int, optional): Seed. Defaults to 1111.

    Attributes:
        model: LSTM model.
    """
    def __init__(
            self,
            rnn_type='LSTM',
            emsize=128,
            nhid=128,
            epochs=200,
            nlayers=2,
            batch_size=64,
            window_size=50,
            dropout=0.2,
            lr=0.0002,
            weight_decay=1e-4,
            clip=10,
            res_connection=False,
            prediction_window_size=10,
            model_path=None,
            seed=1111,):
        super(LSTMED, self).__init__()
        self.rnn_type = rnn_type
        self.emsize = emsize
        self.nhid = nhid
        self.epochs = epochs
        self.nlayers = nlayers
        self.res_connection = res_connection
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = clip
        self.batch_size = batch_size
        self.window_size = window_size  # 'sequence length'
        self.dropout = dropout
        self.prediction_window_size = prediction_window_size
        self.model_path = model_path
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def detect(self, X, channel_idx=0):
        """If X is an array of shape (n_samples, n_features), it need to be
            detected one by one channel.

        Args:
            X (array_like): Input sequence.
            channel_idx (int, optional): The index of feature cahnnel to detect.
            Defaults to 0.

        Returns:
            ndarray: Anomaly score
        """
        X = np.asarray(X).reshape(-1, 1, X.shape[-1])
        X = (X - X.mean()) / X.std()
        X = torch.from_numpy(X).float()

        feature_dim = X.shape[-1]
        self.model = RNNPredictor(
            rnn_type=self.rnn_type,
            enc_inp_size=feature_dim,
            rnn_inp_size=self.emsize,
            rnn_hid_size=self.nhid,
            dec_out_size=feature_dim,
            nlayers=self.nlayers,
            dropout=self.dropout,
            res_connection=self.res_connection).to(self.device)
        self.model = load_model(self.model,
                                Path(self.model_path, 'checkpoint.pt'))

        temp = torch.load(Path(self.model_path, 'params.pt'))
        self.means = temp['means']
        self.covs = temp['covs']
        mean = self.means[channel_idx]
        cov = self.covs[channel_idx]
        predictions = []
        rearranged = []
        errors = []
        hiddens = []
        predicted_scores = []
        with torch.no_grad():
            # Turn on evaluation mode which disables dropout.
            self.model.eval()
            pasthidden = self.model.init_hidden(1)
            for t in range(len(X)):
                out, hidden = self.model(X[t].unsqueeze(0).to(self.device),
                                         pasthidden)
                predictions.append([])
                rearranged.append([])
                errors.append([])
                hiddens.append(self.model.extract_hidden(hidden))

                predictions[t].append(out.data.cpu()[0][0][channel_idx])
                pasthidden = self.model.repackage_hidden(hidden)
                for prediction_step in range(1, self.prediction_window_size):
                    out, hidden = self.model(out, hidden)
                    predictions[t].append(out.data.cpu()[0][0][channel_idx])

                if t >= self.prediction_window_size:
                    for step in range(self.prediction_window_size):
                        rearranged[t].append(
                            predictions[step + t - self.prediction_window_size][
                                self.prediction_window_size - 1 - step])
                    rearranged[t] = torch.FloatTensor(rearranged[t]).to(
                        self.device).unsqueeze(0)
                    errors[t] = rearranged[t] - X[t][0][channel_idx]
                else:
                    rearranged[t] = torch.zeros(
                        1, self.prediction_window_size).to(self.device)
                    errors[t] = torch.zeros(1, self.prediction_window_size).to(
                        self.device)
        predicted_scores = np.array(predicted_scores)
        scores = []
        for error in errors:
            mult1 = error.cpu() - mean.unsqueeze(
                0)  # [ 1 * prediction_window_size ]
            mult2 = torch.inverse(
                cov)  # [ prediction_window_size * prediction_window_size ]
            mult3 = mult1.t()  # [ prediction_window_size * 1 ]
            score = torch.mm(mult1, torch.mm(mult2, mult3))
            scores.append(score[0][0])

        scores = torch.stack(scores)
        self.scores = scores.data.cpu().numpy()
        rearranged = torch.cat(rearranged, dim=0)
        errors = torch.cat(errors, dim=0)
        return self.scores

    def fit(self,
            X,
            y=None,
            augment_length=None,
            split=0.25,
            monitor='val_loss',
            patience=10,
            delta=0,
            verbose=True):
        """Train the detector.

        Args:
            X (array_like): The input sequence of shape (n_length,).
            y (array_like, optional): Ignored. Defaults to None.
            augment_length (int, optional): The total number of samples after
                augmented. Defaults to None.
            split (float, optional): Fration to split for validation set.
                Defaults to 0.25.
            monitor (str, optional): Monitor the validation loss by setting the
                monitor argument to 'val_loss'. Defaults to 'val_loss'.
            patience (int, optional): Patience argument represents the number of
                epochs before stopping once your loss starts to increase
                (stops improving). Defaults to 10.
            delta (int, optional): A threshold to whether quantify a loss at
                some epoch as improvement or not. If the difference of loss is
                below delta, it is quantified as no improvement. Better to leave
                it as 0 since we're interested in when loss becomes worse.
                Defaults to 0.
            verbose (bool, optional): Verbose decides what to print.
                Defaults to True.
        """
        os.makedirs(self.model_path, exist_ok=True)
        early_stopping = EarlyStopping(monitor, patience, delta, verbose)
        X, y = np.asarray(X), np.asarray(y)
        length = len(X)
        if augment_length is not None:
            assert y is not None
            X, y = augmentation(X, y, max_length=50000)
        X = (X - X.mean()) / X.std()
        dataset = gen_dataset(X, self.window_size)
        val_length = int(split * len(dataset))
        train_length = len(dataset) - val_length
        train_dataset, val_dataset = random_split(dataset,
                                                  (train_length, val_length))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=True)
        valid_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=True)

        feature_dim = X.shape[-1]
        self.model = RNNPredictor(
            rnn_type=self.rnn_type,
            enc_inp_size=feature_dim,
            rnn_inp_size=self.emsize,
            rnn_hid_size=self.nhid,
            dec_out_size=feature_dim,
            nlayers=self.nlayers,
            dropout=self.dropout,
            res_connection=self.res_connection).to(self.device)
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_losses = []
        valid_losses = []

        for epoch in range(1, self.epochs + 1):
            train_losses.append(self._train(epoch, train_loader, optimizer))
            valid_losses.append(self._evaluate(epoch, valid_loader))
            train_loss = train_losses[-1]
            valid_loss = valid_losses[-1]
            print(
                f'epoch:{epoch}/{self.epochs}, train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}'
            )
            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if epoch % 20 == 0:
                save_model(self.model,
                           Path(self.model_path, f'checkpoint{epoch}.pt'))
        self.model = early_stopping.best_model
        save_model(self.model,
                   Path(self.model_path, 'checkpoint.pt'))
        self.means, self.covs = self._fit_norm_distribution_param(X[:length])
        torch.save({
            'means': self.means,
            'covs': self.covs
        }, Path(self.model_path, 'params.pt'))

    def _train(self, epoch, train_loader, optimizer, log_interval=500):
        total_loss = 0
        self.model.train()
        with torch.enable_grad():
            # Turn on training mode which enables dropout.
            hidden = self.model.init_hidden(self.batch_size)
            for batch, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.transpose(1, 0).to(
                    self.device), target.transpose(1, 0).to(self.device)
                hidden = self.model.repackage_hidden(hidden)
                hidden_ = self.model.repackage_hidden(hidden)
                optimizer.zero_grad()
                '''Loss1: Free running loss'''
                outVal = data[0].unsqueeze(0)
                outVals = []
                hids1 = []
                for i in range(data.shape[0]):
                    outVal, hidden_, hid = self.model(
                        outVal, hidden_, return_hiddens=True)
                    outVals.append(outVal)
                    hids1.append(hid)
                outSeq1 = torch.cat(outVals, dim=0)
                hids1 = torch.cat(hids1, dim=0)
                loss1 = nn.MSELoss()(outSeq1.view(self.batch_size, -1),
                                     target.view(self.batch_size, -1))
                '''Loss2: Teacher forcing loss'''
                outSeq2, hidden, hids2 = self.model(
                    data, hidden, return_hiddens=True)
                loss2 = nn.MSELoss()(outSeq2.view(self.batch_size, -1),
                                     target.view(self.batch_size, -1))
                '''Loss3: Simplified Professor forcing loss'''
                loss3 = nn.MSELoss()(hids1.view(self.batch_size, -1),
                                     hids2.view(self.batch_size, -1).detach())
                '''Total loss = Loss1+Loss2+Loss3'''
                loss = loss1 + loss2 + loss3
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.clip)
                optimizer.step()
                total_loss += loss.item()
                if batch % log_interval == 0:
                    cur_loss = total_loss / (batch + 1)
                    print('|epoch:{:3d}| batches:{:3d}| loss:{:5.2f}'.format(
                        epoch, batch, cur_loss))
        return cur_loss

    def _evaluate(self, epoch, valid_loader):
        # Turn on evaluation mode which disables dropout.
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            hidden = self.model.init_hidden(self.batch_size)
            for batch, (data, target) in enumerate(valid_loader):
                data, target = data.transpose(1, 0).to(
                    self.device), target.transpose(1, 0).to(self.device)
                # inputSeq: [ seq_len * batch_size * feature_size ]
                hidden_ = self.model.repackage_hidden(hidden)
                '''Loss1: Free running loss'''
                outVal = data[0].unsqueeze(0)
                outVals = []
                hids1 = []
                for i in range(data.size(0)):
                    outVal, hidden_, hid = self.model(
                        outVal, hidden_, return_hiddens=True)
                    outVals.append(outVal)
                    hids1.append(hid)
                outSeq1 = torch.cat(outVals, dim=0)
                hids1 = torch.cat(hids1, dim=0)
                loss1 = nn.MSELoss()(outSeq1.view(self.batch_size, -1),
                                     target.view(self.batch_size, -1))
                '''Loss2: Teacher forcing loss'''
                outSeq2, hidden, hids2 = self.model(
                    data, hidden, return_hiddens=True)
                loss2 = nn.MSELoss()(outSeq2.view(self.batch_size, -1),
                                     target.view(self.batch_size, -1))
                '''Loss3: Simplified Professor forcing loss'''
                loss3 = nn.MSELoss()(hids1.view(self.batch_size, -1),
                                     hids2.view(self.batch_size, -1).detach())
                '''Total loss = Loss1+Loss2+Loss3'''
                loss = loss1 + loss2 + loss3
                total_loss += loss.item()
        return total_loss / batch

    def _fit_norm_distribution_param(self, X):
        X = torch.from_numpy(X).float().view(-1, 1, X.shape[-1])
        means, covs = [], []
        for channel_idx in range(X.shape[-1]):
            predictions = []
            organized = []
            errors = []
            with torch.no_grad():
                # Turn on evaluation mode which disables dropout.
                self.model.eval()
                pasthidden = self.model.init_hidden(1)
                for t in range(len(X)):
                    out, hidden = self.model(X[t].unsqueeze(0).to(self.device),
                                             pasthidden)
                    predictions.append([])
                    organized.append([])
                    errors.append([])
                    predictions[t].append(out.data.cpu()[0][0][channel_idx])
                    pasthidden = self.model.repackage_hidden(hidden)
                    for prediction_step in range(1,
                                                 self.prediction_window_size):
                        out, hidden = self.model(out, hidden)
                        predictions[t].append(out.data.cpu()[0][0][channel_idx])

                    if t >= self.prediction_window_size:
                        for step in range(self.prediction_window_size):
                            organized[t].append(
                                predictions[step + t -
                                            self.prediction_window_size]
                                [self.prediction_window_size - 1 - step])
                        organized[t] = torch.FloatTensor(organized[t])
                        errors[t] = organized[t] - X[t][0][channel_idx]
                        errors[t] = errors[t].unsqueeze(0)

            errors_tensor = torch.cat(
                errors[self.prediction_window_size:], dim=0)
            mean = errors_tensor.mean(dim=0)
            cov = errors_tensor.t().mm(errors_tensor) / errors_tensor.size(
                0) - mean.unsqueeze(1).mm(mean.unsqueeze(0))
        means.append(mean)
        covs.append(cov)
        return means, covs


def gen_dataset(X, window_size):
    """Generate Dataset for training pytorch model.

    Args:
        X (ndarray): The total input sequence.
        window_size (int): Window size.

    Returns:
        object: dataset
    """
    a = []
    b = []
    for i in range(X.shape[0] - window_size):
        a.append(X[i:i + window_size])
        b.append(X[i + 1:i + 1 + window_size])
    assert len(a) == len(
        b) and a[-1].shape[0] == window_size and b[-1].shape[0] == window_size
    X, y = np.asarray(a), np.asarray(b)
    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    dataset = TensorDataset(X, y)
    return dataset


class RNNPredictor(nn.Module):
    def __init__(self,
                 rnn_type,
                 enc_inp_size,
                 rnn_inp_size,
                 rnn_hid_size,
                 dec_out_size,
                 nlayers,
                 dropout=0.5,
                 res_connection=False):
        super(RNNPredictor, self).__init__()
        self.enc_input_size = enc_inp_size

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(enc_inp_size, rnn_inp_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                rnn_inp_size, rnn_hid_size, nlayers, dropout=dropout)
        elif rnn_type == 'SRU':
            from cuda_functional import SRU, SRUCell
            self.rnn = SRU(
                input_size=rnn_inp_size,
                hidden_size=rnn_hid_size,
                num_layers=nlayers,
                dropout=dropout,
                use_tanh=False,
                use_selu=True,
                layer_norm=True)
        else:
            try:
                nonlinearity = {
                    'RNN_TANH': 'tanh',
                    'RNN_RELU': 'relu'
                }[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'SRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                rnn_inp_size,
                rnn_hid_size,
                nlayers,
                nonlinearity=nonlinearity,
                dropout=dropout)
        self.decoder = nn.Linear(rnn_hid_size, dec_out_size)

        self.res_connection = res_connection
        self.init_weights()
        self.rnn_type = rnn_type
        self.rnn_hid_size = rnn_hid_size
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_hiddens=False, noise=False):
        emb = self.drop(
            self.encoder(input.contiguous().view(-1, self.enc_input_size))
        )  # [(seq_len x batch_size) * feature_size]
        emb = emb.view(
            -1, input.size(1),
            self.rnn_hid_size)  # [ seq_len * batch_size * feature_size]
        if noise:
            hidden = (F.dropout(hidden[0], training=True, p=0.9),
                      F.dropout(hidden[1], training=True, p=0.9))

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))
        # [(seq_len x batch_size) * feature_size]
        decoded = decoded.view(
            output.size(0), output.size(1),
            decoded.size(1))  # [ seq_len * batch_size * feature_size]
        if self.res_connection:
            decoded = decoded + input
        if return_hiddens:
            return decoded, hidden, output

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_(),
                    weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_())
        else:
            return weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_()

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()

    def extract_hidden(self, hidden):
        if self.rnn_type == 'LSTM':
            return hidden[0][-1].data.cpu(
            )  # hidden state last layer (hidden[1] is cell state)
        else:
            return hidden[-1].data.cpu()  # last layer
