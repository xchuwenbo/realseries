# -*- encoding: utf-8 -*-
"""function like save load model, early stop in model training.
"""
# @Time    : 2020/01/14 15:44:42
# @Author  : ZHANG Xianrui

import numpy as np
import torch
import os
from torch import nn
from functools import partial
from torch.autograd import Variable


def save_model(model, model_path):
    """Save pytorch model.
    
    Args:
        model (pytorch model): The trained pytorch model.
        model_path (string or path): Path for saving model.
    """    
    print("saving %s" % model_path)
    try:
        torch.save(model.state_dict(), model_path)
    except:
        torch.save(model, model_path)


def load_model(model, path):
    """Load Pytorch model.
    
    Args:
        model (pytorch model): The initialized pytorch model.
        model_path (string or path): Path for loading model.
    
    Returns:
        model: The loaded model.
    """    
    print("loading %s" % path)
    with open(path, 'rb') as f:
        pretrained = torch.load(f, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
    return model


# def unzip(filename='data/yahoo.zip', dest_dir='data'):
#     import zipfile
#     z = zipfile.ZipFile(filename, "r")
#     try:
#         z.extractall(path=dest_dir)
#     except RuntimeError as e:
#         print(e)
#     z.printdir()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve
     after a given patience.

    Args:
        patience (int): How long to wait after last time validation loss improved.
            Default to 7
        verbose (bool): If True, prints a message for each validation loss
            improvement. Default to False
        delta (float): Minimum change in the monitored quantity to qualify as an
            improvement. Default to 0.

    """

    def __init__(self, monitor='val_loss', patience=7, delta=0, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor = monitor
        self.delta = delta
        self.value = np.Inf

    def __call__(self, value, model):
        if self.monitor == 'val_loss':
            score = -value
        elif self.monitor == 'val_acc':
            score = value
        else:
            print('Error in initial monitor')

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(value, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}'
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(value, model)
            self.counter = 0

    def save_checkpoint(self, value, model):
        """Saves checkpoint when validation loss decrease.

        Args:
            value (float): The value of new validation loss.
            model (model): The current better model.
        """

        if self.verbose:
            print(f'{self.monitor} updated:({self.value:.6f} --> {value:.6f}).')
        self.best_model = model
        self.value = value

class aleatoric_loss(nn.Module):
    """The negative log likelihood (NLL) loss. 

    Args:
        gt: the ground truth 
        pred_mean: the predictive mean
        logvar: the log variance
    Attributes:
        loss: the nll loss result for the regression.
    """
    def __init__(self):
        super(aleatoric_loss, self).__init__()
    def forward(self, gt, pred_mean, logvar):
        loss = torch.sum(0.5*(torch.exp((-1)*logvar)) *
                         (gt - pred_mean)**2 + 0.5*logvar)
        return loss

class mmd_loss(nn.Module):
    """The mmd loss. 

    Args:
        source_features: the ground truth 
        target_features: the prediction value
    Attributes:
        loss_value: the nll loss result for the regression.
    """
    def __init__(self):
        super(mmd_loss, self).__init__()
    
    def forward(self, source_features, target_features):
    
        sigmas = [
            1, 4, 8, 16, 24, 32, 64
        ]
        if source_features.is_cuda:
            gaussian_kernel = partial(
                self.gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
            )
        else:
            source_features = source_features.cpu()
            target_features = target_features.cpu()
            gaussian_kernel = partial(
                self.gaussian_kernel_matrix, sigmas = Variable(torch.FloatTensor(sigmas))
            )

        loss_value = self.maximum_mean_discrepancy(
            source_features, target_features, kernel=gaussian_kernel)
        loss_value = loss_value

        return loss_value
        
    def pairwise_distance(self, x, y):
        
        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.shape[1] != y.shape[1]:
            raise ValueError('The number of features should be the same.')

        x = x.view(x.shape[0], x.shape[1], 1)
        y = torch.transpose(y, 0, 1)
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)

        return output

    def gaussian_kernel_matrix(self, x, y, sigmas):

        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = self.pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_)

        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def maximum_mean_discrepancy(self, x, y, kernel= gaussian_kernel_matrix):

        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))

        return cost

