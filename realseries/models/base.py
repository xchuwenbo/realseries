# -*- coding: utf-8 -*-
"""Base class for all time series analysis methods. It includes the methods like
fit, detect and predict etc.
"""
# Author: Wenbo Hu <i@wbhu.net>

import abc
import six

__all__ = ['BaseModel']


class BaseModel(object, metaclass=abc.ABCMeta):
    """BaseModel class for all RealSeries predict/detect algorithms.

    Args:
        contamination (float, optional): The amount of contamination of the data set,
            i.e. the proportion of outliers in the data set. Used when fitting to
            define the threshold on the decision function.. Defaults to 0.1.

    Raises:
        ValueError: Contamination must be in (0, 0.5].

    """

    @abc.abstractmethod
    def __init__(self, contamination=0.1):
        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.contamination = contamination

    @abc.abstractmethod
    def fit(self, x, y=None):
        """Fit the model

        Args:
            x (array_like): The input sequence of shape (n_length, n_features) or (n_length,).
            y (ndarray, optional): Ignored. Defaults to None.
        """
        pass

    @abc.abstractmethod
    def detect(self, x):
        """Predict using the trained detector.

        Args:
            x (array_like): The input sequence of shape (n_length, n_features) or (n_length,).

        Returns:
            ndarray: Outlier labels of shape (n_length,).For each sample of time series, whether or not it is an outlier. 0 for inliers and 1 for outliers.
        """
        pass

    def forecast(self, x, t):
        """Forecast the input.

        Args:
            x (array_like): The input sequence of shape (n_length, n_features) or (n_length,).
            t (int): time index of to-be-forecast samples.

        Returns:
            X_1 (ndarray): Forecast samples of shape (n_length, n_features)
        """
        pass

    def impute(self, x, t):
        """Impute the input data X at time index t.

        Args:
            x (array_like): The input sequence of shape (n_length, n_features) or (n_length,).
            t (int): time index of to-be-forecast samples.

        Returns:
            X_1 (ndarray): Impute samples of shape (n_length, n_features)
        """
        pass

    def save(self, path):
        """Save the model to path

        Args:
            path (string): model save path 
        """

    def load(self, path):
        """Load the model from path

        Args:
            path (string): model load path
        """

