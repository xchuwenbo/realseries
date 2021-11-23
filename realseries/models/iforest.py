# -*- encoding: utf-8 -*-
'''The implementation of isolation forest method based on sklearn.
'''

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import BaseModel

__all__=['IForest']

class IForest(BaseModel):
    """Isolation forest algorithm.

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Args:
        n_estimators (int, optional): The number of base estimators in the
            ensemble. Defaults to 100.
        max_samples (int or float, optional): The number of samples to draw from
            X to train each base estimator. Defaults to "auto".

                - If int, then draw `max_samples` samples.
                - If float, then draw `max_samples * X.shape[0]` samples.
                - If "auto", then `max_samples=min(256, n_samples)`.

            If max_samples is larger than the number of samples provided, all
            samples will be used for all trees (no sampling).
        contamination ('auto' or float, optional): The amount of contamination
            of the data set. Defaults to 'auto'.
        max_features (int or float, optional): The number of features to draw
            from X to train each base estimator. Defaults to 1.
        bootstrap (bool, optional):  If True, individual trees are fit on random
            subsets of the training data sampled with replacement. If False,
            sampling without replacement is performed. Defaults to False.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults
            to 1.
        random_state (int, optional): If RandomState instance, random_state is
            the random number generator. If None, the random number generator
            is the RandomState instance used by `np.random`. Defaults to 0.
        verbose (int, optional): Controls the verbosity of the tree building
            process. Defaults to None.

    Attributes:
        anomaly_score: Array of anomaly score.
        IF: The isolation model.
        estimators_: List of DecisionTreeClassifier.The collection of fitted sub-estimators.
        estimators_samples_ : List of arrays.The subset of drawn samples (i.e.,
            the in-bag samples) for each base estimator.
        max_samples_ : The actual number of samples

    """

    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination='auto',
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(IForest, self).__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """Train the model.

        Args:
            X (array_like): The input sequence with shape (n_sample, n_features).
            y (ndarray, optional): The label. Defaults to None.
        """
        X = np.asarray(X)
        self.IF = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose)
        self.IF.fit(X)

    def detect(self, X):
        """Detect the test data by trained model.

        Args:
            X (array_like): The input sequence with shape (n_sample, n_features).

        Returns:
            ndarray: The predicted anomaly score.
        """
        X = np.asarray(X)
        anomaly_score = -self.IF.score_samples(X)
        self.anomaly_score = ((anomaly_score - anomaly_score.min()) /
                              (anomaly_score.max() - anomaly_score.min()))
        return self.anomaly_score

    @property
    def estimators_(self):
        """The collection of fitted sub-estimators.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.IF.estimators_

    @property
    def estimators_samples_(self):
        """The subset of drawn samples (i.e., the in-bag samples) for
        each base estimator.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.IF.estimators_samples_

    @property
    def max_samples_(self):
        """The actual number of samples.
        Decorator for scikit-learn Isolation Forest attributes.
        """
        return self.IF.max_samples_
