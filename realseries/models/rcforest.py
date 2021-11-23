# -*- encoding: utf-8 -*-
"""The implementation of random cur forest method. Reference: S. Guha, N. Mishra,
G. Roy, & O. Schrijvers, Robust random cut forest based anomaly detection on
streams, in Proceedings of the 33rd International conference on machine learning,
New York, NY, 2016 (pp. 2712-2721). https://github.com/kLabUM/rrcf
"""

import numpy as np
import pandas as pd
import rrcf

from .base import BaseModel

__all__=['RCForest']

class RCForest(BaseModel):
    """Random cut forest.The Robust Random Cut Forest (RRCF) algorithm is an
    ensemble method for detecting outliers in streaming data. RRCF offers a
    number of features that many competing anomaly detection algorithms lack.
    Specifically, RRCF:

        - Is designed to handle streaming data.
        - Performs well on high-dimensional data.
        - Reduces the influence of irrelevant dimensions.
        - Gracefully handles duplicates and near-duplicates that could otherwise
          mask the presence of outliers.
        - Features an anomaly-scoring algorithm with a clear underlying
          statistical meaning.

    Args:
        shingle_size (int, optional): Window size. Defaults to 32.
        num_trees (int, optional): Number of estimators. Defaults to 100.
        tree_size (int, optional): Number of leaf. Defaults to 50.
        random_state (int, optional): Random state seed. Defaults to None.
    """

    def __init__(self,
                 shingle_size=32,
                 num_trees=100,
                 tree_size=50,
                 random_state=0):
        super(RCForest, self).__init__()
        self.shingle_size = shingle_size
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y=None):
        pass
        # points = rrcf.shingle(X, self.shingle_size)
        # points = np.vstack([_ for _ in points])
        # n = points.shape[0]
        # sample_size_range = (n // self.tree_size, self.tree_size)
        # forest = []
        # while len(forest) < self.num_trees:
        #     ixs = np.random.choice(n, size=sample_size_range, replace=False)
        #     trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in ixs]
        #     forest.extend(trees)
        # self.model = forest

    # def predict(self, X, y=None):
    #     points = rrcf.shingle(X, self.shingle_size)
    #     forest = self.model

    #     anomaly_score = {}
    #     points = np.vstack([_ for _ in points])
    #     for index, point in enumerate(points):
    #         for tree in forest:
    #             tree.insert_point(point, index=str(index) + 'b')
    #             new_codisp = tree.codisp(str(index) + 'b')
    #             tree.forget_point(str(index) + 'b')
    #             if not index in anomaly_score:
    #                 anomaly_score[index] = 0
    #             anomaly_score[index] += new_codisp / self.num_trees

    #     anomaly_score = pd.Series(anomaly_score).values
    #     anomaly_score = ((anomaly_score - anomaly_score.min()) /
    #                      (anomaly_score.max() - anomaly_score.min()))
    #     self.anomaly_score = anomaly_score
    #     return self.anomaly_score

    def detect(self, X):
        """Detect the input.

        Args:
            X (array_like): Input sequence.

        Returns:
            ndarray: Anomaly score.
        """
        X = np.asarray(X)
        points = rrcf.shingle(X, self.shingle_size)
        points = np.vstack([_ for _ in points])
        n = points.shape[0]
        sample_size_range = (n // self.tree_size, self.tree_size)
        forest = []
        while len(forest) < self.num_trees:
            ixs = self.rng.choice(n, size=sample_size_range, replace=False)
            trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in ixs]
            forest.extend(trees)
        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)
        for tree in forest:
            codisp = pd.Series(
                {leaf: tree.codisp(leaf) for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)

        avg_codisp /= index
        avg_codisp = ((avg_codisp - avg_codisp.min()) /
                      (avg_codisp.max() - avg_codisp.min()))
        anomaly_score = avg_codisp.values
        self.anomaly_score = np.r_[[0] * (self.shingle_size - 1), anomaly_score]
        return self.anomaly_score
