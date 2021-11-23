# -*- encoding: utf-8 -*-
"""The implementation of luminol method.
Reference: https://github.com/linkedin/luminol
"""
import numpy as np
import pandas as pd
from luminol.anomaly_detector import AnomalyDetector

from .base import BaseModel

__all__=['Lumino']
class Lumino(BaseModel):

    def __init__(self):
        super(Lumino, self).__init__()

    def detect(self, X, algorithm_name=None, algorithm_params=None):
        """Detect the input sequence and return anomaly socre.

        Args:
            X (array_like): 1-D time series with shape (n_samples,)
            algorithm_name (str, optional): Algorithm_name. Defaults to None.
            algorithm_params (dict, optional): Algorithm_params. Defaults to None.
                The algorithm_name and the corresponding algorithm_params are:

                .. code-block:: python
                    :linenos:

                    1.  'bitmap_detector': # behaves well for huge data sets, and it is the default detector.
                        {
                        'precision'(4): # how many sections to categorize values,
                        'lag_window_size'(2% of the series length): # lagging window size,
                        'future_window_size'(2% of the series length): # future window size,
                        'chunk_size'(2): # chunk size.
                        }
                    2.  'default_detector': # used when other algorithms fails, not meant to be explicitly used.
                    3.  'derivative_detector': # meant to be used when abrupt changes of value are of main interest.
                        {
                        'smoothing factor'(0.2): # smoothing factor used to compute exponential moving averages
                                                    # of derivatives.
                        }
                    4.  'exp_avg_detector': # meant to be used when values are in a roughly stationary range.
                                            # and it is the default refine algorithm.
                        {
                        'smoothing factor'(0.2): # smoothing factor used to compute exponential moving averages.
                        'lag_window_size'(20% of the series length): # lagging window size.
                        'use_lag_window'(False): # if asserted, a lagging window of size lag_window_size will be used.
                        }

        Returns:
            ndarray: Normalized anomaly score in [0,1].

        """
        X = np.asarray(X)
        self.model = AnomalyDetector(
            time_series=dict(pd.Series(X)),
            algorithm_name=algorithm_name,
            algorithm_params=algorithm_params)
        score = self.model.get_all_scores()
        score = [value for _, value in score.iteritems()]
        score = np.array(score)

        self.anomaly_score = ((score - score.min()) /
                              (score.max() - score.min()))
        return self.anomaly_score

    def fit(self):
        pass

