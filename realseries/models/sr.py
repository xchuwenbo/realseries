# -*- encoding: utf-8 -*-

# @Time    : 2020/01/14 14:28:37
# @Author  : ZHANG Xianrui

import numpy as np
import pandas as pd

from .base import BaseModel

__all__ = ['SpectralResidual']
def average_filter(values, n=3):
    """Average filter.

    Args:
        values (array_like): A list of float numbers
        n (int, optional): Number of values for average. Defaults to 3.

    Returns:
        list: A list of value after the average_filter processing.
    """
    if n >= len(values):
        n = len(values)
    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n
    for i in range(1, n):
        res[i] /= i + 1
    return res


class SpectralResidual(BaseModel):
    """SpectralResidual calss.

    Args:
        series: input time series with shape (n_sample,)
        threshold: the threshold that apply anomaly score
        mag_window: the window of avarage filter when calculating spectral mag
        score_window: the window of average filter when calculating  score
    """

    def __init__(self, series, threshold, mag_window, score_window):
        self.series = series
        self.threshold = threshold
        self.mag_window = mag_window
        self.score_window = score_window
        self.EPS = 1e-8

    def detect(self):
        anomaly_score = self.generate_spectral_score(self.series)
        anomaly_score = (anomaly_score - anomaly_score.min()
                        ) / anomaly_score.max() - anomaly_score.min()

        anomaly_frame = pd.DataFrame({
            "value": self.series,
            "id": np.arange(len(anomaly_score)),
            "score": anomaly_score,
        })
        anomaly_frame["isAnomaly"] = np.where(
            anomaly_frame["score"] >= self.threshold, 1, 0)
        anomaly_frame.set_index("id", inplace=True)
        return anomaly_frame

    def generate_spectral_score(self, series):
        extended_series = SpectralResidual.extend_series(series)
        mag = self.spectral_residual_transform(extended_series)[:len(series)]
        ave_mag = average_filter(mag, n=self.score_window)
        ave_mag[np.where(ave_mag <= self.EPS)] = self.EPS

        return abs(mag - ave_mag) / ave_mag

    def spectral_residual_transform(self, values):
        """Transform a time series into spectral residual series by FFT.

        Args:
            values (ndarray): Array of values.

        Returns:
            ndarray: Spectral residual values.
        """
        trans = np.fft.fft(values)
        mag = np.sqrt(trans.real**2 + trans.imag**2)
        eps_index = np.where(mag <= self.EPS)[0]
        mag[eps_index] = self.EPS

        mag_log = np.log(mag)
        mag_log[eps_index] = 0

        spectral = np.exp(mag_log - average_filter(mag_log, n=self.mag_window))

        trans.real = trans.real * spectral / mag
        trans.imag = trans.imag * spectral / mag
        trans.real[eps_index] = 0
        trans.imag[eps_index] = 0

        wave_r = np.fft.ifft(trans)
        mag = np.sqrt(wave_r.real**2 + wave_r.imag**2)
        return mag

    @staticmethod
    def predict_next(values):
        """Predicts the next value by sum up the slope of the last value with
        previous values.

        Mathematically, :math:`g = 1/m * \sum_{i=1}^{m} g(x_n, x_{n-i})`,
        :math:`x_{n+1} = x_{n-m+1} + g * m`,
        where :math:`g(x_i,x_j) = (x_i - x_j) / (i - j)`.

        Args:
            values (list): a list of float numbers.

        Raises:
            ValueError: Length lsit should at least 2.

        Returns:
            float: The predicted next value.
        """
        if len(values) <= 1:
            raise ValueError(f"data should contain at least 2 numbers")

        v_last = values[-1]
        n = len(values)

        slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

        return values[1] + sum(slopes)

    @staticmethod
    def extend_series(values, extend_num=5, look_ahead=5):
        """extend the array data by the predicted next value

        Args:
            values (ndarray):  array of float numbers.
            extend_num (int, optional): number of values added to the back of
                data. Defaults to 5.
            look_ahead (int, optional): number of previous values used in
                prediction. Defaults to 5.

        Raises:
            ValueError: the parameter 'look_ahead' must be at least 1

        Returns:
            ndarray:  The result array.
        """
        if look_ahead < 1:
            raise ValueError("look_ahead must be at least 1")

        extension = [SpectralResidual.predict_next(values[-look_ahead - 2:-1])
                    ] * extend_num
        return np.r_[values, extension]

    def fit(self):
        pass

