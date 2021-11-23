# -*- encoding: utf-8 -*-

# @Time    : 2020/01/15 15:51:50
# @Author  : ZHANG Xianrui

import pandas as pd
import numpy as np

from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.filters._utils import _maybe_get_pandas_wrapper_freq
import statsmodels.api as sm
from .base import BaseModel

__all__=['STL']
class STL(BaseModel):
    def __init__(self):
        super(STL, self).__init__()

    def fit(self, df, period=365, lo_frac=0.6, lo_delta=0.01):
        """Train the STL decompose model.Y[t] = T[t] + S[t] + e[t]

        Args:
            df (DataFrame): Input data.
            period (int, optional): Defaults to 365.
            lo_frac (float, optional): Defaults to 0.6.
            lo_delta (float, optional): Defaults to 0.01.

        Returns:
            dict: Dict results.
        """
        _pandas_wrapper, _ = _maybe_get_pandas_wrapper_freq(df)
        observed = df.values.squeeze()
        # calc trend, remove from observation
        trend = STL.calc_trend(observed, lo_frac, lo_delta)
        detrended = observed - trend
        seasonal, period_averages = STL.calc_seasonal(detrended, period)
        resid = detrended - seasonal
        results = list(map(_pandas_wrapper, [seasonal, trend, resid, observed]))
        dr = {
            'seasonal': results[0],
            'trend': results[1],
            'resid': results[2],
            'observed': results[3],
            'period_averages': period_averages
        }
        return dr

    def forecast(self, stl, forecast_func='drift', steps=10, seasonal=False):
        """Forecast the given decomposition ``stl`` forward by ``steps``

        Args:
            stl (object): STL object.
            forecast_func (str, optional): Defaults to 'drift'.
            steps (int, optional): Defaults to 10.
            seasonal (bool, optional): Defaults to False.

        Returns:
            DataFrame: forecast dataframe
        """
        forecast_array = np.array([])
        if forecast_func == 'naive':
            fc_func = STL.naive
        elif forecast_func == 'mean':
            fc_func = STL.mean
        else:
            fc_func = STL.drift

        # forecast trend
        trend_array = stl['trend']
        for step in range(steps):
            # make this prediction on all available data
            pred = fc_func(np.append(trend_array, forecast_array))
            # add this prediction to current array
            forecast_array = np.append(forecast_array, pred)
        col_name = fc_func.__name__

        # forecast start and index are determined by observed data
        observed_timedelta = stl['observed'].index[-1] - stl['observed'].index[
            -2]
        forecast_idx_start = stl['observed'].index[-1] + observed_timedelta
        forecast_idx = pd.date_range(
            start=forecast_idx_start,
            periods=steps,
            freq=pd.tseries.frequencies.to_offset(observed_timedelta))

        if seasonal:
            seasonal_ix = 0
            max_correlation = -np.inf
            detrended_array = (stl['observed'] - stl['trend']).values.squeeze()
            for i, x in enumerate(stl['period_averages']):
                if i == 0:
                    detrended_slice = detrended_array[
                        -len(stl['period_averages']):]
                else:
                    detrended_slice = detrended_array[-(
                        len(stl['period_averages']) + i):-i]

                this_correlation = np.correlate(detrended_slice,
                                                stl['period_averages'])[0]
                if this_correlation > max_correlation:
                    max_correlation = this_correlation
                    seasonal_ix = i
            rolled_period_averages = np.roll(stl['period_averages'],
                                             -seasonal_ix)
            tiled_averages = np.tile(
                rolled_period_averages,
                (steps // len(stl['period_averages']) + 1))[:steps]
            forecast_array += tiled_averages
            col_name += '+seasonal'

        forecast_frame = pd.DataFrame(data=forecast_array, index=forecast_idx)
        forecast_frame.columns = [col_name]
        return forecast_frame

    def detect(self):
        pass

    @staticmethod
    def naive(data, n=7):
        result = data[-n]
        return result

    @staticmethod
    def mean(data, n=3):
        if len(data[-n:]) < n:
            result = np.nan
        else:
            # nb: we'll keep the forecast as a float
            result = np.mean(data[-n:])
        return result

    @staticmethod
    def drift(data, n=3):
        """The drift forecast for the next point is a linear extrapolation from
            the previous n points in the series.

        Args:
            data (ndrray): Observed data, presumed to be ordered in time.
            n (int): period over which to calculate linear model for
                extrapolation.

        Returns:
            float: a single-valued forecast for the next value in the series.
        """
        yi = data[-n]
        yf = data[-1]
        slope = (yf - yi) / (n - 1)
        result = yf + slope
        return result

    @staticmethod
    def calc_trend(observed, lo_frac=0.6, lo_delta=0.01):
        """calculate trend from observed data.

        Args:
            observed (ndarray): Input array.
            lo_frac (float, optional): Defaults to 0.6.
            lo_delta (float, optional): Defaults to 0.01.

        Returns:
            ndarray: The trend.
        """
        trend = sm.nonparametric.lowess(
            observed,
            np.arange(0, len(observed)),
            frac=lo_frac,
            delta=lo_delta * len(observed),
            return_sorted=False)
        return trend

    @staticmethod
    def calc_seasonal(detrended, period):
        """Calculate seasonal from detrended data.

        Args:
            detrended (ndarray): Input detrended data.
            period (float or int): The period of data.

        Returns:
            (ndarray, ndarray): The seasonal and the period_averages.
        """
        # period must not be larger than size of series
        period = min(period, len(detrended))
        period_averages = np.array(
            [pd_nanmean(detrended[i::period]) for i in range(period)])

        period_averages -= np.mean(period_averages)
        seasonal = np.tile(period_averages,
                           len(detrended) // period + 1)[:len(detrended)]
        return seasonal, period_averages