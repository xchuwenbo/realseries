# -*- encoding: utf-8 -*-
"""Preprocess function
"""
import torch
import numpy as np
import pandas as pd
import scipy
import scipy.signal


def normalization(X):
    """Normalization to [0, 1] on each column data of input array.

    Args:
        X (array_like): The input array for formalization.

    Returns:
        ndarray : Normalized array in [0, 1].
    """
    X = np.asarray(X)
    if len(X.shape) == 1:
        X = (X - X.min()) / (X.max() - X.min())
        return X
    for i in range(X.shape[-1]):
        temp = X[:, i]
        temp = (temp - temp.min()) / (temp.max() - temp.min())
        X[:, i] = temp
    return X


def standardization(X):
    """Standardization each column data by reduce mean and divide std.

    Args:
        X (array_like): The input array for standardization.

    Returns:
        ndarray : Standardized array with 0 mean and 1 std.
    """
    X = np.asarray(X)
    if len(X.shape) == 1:
        X = (X - X.mean()) / X.std()
        return X
    for i in range(X.shape[-1]):
        temp = X[:, i]
        temp = (temp - temp.mean()) / temp.std()
        X[:, i] = temp
    return X


def augmentation(data,
                 label,
                 noise_ratio=0.05,
                 noise_interval=0.0005,
                 max_length=100000):
    """Data augmentation by add anomaly points to origin data.

    Args:
        data (array_like): The origin data.
        label (array_like): The origin label.
        noise_ratio (float, optional): The ratio of adding noise to data.
            Defaults to 0.05.
        noise_interval (float, optional): Noise_interval. Defaults to 0.0005.
        max_length (int, optional): The max length of data after augmentation.
            Defaults to 100000.
    """
    data, label = torch.from_numpy(np.asarray(data)).float(), torch.from_numpy(
        np.asarray(label)).float()
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    noiseSeq = torch.randn(data.size())
    augmentedData = data.clone()
    augmentedLabel = label.clone()
    for i in np.arange(0, noise_ratio, noise_interval):
        scaled_noiseSeq = noise_ratio * std.expand_as(data) * noiseSeq
        augmentedData = torch.cat([augmentedData, data + scaled_noiseSeq],
                                  dim=0)
        augmentedLabel = torch.cat([augmentedLabel, label])
        if len(augmentedData) > max_length:
            augmentedData = augmentedData[:max_length]
            augmentedLabel = augmentedLabel[:max_length]
            break
    return augmentedData.numpy(), augmentedLabel.numpy()


def exponential_running_standardize(data,
                                    factor_new=0.001,
                                    init_block_size=None,
                                    eps=1e-4):
    """Perform exponential running standardization.

    Compute the exponental running mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential running variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.

    Args:
        data (2darray): The shape is (time, channels)
        factor_new (float, optional): Defaults to 0.001.
        init_block_size (int, optional): Standardize data before to this index
            with regular standardization. Defaults to None.
        eps (float, optional): Stabilizer for division by zero variance.. Defaults to 1e-4.

    Returns:
        2darray: Standardized data (time, channels).
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True)
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True)
        init_block_standardized = (data[0:init_block_size] -
                                   init_mean) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized


def exponential_running_demean(data, factor_new=0.001, init_block_size=None):
    """Perform exponential running demeanining.
    Compute the exponental running mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    Deman the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t)`.

    Args:
        data (2darray): Shape is (time, channels)
        factor_new (float, optional): Defaults to 0.001.
        init_block_size (int, optional): Demean data before to this index with regular demeaning. Defaults to None.

    Returns:
        2darray: Demeaned data (time, channels).
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True)
        demeaned[0:init_block_size] = data[0:init_block_size] - init_mean
    return demeaned


def highpass_cnt(data, low_cut_hz, fs, filt_order=3, axis=0):
    """signal applying **causal** butterworth filter of given order.

    Args:
        data (2d-array): Time x channels.
        low_cut_hz (float): Low cut frequency HZ.
        fs (float): Sample frequency.
        filt_order (int): Defaults to 3.
        axis (int, optional): Time axis. Defaults to 0.

    Returns:
        highpassed_data (2d-array): Data after applying highpass filter.
    """
    if (low_cut_hz is None) or (low_cut_hz == 0):
        log.info("Not doing any highpass, since low 0 or None")
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, low_cut_hz / (fs / 2.0), btype="highpass")
    assert filter_is_stable(a)
    data_highpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_highpassed


def lowpass_cnt(data, high_cut_hz, fs, filt_order=3, axis=0):
    """Lowpass signal applying **causal** butterworth filter of given order.

    Args:
        data (2d-array): Time x channels.
        high_cut_hz ([type]): High cut frequency.
        fs ([type]): Sample frequency.
        filt_order (int, optional): Defaults to 3.

    Returns:
        2d-array: Data after applying lowpass filter.
    """

    if (high_cut_hz is None) or (high_cut_hz == fs / 2.0):
        log.info(
            "Not doing any lowpass, since high cut hz is None or nyquist freq.")
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, high_cut_hz / (fs / 2.0), btype="lowpass")
    assert filter_is_stable(a)
    data_lowpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_lowpassed


def bandpass_cnt(data,
                 low_cut_hz,
                 high_cut_hz,
                 fs,
                 filt_order=3,
                 axis=0,
                 filtfilt=False):
    """Bandpass signal applying **causal** butterworth filter of given order.

    Args:
        data ( 2d-array): Time x channels.
        low_cut_hz (float): Low cut hz.
        high_cut_hz (float): High cut hz.
        fs (float): Sample frequency.
        filt_order (int, optional): Defaults to 3.
        axis (int, optional): Time axis. Defaults to 0.
        filtfilt (bool, optional): Whether to use filtfilt instead of lfilter. Defaults to False.

    Returns:
         2d-array: Data after applying bandpass filter.
    """

    if (low_cut_hz == 0 or low_cut_hz is None) and (high_cut_hz == None or
                                                    high_cut_hz == fs / 2.0):
        log.info("Not doing any bandpass, since low 0 or None and "
                 "high None or nyquist frequency")
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        log.info("Using lowpass filter since low cut hz is 0 or None")
        return lowpass_cnt(
            data, high_cut_hz, fs, filt_order=filt_order, axis=axis)
    if high_cut_hz == None or high_cut_hz == (fs / 2.0):
        log.info(
            "Using highpass filter since high cut hz is None or nyquist freq")
        return highpass_cnt(
            data, low_cut_hz, fs, filt_order=filt_order, axis=axis)

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype="bandpass")
    assert filter_is_stable(a), "Filter should be stable..."
    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed


def filter_is_stable(a):
    """Check if filter coefficients of IIR filter are stable.

    Args:
        a (list): list or 1darray of number. Denominator filter coefficients a.

    Returns:
        bool: Filter is stable or not.

    Notes:
        Filter is stable if absolute value of all  roots is smaller than 1,
        see http://stackoverflow.com/a/8812737/1469195.
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a)))
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a)) < 1)