# -*- coding: utf-8 -*-
"""Plot the data.
"""

import matplotlib.pyplot as plt
import mne
import pandas as pd
import numpy as np
import os


def pd_plot(tab,
            fig_size=(15, 10),
            cols=None,
            title=None,
            if_save=False,
            name=None):
    """Plot time series for pandas data.

    Args:
        tab: Pandas file.
        fig_size: Figure size.
        cols: Specify which cols to plot.
        title: Figure title.
        if_save: Whether or not save.
        name: Save figure name.
    """
    if cols is None:
        tab.plot(figsize=fig_size)
        plt.title(title)
        if if_save:
            plt.savefig(name)
        plt.show()

    else:
        tab[cols].plot(figsize=fig_size)
        plt.title(title)
        if if_save:
            plt.savefig(name)
        plt.show()
    return None


def mat_plot(X, y, fig_size=(15, 10), title=None, if_save=False, name=None):
    """Plot array X and y.

    Args:
        X (1darray): Array 1.
        y (1darray): Array 2.
        fig_size (tuple, optional): Size of the figure. Defaults to (15, 10).
        title (str, optional): Figure title.. Defaults to None.
        if_save (bool, optional): Whether or not save.. Defaults to False.
        name (Str, optional): Save figure name.. Defaults to None.

    """
    plt.figure(figsize=fig_size)
    plt.plot(X, y)
    plt.title(title)
    if if_save:
        plt.savefig(name)
    plt.show()


def _bar_plot():
    pass


def _plot_interval():
    pass


def _create_raw_mne(X, columns=None, sfreq=1, ch_types=None):
    """
    create mne_data instance

    Args:
        X ([type]): [description]
        columns ([type], optional): [description]. Defaults to None.
        sfreq (int, optional): [description]. Defaults to 1.

    """
    pd_data = pd.DataFrame(X, columns=columns)
    ch_names = [str(_) for _ in pd_data.columns]
    if ch_types is None:
        ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(pd_data.values.T, info)
    return raw


def plot_mne(
        X,
        columns=None,
        sfreq=1,
        duration=1000,
        start=0,
        n_channels=20,
        scalings="auto",
        ch_types=None,
        color=None,
        highpass=None,
        lowpass=None,
        filtorder=4,
):
    """plot mne raw data

    Args:
        X (numpy array): data with shape (n_samples, n_features)
        columns (list, optional): the string name or ID of each column features
        sfreq (int, optional): sample frequency. Defaults to 1.
        duration (int, optional):  Time window (s) to plot in the frame for showing.
            The lesser of this value and the duration of the raw file will be used.
            Defaults to 1000.
        start (int, optional): The start time to show. Defaults to 0.
        n_channels (int, optional): num of channels to show in one frame. Defaults
            to 20.
        scalings (dict, optional):  Scaling factors for the traces. If any fields
            in scalings are 'auto', the scaling factor is set to match the 99.5th
            percentile of a subset of the corresponding data. If
            scalings == 'auto', all scalings fields are set to 'auto'. If any
            fields are 'auto' and data is not preloaded, a subset of times up to
            100mb will be loaded. If None, defaults to::

                dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
                emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                resp=1, chpi=1e-4, whitened=1e2).

            The larger the scale is, the amplitudes of this channel will zoom smaller.

        color ( dict | color object, optional):  Color for the data traces. If
            None, defaults to::

                dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m',
                emg='k', ref_meg='steelblue', misc='k', stim='k',
                resp='k', chpi='k'). Defaults to None.

        ch_types (list, optional): Definition of channel types like
            ['eeg', 'eeg', 'eeg', 'ecg']. It can be used to change the color of
            each channel by setting `color`. Defaults to None.
        highpass (float, optional): Highpass to apply when displaying data.
            Defaults to None.
        lowpass (float, optional): Lowpass to apply when displaying data.
            If `highpass` > `lowpass`, a bandstop rather than bandpass filter
            will be applied. Defaults to None.
        filtorder (int, optional): 0 will use FIR filtering with MNE defaults.
            Other values will construct an IIR filter of the given order. This
            parameter will work when `lowpass` or `highpass` is not None.
            Defaults to 4.

    Returns:
        fig : Instance of matplotlib.figure.Figure
    """
    X = np.asarray(X)
    raw = _create_raw_mne(X, columns, sfreq, ch_types)
    fig = raw.plot(
        duration=duration,
        start=start,
        n_channels=n_channels,
        scalings=scalings,
        color=color,
        highpass=highpass,
        lowpass=lowpass,
        filtorder=filtorder,
    )
    return fig


def _plot_score(pd_data, pred_score, fig_size=(10, 8), if_save=False,
                name=None):
    fig, ax = plt.subplots(2, figsize=fig_size)
    pred_score = pd.Series(pred_score, index=pd_data.index)
    pd_data.iloc[:, 0].plot(ax=ax[0], color='black', alpha=0.8)
    pred_score.plot(ax=ax[1], color='blue', alpha=0.8)

    index = np.where(pd_data.iloc[:, -1] == 1)[0]
    temp = _get_contiu_index(index)
    for (start, end) in temp:
        start = pd_data.index[start]
        end = pd_data.index[end]
        ax[0].axvspan(start, end, alpha=0.3, color='springgreen')

    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    ax[0].set_ylabel('Origin Value', size=13)
    ax[1].set_ylabel('Anomaly Score', size=13)
    ax[0].set_title(name, size=14)

    ax[0].xaxis.set_ticklabels([])
    ax[0].set_xlim(pd_data.index[0], pd_data.index[-1])
    # ax[1].set_xlim(pd_data.index[0], pd_data.index[-1])
    plt.tight_layout()
    if if_save:
        path = os.path.join(os.path.dirname(__file__), '../../saved_fig/', name)
        dirs = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        fig.savefig(path)
        print('save fig in {:s}'.format(path))


def plot_anom(pd_data_label,
              pred_anom,
              pred_score,
              fig_size=(9, 5),
              if_save=False,
              name=None):
    """Visualize origin time series and predicted result.

    Args:
        pd_data_label (dataframe): Pandas dataframe and the last column is label.
        pred_anom (1darray): The predicted label.
        pred_score (1darray): The predicted anomaly score.
        fig_size (tuple, optional): Figure size. Defaults to (9, 5).
        if_save (bool, optional): Whether to save or not. Defaults to False.
        name (str, optional): Save file name. Defaults to None.
    """
    fig, ax = plt.subplots(2, figsize=fig_size)
    pred_score = pd.Series(pred_score, index=pd_data_label.index)
    pd_data_label.iloc[:, 0].plot(ax=ax[0], color='black', alpha=0.8)
    pred_score.plot(ax=ax[1], color='black', alpha=0.8)

    index = np.where(pd_data_label.iloc[:, -1] == 1)[0]
    temp = _get_contiu_index(index)
    for (start, end) in temp:
        start = pd_data_label.index[start]
        end = pd_data_label.index[end]
        ax[0].axvspan(start, end, alpha=0.3, color='springgreen')

    index = np.where(pred_anom == 1)[0]
    temp = _get_contiu_index(index)
    for (start, end) in temp:
        start = pd_data_label.index[start]
        end = pd_data_label.index[end]
        ax[1].axvspan(start, end, alpha=0.5, color='coral')
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    ax[0].set_ylabel('Origin Value', size=13)
    ax[1].set_ylabel('Pred Value', size=13)
    ax[0].set_title(name, size=14)

    ax[0].xaxis.set_ticklabels([])
    ax[0].set_xlim(pd_data_label.index[0], pd_data_label.index[-1])
    ax[1].set_xlim(pd_data_label.index[0], pd_data_label.index[-1])
    plt.tight_layout()
    if if_save:
        path = os.path.join(os.path.dirname(__file__), '../../saved_fig/', name)
        dirs = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        fig.savefig(path)
        print('save fig in {:s}'.format(path))


def _get_contiu_index(arr):
    """return the start and end index of a continuous intrval with 0 and 1.
    """
    from itertools import groupby
    result = []
    fun = lambda x: x[1] - x[0]
    for k, g in groupby(enumerate(arr), fun):
        temp = [j for i, j in g]
        result.append((min(temp), max(temp)))
    return result
