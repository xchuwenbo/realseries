# -*- coding: utf-8 -*-
""""The function in lstm dynamic method.
"""
import numpy as np
import pandas as pd
import more_itertools as mit
import os

__all__ = ['get_errors', 'process_errors']


def get_errors(batch_size,
               window_size,
               smoothing_perc,
               y_test,
               y_hat,
               smoothed=True):
    """Calculate the difference between predicted telemetry values and actual values, then smooth residuals using
    ewma to encourage identification of sustained errors/anomalies.

    Args:
        batch_size (int): Number of values to evaluate in each batch in the prediction stage.
        window_size (int): Window_size to use in error calculation.
        smoothing_perc (float): Percentage of total values used in EWMA smoothing.
        y_test (ndarray): Array of test targets corresponding to true values to be predicted at end of each sequence
        y_hat (ndarray): predicted test values for each timestep in y_test
        smoothed (bool, optional): If False, return unsmooothed errors (used for assessing quality of predictions)

    Returns:
        e (list): unsmoothed errors (residuals)
        e_s (list): smoothed errors (residuals)
    """

    e = [abs(y_h - y_t[0]) for y_h, y_t in zip(y_hat, y_test)]

    if not smoothed:
        return e

    smoothing_window = int(batch_size * window_size * smoothing_perc)
    if not len(y_hat) == len(y_test):
        raise ValueError(
            "len(y_hat) != len(y_test), can't calculate error: %s (y_hat) , %s (y_test)"
            % (len(y_hat), len(y_test)))

    e_s = list(
        pd.DataFrame(e).ewm(span=smoothing_window).mean().values.flatten())

    # for values at beginning < sequence length, just use avg
    # if not anom['chan_id'] == 'C-2': #anom occurs early in window (limited data available for channel)
    #     e_s[:l_s] = [np.mean(e_s[:l_s*2])]*l_s
    return e_s


def process_errors(
        p,
        l_s,
        batch_size,
        window_size,
        error_buffer,
        y_test,
        y_hat,
        e_s,
):
    """Using windows of historical errors (h = batch size * window size), calculate the anomaly
    threshold (epsilon) and group any anomalous error values into continuos sequences. Calculate
    score for each sequence using the max distance from epsilon.

    Args:
        p (float, optional): Minimum percent decrease between max errors in
            anomalous sequences (used for pruning).
        l_s (int, optional): Length of the input sequence for LSTM.
        batch_size (int): Number of values to evaluate in each  batch in the
            prediction stage.
        window_size (int): Window_size to use in error calculation.
        error_buffer (int, optional): Number of values surrounding an error
            that are brought into the sequence.
        y_test (np array): test targets corresponding to true telemetry values
            at each timestep `t`.
        y_hat (np array): test target predictions at each timestep `t`.
        e_s (list): smoothed errors (residuals) between ``y_test`` and ``y_hat``.

    Returns:
        E_seq (list of tuples): Start and end indices for each anomaloues sequence.
        anom_scores (list): Score for each anomalous sequence.
    """

    i_anom = []  # anomaly indices
    window_size = window_size

    num_windows = int(
        (y_test.shape[0] - (batch_size * window_size)) / batch_size)

    # decrease the historical error window size (h) if number of test values is limited
    while num_windows < 0:
        window_size -= 1
        if window_size <= 0:
            window_size = 1
        num_windows = int(
            (y_test.shape[0] - (batch_size * window_size)) / batch_size)
        if window_size == 1 and num_windows < 0:
            raise ValueError("Batch_size (%s) larger than y_test (len=%s)." %
                             (batch_size, y_test.shape[0]))

    # Identify anomalies for each new batch of values
    for i in range(1, num_windows + 2):
        prior_idx = (i - 1) * (batch_size)
        idx = (window_size * batch_size) + ((i - 1) * batch_size)

        if i == num_windows + 1:
            idx = y_test.shape[0]

        window_e_s = e_s[prior_idx:idx]
        window_y_test = y_test[prior_idx:idx]

        epsilon = find_epsilon(e_s=window_e_s, error_buffer=error_buffer)
        window_anom_indices = get_anomalies(
            p=p,
            l_s=l_s,
            batch_size=batch_size,
            error_buffer=error_buffer,
            e_s=window_e_s,
            y_test=window_y_test,
            z=epsilon,
            window=i - 1,
            i_anom_full=i_anom,
            len_y_test=len(y_test))
        # update indices to reflect true indices in full set of values (not just window)
        i_anom.extend(
            [i_a + (i - 1) * batch_size for i_a in window_anom_indices])

    # group anomalous indices into continuous sequences
    i_anom = sorted(list(set(i_anom)))
    groups = [list(group) for group in mit.consecutive_groups(i_anom)]
    E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

    # calc anomaly scores based on max distance from epsilon for each sequence
    anom_scores = []
    for e_seq in E_seq:
        score = max([
            abs(e_s[x] - epsilon) / (np.mean(e_s) + np.std(e_s))
            for x in range(e_seq[0], e_seq[1])
        ])
        anom_scores.append(score)

    return E_seq, anom_scores


def get_anomalies(p, l_s, batch_size, error_buffer, e_s, y_test, z, window,
                  i_anom_full, len_y_test):
    """Find anomalous sequences of smoothed error values that are above error threshold (epsilon). Both
    smoothed errors and the inverse of the smoothed errors are evaluated - large dips in errors often
    also indicate anomlies.

    Args:
        e_s (list): smoothed errors between y_test and y_hat values
        y_test (np array): test targets corresponding to true telemetry values at each timestep for given window
        z (float): number of standard deviations above mean corresponding to epsilon
        window (int): number of error windows that have been evaluated
        i_anom_full (list): list of all previously identified anomalies in test set
        len_y_test (int): num total test values available in dataset

    Returns:
        i_anom (list): indices of errors that are part of an anomlous sequnces
    """

    perc_high, perc_low = np.percentile(y_test, [95, 5])
    inter_range = perc_high - perc_low

    mean = np.mean(e_s)
    std = np.std(e_s)
    chan_std = np.std(y_test)

    e_s_inv = [mean + (mean - e) for e in e_s]  # flip it around the mean
    z_inv = find_epsilon(e_s=e_s_inv, error_buffer=error_buffer)

    epsilon = mean + (float(z) * std)
    epsilon_inv = mean + (float(z_inv) * std)

    # find sequences of anomalies greater than epsilon
    E_seq, i_anom, non_anom_max = compare_to_epsilon(l_s, batch_size, e_s,
                                                     epsilon, len_y_test,
                                                     inter_range, chan_std, std,
                                                     error_buffer, window,
                                                     i_anom_full)

    # find sequences of anomalies using inverted error values (lower than normal errors are also anomalous)
    E_seq_inv, i_anom_inv, inv_non_anom_max = compare_to_epsilon(
        l_s, batch_size, e_s_inv, epsilon_inv, len_y_test, inter_range,
        chan_std, std, error_buffer, window, i_anom_full)

    if len(E_seq) > 0:
        i_anom = prune_anoms(
            p=p, E_seq=E_seq, e_s=e_s, non_anom_max=non_anom_max, i_anom=i_anom)
    if len(E_seq_inv) > 0:
        i_anom_inv = prune_anoms(
            p=p,
            E_seq=E_seq_inv,
            e_s=e_s_inv,
            non_anom_max=inv_non_anom_max,
            i_anom=i_anom_inv)

    i_anom = list(set(i_anom + i_anom_inv))

    return i_anom


def compare_to_epsilon(l_s, batch_size, e_s, epsilon, len_y_test, inter_range,
                       chan_std, std, error_buffer, window, i_anom_full):
    '''Compare smoothed error values to epsilon (error threshold) and group consecutive errors together into
    sequences.

    Args:
        e_s (list): smoothed errors between y_test and y_hat values
        epsilon (float): Threshold for errors above which an error is considered anomalous
        len_y_test (int): number of timesteps t in test data
        inter_range (tuple of floats): range between 5th and 95 percentile values of error values
        chan_std (float): standard deviation on test values
        std (float): standard deviation of smoothed errors
        error_buffer (int): number of values surrounding anomalous errors to be included in anomalous sequence
        window (int): Count of number of error windows that have been processed
        i_anom_full (list): list of all previously identified anomalies in test set

    Returns:
        E_seq (list of tuples): contains start and end indices of anomalous ranges
        i_anom (list): indices of errors that are part of an anomlous sequnce
        non_anom_max (float): highest smoothed error value below epsilon
    '''

    i_anom = []
    E_seq = []
    non_anom_max = 0

    # Don't consider anything in window because scale of errors too small compared to scale of values
    if not (std > (.05 * chan_std) or max(e_s) >
            (.05 * inter_range)) or not max(e_s) > 0.05:
        return E_seq, i_anom, non_anom_max

    # ignore initial error values until enough history for smoothing, prediction, comparisons
    num_to_ignore = l_s * 2
    # if y_test is small, ignore fewer
    if len_y_test < 2500:
        num_to_ignore = l_s
    if len_y_test < 1800:
        num_to_ignore = 0

    for x in range(0, len(e_s)):

        anom = True
        if not e_s[x] > epsilon or not e_s[x] > 0.05 * inter_range:
            anom = False

        if anom:
            for b in range(0, error_buffer):
                if not x + b in i_anom and not x + b >= len(e_s) and (
                    (x + b) >= len(e_s) - batch_size or window == 0):
                    if not (window == 0 and x + b < num_to_ignore):
                        i_anom.append(x + b)
                # only considering new batch of values added to window, not full window
                if not x - b in i_anom and ((x - b) >= len(e_s) - batch_size or
                                            window == 0):
                    if not (window == 0 and x - b < num_to_ignore):
                        i_anom.append(x - b)

    # capture max of values below the threshold that weren't previously identified as anomalies
    # (used in filtering process)
    for x in range(0, len(e_s)):
        adjusted_x = x + window * batch_size
        if e_s[x] > non_anom_max and adjusted_x not in i_anom_full and x not in i_anom:
            non_anom_max = e_s[x]

    # group anomalous indices into continuous sequences
    i_anom = sorted(list(set(i_anom)))
    groups = [list(group) for group in mit.consecutive_groups(i_anom)]
    E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

    return E_seq, i_anom, non_anom_max


def prune_anoms(p, E_seq, e_s, non_anom_max, i_anom):
    '''Remove anomalies that don't meet minimum separation from the next closest anomaly or error value

    Args:
        E_seq (list of lists): contains start and end indices of anomalous ranges
        e_s (list): smoothed errors between y_test and y_hat values
        non_anom_max (float): highest smoothed error value below epsilon
        i_anom (list): indices of errors that are part of an anomlous sequnce

    Returns:
        i_pruned (list): remaining indices of errors that are part of an anomlous sequnces
            after pruning procedure
    '''

    E_seq_max, e_s_max = [], []
    for e_seq in E_seq:
        if len(e_s[e_seq[0]:e_seq[1]]) > 0:
            E_seq_max.append(max(e_s[e_seq[0]:e_seq[1]]))
            e_s_max.append(max(e_s[e_seq[0]:e_seq[1]]))
    e_s_max.sort(reverse=True)

    if non_anom_max and non_anom_max > 0:
        e_s_max.append(
            non_anom_max
        )  # for comparing the last actual anomaly to next highest below epsilon

    i_to_remove = []

    for i in range(0, len(e_s_max)):
        if i + 1 < len(e_s_max):
            if (e_s_max[i] - e_s_max[i + 1]) / e_s_max[i] < p:
                i_to_remove.append(E_seq_max.index(e_s_max[i]))
                # p += 0.03 # increase minimum separation by this amount for each step further from max error
            else:
                i_to_remove = []
    for idx in sorted(i_to_remove, reverse=True):
        del E_seq[idx]

    i_pruned = []
    for i in i_anom:
        keep_anomaly_idx = False

        for e_seq in E_seq:
            if i >= e_seq[0] and i <= e_seq[1]:
                keep_anomaly_idx = True

        if keep_anomaly_idx is True:
            i_pruned.append(i)

    return i_pruned


def find_epsilon(e_s, error_buffer, sd_lim=12.0):
    '''Find the anomaly threshold that maximizes function representing tradeoff between a) number of anomalies
    and anomalous ranges and b) the reduction in mean and st dev if anomalous points are removed from errors
    (see https://arxiv.org/pdf/1802.04431.pdf)

    Args:
        e_s (array): residuals between y_test and y_hat values (smoothes using ewma)
        error_buffer (int): if an anomaly is detected at a point, this is the number of surrounding values
            to add the anomalous range. this promotes grouping of nearby sequences and more intuitive results
        sd_lim (float): The max number of standard deviations above the mean to calculate as part of the
            argmax function

    Returns:
        sd_threshold (float): the calculated anomaly threshold in number of standard deviations above the mean
    '''

    mean = np.mean(e_s)
    sd = np.std(e_s)

    max_s = 0
    sd_threshold = sd_lim  # default if no winner or too many anomalous ranges

    for z in np.arange(2.5, sd_lim, 0.5):
        epsilon = mean + (sd * z)
        pruned_e_s, pruned_i, i_anom = [], [], []

        for i, e in enumerate(e_s):
            if e < epsilon:
                pruned_e_s.append(e)
                pruned_i.append(i)
            if e > epsilon:
                for j in range(0, error_buffer):
                    if not i + j in i_anom and not i + j >= len(e_s):
                        i_anom.append(i + j)
                    if not i - j in i_anom and not i - j < 0:
                        i_anom.append(i - j)

        if len(i_anom) > 0:
            # preliminarily group anomalous indices into continuous sequences (# sequences needed for scoring)
            i_anom = sorted(list(set(i_anom)))
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            perc_removed = 1.0 - (float(len(pruned_e_s)) / float(len(e_s)))
            mean_perc_decrease = (mean - np.mean(pruned_e_s)) / mean
            sd_perc_decrease = (sd - np.std(pruned_e_s)) / sd
            s = (mean_perc_decrease + sd_perc_decrease) / (
                len(E_seq)**2 + len(i_anom))

            # sanity checks
            if s >= max_s and len(E_seq) <= 5 and len(i_anom) < (len(e_s) *
                                                                 0.5):
                sd_threshold = z
                max_s = s

    return sd_threshold  # multiply by sd to get epsilon
