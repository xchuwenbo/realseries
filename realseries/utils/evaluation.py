# -*- coding: utf-8 -*-
"""Evaluation function.
"""
# Author: Wenbo Hu <i@wbhu.net>

import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def evaluate(y_true, y_pred):
    """Eval metrics. Here 1 stand for anomaly label and 0 is normal samples.

    Args:
        y_true (1-D array_like): The actual value.
        y_pred (1-D array_like): The predictive value.

    Returns:
        dict: a dictionary which includes mse, rmse, mae and r2.
    """
    mse_value = mse(y_true, y_pred)
    rmse_value = (mse(y_true, y_pred))**(1 / 2)
    mae_value = mae(y_true, y_pred)
    r2_value = r2_score(y_true, y_pred)

    return {
        "mse": mse_value,
        "rmse": rmse_value,
        "mae": mae_value,
        "r2": r2_value,
    }


def baseline_oneday(y_true):
    """Use the previous value as the predicted value
    
    Args:
        y_true (1-D arral_like):  Auto-regressive inputs.
    
    Returns:
        dict: Evaluation result of one-day ahead baselinesss
    """
    return evaluate(y_true[1:], y_true[:-1])


def baseline_threeday(y_true):
    """Use the average of 3 previous value as the predicted value.
    
    Args:
        y_true (aray_like): Auto-regressive inputs.
    
    Returns:
        dcit: Evaluation result of three-day-ahead-average baseline.
    """    

    y_pred = []
    for i in range(3, len(y_true)):
        t = (y_true[i - 1] + y_true[i - 2] + y_true[i - 3]) / 3
        y_pred.append(t)
    return evaluate(y_true[3:], y_pred)


def point_metrics(y_pred, y_true, beta=1.0):
    """Calculate precison recall f1 bny point to point comparison.

    Args:
        y_pred (ndarray): The predicted y.
        y_true (ndarray): The true y.
        beta (float): The balance for calculating `f score`.

    Returns:
        tuple: Tuple contains ``precision, recall, f1, tp, tn, fp, fn``.
    """
    idx = y_pred * 2 + y_true
    tn = (idx == 0).sum()  # tn 0+0
    fn = (idx == 1).sum()  # fn 0+1
    fp = (idx == 2).sum()  # fp 2+0
    tp = (idx == 3).sum()  # tp 2+1

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = (1 + beta**2) * (precision * recall) / (
        beta**2 * precision + recall + 1e-7)
    return precision, recall, f1, tp, tn, fp, fn


# def adjust_predicts(score, label, threshold=None, pred=None,
#                     calc_latency=False):
#     """
#     Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

#     Args:
#         score (np.ndarray): The anomaly score
#         label (np.ndarray): The ground-truth label
#         threshold (float): The threshold of anomaly score.
#             A point is labeled as "anomaly" if its score is lower than the threshold.
#         pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
#         calc_latency (bool):

#     Returns:
#         np.ndarray: predict labels
#     """
#     if len(score) != len(label):
#         raise ValueError("score and label must have the same length")
#     score = np.asarray(score)
#     label = np.asarray(label)
#     latency = 0
#     if pred is None:
#         predict = score < threshold
#     else:
#         predict = pred
#     actual = label > 0.1
#     anomaly_state = False
#     anomaly_count = 0
#     for i in range(len(score)):
#         if actual[i] and predict[i] and not anomaly_state:
#             anomaly_state = True
#             anomaly_count += 1
#             for j in range(i, 0, -1):
#                 if not actual[j]:
#                     break
#                 else:
#                     if not predict[j]:
#                         predict[j] = True
#                         latency += 1
#         elif not actual[i]:
#             anomaly_state = False
#         if anomaly_state:
#             predict[i] = True
#         '''
#         pred ,actual,flag->, , ,
#         0 0 true-> 0 0 false
#         0 0 false -> 0 0 false
#         0 1 true -> 1 1 true
#         0 1 false -> 0 1 false
#         1 0 true ->  1 0  false
#         1 0 false -> 1 0 false
#         1 1 true -> 1 1 true
#         1 1 false -> 1 1 true 需要纠正
#         '''
#     if calc_latency:
#         return predict, latency / (anomaly_count + 1e-4)
#     else:
#         return predict


def adjust_predicts(predict, label, delay=7):
    """Adjust the predicted results.
    
    Args:
        predict (ndarray): The predicted y.
        label (ndarray): The true y label.
        delay (int, optional): The max allowed delay of the anomaly occuring.
            Defaults to 7.
    
    Returns:
        naarray: The adjusted predicted array y.
    """    
    predict = np.array(predict)
    label = np.array(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos:sp] = 1
            else:
                new_predict[pos:sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:
        if 1 in predict[pos:min(pos + delay + 1, sp)]:
            new_predict[pos:sp] = 1
        else:
            new_predict[pos:sp] = 0

    return new_predict


def adjust_metrics(pred, label, delay=7, beta=1.0):
    """Calculating the precison and recall etc. using adjusted label.
    
    Args:
        pred (ndarray): The predicted y.
        label (ndarray): The true y label.
        delay (int, optional): The max allowed delay of the anomaly occuring.
            Defaults to 7.
        beta (float, optional): The balance between presicion and recall
            for`` f score``. Defaults to 1.0.

    Returns:
        tuple: Tuple contains ``precision, recall, f1, tp, tn, fp, fn``.

    """
    new_pred = adjust_predicts(pred, label, delay)
    return point_metrics(new_pred, label, beta)


def thres_search(score,
                 label,
                 num_samples=1000,
                 beta=1.0,
                 sampling='log',
                 adjust=True,
                 delay=7):
    """Find the best-f1 score by searching best `threshold`

    Args:
        score (ndarray): The anomaly score.
        label (ndarray): The true label.
        num_samples (int): The number of sample points between ``[min_score, max_score]``.
        beta (float, optional): The balance between precison and recall in
            ``f score``. Defaults to 1.0.
        sampling (str, optional): The sampling method including 'log' and 'linear'.
            Defaults to 'log'.

    Returns:
        tuple: Results in best threshold ``precison, recall, f1, best_thres, predicted labele``.
    """
    score,label = np.asarray(score),np.asarray(label)
    maximum = score.max()
    if sampling == 'log':
        # Sample thresholds logarithmically
        th = np.logspace(0, np.log10(maximum), num_samples)
    else:
        # Sample thresholds equally
        # The sampled thresholds are equally spaced points
        th = np.linspace(0, maximum, num_samples)

    temp_pre, temp_recall, temp_f1 = -1., -1., -1.
    for i in range(len(th)):
        y_pred = score > th[i]
        if adjust is True:
            pre, recall, f1, _, _, _, _ = adjust_metrics(y_pred, label, delay)
        else:
            pre, recall, f1, _, _, _, _ = point_metrics(y_pred, label, beta)
        if f1 > temp_f1:
            temp_f1 = f1
            temp_pre = pre
            temp_recall = recall
            best_thres = th[i]
    print('pre:{}, recall:{}, f1:{}'.format(temp_pre, temp_recall, temp_f1))
    return temp_pre, temp_recall, temp_f1, best_thres,score>best_thres