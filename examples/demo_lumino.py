import sys
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
# add RealSeries/realseries to sys path for import realseries
sys.path.append(str(Path.cwd()))
from realseries.models.iforest import IForest
from realseries.models.lumino import Lumino
from realseries.models.rcforest import RCForest
from realseries.utils.evaluation import point_metrics, adjust_predicts
from realseries.utils.data import load_split_NASA, load_Yahoo, load_NAB
from realseries.utils.visualize import plot_anom
warnings.filterwarnings("ignore")
cwd = str(Path.cwd())

def main(args, train_set, test_set, model_path):
    # the last column is label; other columns are values
    test_label = test_set.iloc[:, -1]

    # lumino model
    model = Lumino()
    X = test_set.iloc[:,0]
    score = model.detect(X)

    # find thres
    thres_percent=args.thres_percent
    thres = np.percentile(score,thres_percent)
    pred_label = (score>thres)

    precision, recall, f1, tp, tn, fp, fn = point_metrics(
        pred_label, test_label)
    print('precision:{}, recall:{}, f1:{}, tp:{}, tn:{}, fp:{}, fn:{}'.format(
        precision, recall, f1, tp, tn, fp, fn))

    # adjust the pred_label for better visualization
    adjust_pred_label = adjust_predicts(pred_label, test_label, delay=args.delay)
    precision, recall, f1, tp, tn, fp, fn = point_metrics(
        adjust_pred_label, test_label)
    print('precision:{}, recall:{}, f1:{}, tp:{}, tn:{}, fp:{}, fn:{}'.format(
        precision, recall, f1, tp, tn, fp, fn))

    result = {
        'pred_label': pred_label,
        'adjust_pred_label': adjust_pred_label,
        'test_label': test_label,
        'precision': precision,
        'recall': recall
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Iforest.')
    parser.add_argument('--contamination', type=float, default=0.01)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--fraction', type=float, default=0.5)
    parser.add_argument('--thres_percent', type=float, default=99.5)
    parser.add_argument('--delay', type=int, default=24)
    args = parser.parse_args(args=[])

    # data_type = 'NAB'
    # data_type = 'NASA'
    data_type = 'Yahoo'

    if data_type is 'NAB':
        temp = []
        dirname = 'realKnownCause'
        print(dirname)
        filenames = os.listdir(Path(cwd, f'examples/data/NAB_data/{dirname}'))
        for filename in filenames:
            filename = 'nyc_taxi.csv'
            print(filename)
            args.filename = filename
            train_set, test_set = load_NAB(
                dirname, filename, fraction=args.fraction)
            model_path = Path(cwd, f'snapshot/sr/{dirname}',
                                args.filename[:-4])
            print(f'model path: {model_path}')
            result = main(args, train_set, test_set, model_path)
            temp.append(result)

    elif data_type is 'Yahoo':
        adjust_pred_label, test_label = [], []
        dirnames = ['A1Benchmark', 'A2Benchmark', 'A3Benchmark', 'A4Benchmark']
        dirname = 'A1Benchmark'
        filenames = os.listdir(Path(cwd,f'examples/data/Yahoo_data/{dirname}'))
        filenames = [i for i in filenames if 'csv' in i]
        for filename in filenames:
            # filename = 'nyc_taxi.csv'
            print(filename)
            args.filename = filename
            train_set, test_set = load_Yahoo(
                dirname, filename, fraction=args.fraction)
            train_set, test_set = train_set[['value',
                                                'anomaly']], test_set[[
                                                    'value', 'anomaly'
                                                ]]

            model_path = Path(cwd, args.filename[:-4])
            print(f'model path: {model_path}')
            result = main(args, train_set, test_set, model_path)
            adjust_pred_label.append(result['adjust_pred_label'])
            test_label.append(result['test_label'])
        pred = np.concatenate(adjust_pred_label)
        label = np.concatenate(test_label)
        print (point_metrics(pred,label))

    elif data_type is 'NASA':
        adjust_pred_label, test_label = [], []
        for chan_id in NASA_chans[:1]:
            chan_id = 'A-4'
            print(chan_id)
            model_path = Path(cwd, f'snapshot/lstm_dym/{chan_id}')
            train_set, test_set = load_split_NASA(chan_id)
            result = main(args, train_set, test_set, model_path)

            adjust_pred_label.append(result['adjust_pred_label'])
            test_label.append(result['test_label'])
