# note the current workdirect cwd is './RealSeries'
import sys
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# add RealSeries/realseries to sys path
sys.path.append(str(Path.cwd()))
from realseries.models.lstm_dynamic import LSTM_dynamic
from realseries.utils.evaluation import point_metrics, adjust_predicts
from realseries.utils.data import load_split_NASA, load_Yahoo, load_NAB
from realseries.utils.preprocess import normalization,standardization
warnings.filterwarnings("ignore")
cwd = str(Path.cwd())

def main(args,train_set,test_st,model_path):
    train_data, train_label = train_set.iloc[:, :-1], train_set.iloc[:, -1]
    test_data, test_label = test_set.iloc[:, :-1], test_set.iloc[:, -1]

    train_data,test_data = standardization(train_data),standardization(test_data)

    # init the model class
    lstm_dym = LSTM_dynamic(
        batch_size=args.batch_size,
        window_size=args.window_size,
        smoothing_perc=args.smoothing_perc,
        error_buffer=args.error_buffer,
        dropout=args.dropout,
        lstm_batch_size=args.lstm_batch_size,
        epochs=args.epochs,
        num_layers=args.num_layers,
        l_s=args.l_s,
        n_predictions=args.n_predictions,
        p=args.p,
        model_path=model_path,
        hidden_size=args.hidden_size,
        lr=args.lr)

    if args.train:
        # fit and test the model
        lstm_dym.fit(train_data)
    anomaly_list, score_list = lstm_dym.detect(test_data)
    # create anomaly score array for ploting and evaluation
    pred_label = np.zeros(len(test_label))
    score = np.zeros(len(test_label))
    for (l, r), score_ in zip(anomaly_list, score_list):
        pred_label[l:r] = 1
        score[l:r] = score_

    # calc metrics
    precision, recall, f1, tp, tn, fp, fn = point_metrics(pred_label,test_label)
    print('precision:{}, recall:{}, f1:{}, tp:{}, tn:{}, fp:{}, fn:{}'.format(precision, recall, f1, tp, tn, fp, fn))

    # adjust the pred_label for better visualization
    adjust_pred_label = adjust_predicts(pred_label,test_label,delay=200)
    precision, recall, f1, tp, tn, fp, fn = point_metrics(adjust_pred_label,test_label)
    print('precision:{}, recall:{}, f1:{}, tp:{}, tn:{}, fp:{}, fn:{}'.format(precision, recall, f1, tp, tn, fp, fn))

    return {
        'pred_label': pred_label,
        'adjust_pred_label': adjust_pred_label,
        'test_label': test_label,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LSTM dynamic.')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--l_s', type=int, default=150)
    parser.add_argument('--n_predictions', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--fraction', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lstm_batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--smoothing_perc', type=float, default=0.05)
    parser.add_argument('--error_buffer', type=int, default=30)
    parser.add_argument('--p', type=float, default=0.13)
    args = parser.parse_args(args=[])

    data_type = 'NAB'
    # data_type = 'NASA'
    # data_type = 'Yahoo'

    if data_type is 'NAB':
        temp = []
        dirname = 'realTweets'
        print(dirname)
        filenames = os.listdir(Path(cwd,f'examples/data/NAB_data/{dirname}'))
        for filename in filenames:
            # filename = 'nyc_taxi.csv'
            print(filename)
            args.filename = filename
            train_set, test_set = load_NAB(
                dirname, filename, fraction=args.fraction)
            model_path = Path(cwd, f'snapshot/lstm_dym/{dirname}',
                                args.filename[:-4])
            print(f'srcnn model path: {model_path}')
            result = main(args, train_set, test_set, model_path)
            temp.append((result['precision'],result['recall'],result['f1']))

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

            model_path = Path(cwd,f'snapshot/lstm_dym/{dirname}',
                                args.filename[:-4])
            print(f'srcnn model path: {model_path}')
            result = main(args, train_set, test_set, model_path)
            adjust_pred_label.append(result['adjust_pred_label'])
            test_label.append(result['test_label'])
        pred = np.concatenate(adjust_pred_label)
        label = np.concatenate(test_label)
        print (point_metrics(pred,label))

    elif data_type is 'NASA':
        adjust_pred_label, test_label = [], []
        NASA_chans = [
    'T-9', 'T-8', 'T-5', 'T-4', 'T-3', 'T-2', 'T-13', 'T-12', 'T-10', 'T-1',
    'S-2', 'S-1', 'R-1', 'P-7', 'P-4', 'P-3', 'P-2', 'P-15', 'P-14', 'P-11',
    'P-10', 'P-1', 'M-7', 'M-6', 'M-5', 'M-4', 'M-3', 'M-2', 'M-1', 'G-7',
    'G-6', 'G-4', 'G-3', 'G-2', 'G-1', 'F-8', 'F-7', 'F-5', 'F-4', 'F-3', 'F-2',
    'F-1', 'E-9', 'E-8', 'E-7', 'E-6', 'E-5', 'E-4', 'E-3', 'E-2', 'E-13',
    'E-12', 'E-11', 'E-10', 'E-1', 'D-9', 'D-8', 'D-7', 'D-6', 'D-5', 'D-4',
    'D-3', 'D-2', 'D-16', 'D-15', 'D-14', 'D-13', 'D-12', 'D-11', 'D-1', 'C-2',
    'C-1', 'B-1', 'A-9', 'A-8', 'A-7', 'A-6', 'A-5', 'A-4', 'A-3', 'A-2', 'A-1'
]

        for chan_id in NASA_chans[:1]:
            # chan_id = 'A-4'
            print(chan_id)
            model_path = Path(cwd, f'snapshot/lstm_dym/NASA/{chan_id}')
            train_set, test_set = load_split_NASA(chan_id)
            result = main(args, train_set, test_set, model_path)
            adjust_pred_label.append(result['adjust_pred_label'])
            test_label.append(result['test_label'])
        pred = np.concatenate(adjust_pred_label)
        label = np.concatenate(test_label)
        print (point_metrics(pred,label))



