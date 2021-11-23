import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
# add RealSeries/realseries to sys path for import realseries
sys.path.append(str(Path.cwd()))
from realseries.models.srcnn import SR_CNN
from realseries.utils.evaluation import point_metrics, adjust_predicts
from realseries.utils.data import load_Yahoo, load_NAB, load_split_NASA
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
cwd = str(Path.cwd())


def run(
        train_set,
        test_set,
        model_path,
        window,
        lr=1e-5,
        seed=0,
        epochs=30,
        batch_size=64,
        dropout_rate=0.5,
        step=1,
        num=10,
        thres_percent=99.9,
        delay=200,
        train=False,
):
    # the last column is label; other columns are values
    train_data, train_label = train_set.iloc[:, 0], train_set.iloc[:, -1]
    test_data, test_label = test_set.iloc[:, 0], test_set.iloc[:, -1]

    #build srcnn model
    sr_cnn = SR_CNN(
        model_path,
        window=window,
        lr=lr,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout_rate)

    if train:
        # train the model
        sr_cnn.fit(train_data, step=step, num=num)

    # get anomaly score
    score = sr_cnn.detect(test_data, test_label, step=1)

    thres = np.percentile(score, thres_percent)
    pred_label = (score > thres)

    # calc metrics
    precision, recall, f1, tp, tn, fp, fn = point_metrics(
        pred_label, test_label)
    print('precision:{}, recall:{}, f1:{}, tp:{}, tn:{}, fp:{}, fn:{}'.format(
        precision, recall, f1, tp, tn, fp, fn))

    # adjust the pred_label for better visualization
    adjust_pred_label = adjust_predicts(pred_label, test_label, delay)

    precision, recall, f1, tp, tn, fp, fn = point_metrics(
        adjust_pred_label, test_label)
    print('precision:{}\nrecall:{}\nf1:{}, tp:{}, tn:{}, fp:{}, fn:{}'.format(
        precision, recall, f1, tp, tn, fp, fn))
    return {
        'pred_label': pred_label,
        'adjust_pred_label': adjust_pred_label,
        'test_label': test_label,
        'precision': precision,
        'recall': recall
    }


def main(args, train_set, test_set, model_path):
    result = run(
        train_set,
        test_set,
        model_path,
        args.window,
        args.lr,
        args.seed,
        args.epochs,
        args.batch_size,
        args.dropout_rate,
        args.step,
        args.num,
        args.thres_percent,
        args.delay,
        args.train,
    )

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SRCNN.')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--thres_percent', type=float, default=99.9)
    parser.add_argument('--fraction', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--delay', type=int, default=24)

    args = parser.parse_args(args=[])

    #     data_type = 'NAB'
    #     data_type = 'NASA'
    data_type = 'Yahoo'

    if data_type is 'NAB':
        temp = []
        dirnames = 'realtweets'
        for dirname in dirnames[:1]:
            # dirname = 'realKnownCause'
            print(dirname)
            filenames = os.listdir(
                Path(cwd, f'examples/data/NAB_data/{dirname}'))
            for filename in filenames:
                filename = 'nyc_taxi.csv'
                print(filename)
                args.filename = filename
                train_set, test_set = load_NAB(
                    dirname, filename, fraction=args.fraction)
                model_path = Path(cwd, f'snapshot/sr/{dirname}',
                                  args.filename[:-4])
                print(f'srcnn model path: {model_path}')
                result = main(args, train_set, test_set, model_path)
                temp.append(result)

    elif data_type is 'Yahoo':
        adjust_pred_label, test_label = [], []
        dirnames = ['A1Benchmark', 'A2Benchmark', 'A3Benchmark', 'A4Benchmark']
        dirname = 'A1Benchmark'
        print(dirname)
        filenames = os.listdir(
            Path(cwd, f'examples/data/Yahoo_data/{dirname}'))
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

            model_path = Path(cwd, f'snapshot/sr/{dirname}',
                                args.filename[:-4])
            print(f'srcnn model path: {model_path}')
            result = main(args, train_set, test_set, model_path)
            adjust_pred_label.append(result['adjust_pred_label'])
            test_label.append(result['test_label'])
        pred = np.concatenate(adjust_pred_label)
        label = np.concatenate(test_label)
        print (point_metrics(pred,label))
