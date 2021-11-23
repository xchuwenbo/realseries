# -*- coding: utf-8 -*-
"""The function for load and process data.
"""
# Author: Wenbo Hu <i@wbhu.net>


import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy import random
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

MAX_INT = np.iinfo(np.int32).max

def _series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """ 
    reading data and pro-processing, get training data, validation data and test data for model.

    Args:
        data: Sequence of observations as a list or 2D NumPy array. Required.
        n_in (int, optional): Number of lag observations as input (X). Values may be between [1..len(data)].  Defaults to 1.
        n_out (int, optional): Number of observations as output (y). Values may be between [0..len(data)-1]. Defaults to 1.
        dropnan (bool, optional): Boolean whether or not to drop rows with NaN values. Defaults to True
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def generate_arma_data(
        n=1000,
        ar=None,
        ma=None,
        contamination_rate=0.05,
        contamination_variance=20,
        random_seed=None, ):
    """Generate synthetic data. Utility function for generate synthetic data for
    time series data

        - raw data for forecasting.
        - with contamination for anomaly detection.

    Generate using linear method.

    Args:
        n (int, optional): The length of training time series to generate.
            Defaults to 1000.
        ar (float array, optional): Parameter of AR model. Defaults to None.
        ma (float array, optional): Parameter of MA model. Defaults to None.
        contamination_rate (float, optional): The amount of contamination of the
            dataset in (0., 0.1). Defaults to 0.05.
        contamination_variance (float, optional): Variance of contamination.
            Defaults to 20.
        random_seed (int, optional):  Specify a random seed if need. Defaults to
            None.
    """
    if random_seed:
        random.seed(random_seed)
    if ar is None:
        ar = np.array([0.75, -0.25])
    if ma is None:
        ma = np.array([0.65, 0.35])
    x = sm.tsa.arma_generate_sample(ar, ma, n)
    contamination_index = random.choice(n, int(n * contamination_rate))
    x[contamination_index] += random.normal(0, contamination_variance,
                                            len(contamination_index))
    return x, contamination_index


def load_NAB(dirname='realKnownCause', filename='nyc_taxi.csv', fraction=0.5):
    """Load data from NAB dataset.

    Args:
        dirname (str, optional): Dirname in ``examples/data/NAB_data`` .
            Defaults to 'realKnownCause'.
        filename (str, optional): The name of csv file. Defaults to 'nyc_taxi.csv'.
        fraction (float, optional): The amount of data used for test set.
            Defaults to 0.5.

    Returns:
        (DataFrame, DataFrame): The pd.DataFrame instance of train and test set.
    """
    parent_path = os.path.join(os.path.dirname(__file__), '../../examples/data/NAB_data')
    path = os.path.join(parent_path, dirname, filename)
    data = pd.read_csv(path)
    print('load data from {:}'.format(os.path.abspath(path)))
    data.index = pd.to_datetime(data['timestamp'])
    pd_data = data[['value', 'label']].astype('float')
    train_data = pd_data.iloc[:int(len(pd_data) * fraction)]
    test_data = pd_data.iloc[int(len(pd_data) * fraction):]
    return train_data, test_data


def load_splitted_RNN(dirname='power_demand', filename='power_data.csv'):
    """Load data from RNN dataset.

    Args:
        dirname (str, optional): Dirname in ``examples/data/RNN_data`` .
            Defaults to 'power_demand'.
        filename (str, optional): The name of csv file. Defaults to
            'power_data.csv'.

    Returns:
        (DataFrame, DataFrame): The pd.DataFrame instance of train and test set.
    """
    parent_path = os.path.join(os.path.dirname(__file__), '../../examples/data/RNN_data')
    path = os.path.join(parent_path, dirname, 'train', filename)
    print('load data from {:}'.format(os.path.abspath(path)))
    train_data = pd.read_csv(path, index_col=0)
    path = path.replace('train', 'test')
    print('load data from {:}'.format(os.path.abspath(path)))
    test_data = pd.read_csv(path, index_col=0)
    return train_data, test_data


def load_Yahoo(dirname='A1Benchmark', filename='real_1.csv', fraction=0.5, use_norm=False, detail=True):
    """Load Yahoo dataset.

    Args:
        dirname (str, optional): Directory name. Defaults to 'A1Benchmark'.
        filename (str, optional): File name. Defaults to 'real_1.csv'.
        fraction (float, optional): Data split rate. Defaults to 0.5.
        use_norm (bool, optional): Whether to use data normalize.

    Returns:
        pd.DataFrame: train and test DataFrame.
    """
    yahoo_root_dir = os.path.join(os.path.dirname(__file__), '../../examples/data/Yahoo_data')
    path = os.path.join(yahoo_root_dir, dirname, filename)
    if detail:
        print('load data from {:}'.format(os.path.abspath(path)))
        print('split_rate: {}'.format(fraction))
    df = pd.read_csv(path)
    res = pd.DataFrame()
    res['timestamp'] = df['timestamp'] if 'timestamp' in df.columns else df['timestamps']
    res['value'] = df['value']
    res['anomaly'] = df['anomaly'] if 'anomaly' in df.columns else df['is_anomaly']
    train_data = res[:int(len(res) * fraction)]
    test_data = res[int(len(res) * fraction):]

    if use_norm:
        mean_ = train_data['value'].mean()
        std_ = train_data['value'].std()
        train_data['value'] = train_data['value'].apply(lambda x: (x - mean_) / std_)
        test_data['value'] = test_data['value'].apply(lambda x: (x - mean_) / std_)

    return train_data, test_data


# def load_pickle(path):
#     with open(path, 'rb') as f:
#         array = pickle.load(f)
#     data = np.array(array, dtype='float')
#     columns = ['value_{:}'.format(i) for i in range(data.shape[1] - 1)]
#     pd_data = pd.DataFrame(data, columns=columns + ['label'])
#     pd_data.index = pd.to_datetime(pd_data.index, unit='m')
#     return pd_data


def load_SMD(data_name='machine-1-1'):
    """Load SMD dataset.

    Args:
        data_name (str, optional): The filename of txt. Defaults to 'machine-1-1'.

    Returns:
        pd.DataFrame: Train_data, test_data and test_label
    """
    train_file = os.path.join('~/al/data/ServerMachineDataset', 'train',
                              data_name + '.txt')
    train_data = pd.read_csv(train_file, header=None)

    test_file = train_file.replace('train', 'test')
    test_data = pd.read_csv(test_file, header=None)

    test_label_file = train_file.replace('train', 'test_label')
    test_label = pd.read_csv(test_label_file, header=None)
    return train_data, test_data, test_label


# def load_NASA(data_name='MSL'):
#     """Load NASA data for lstm dynamic method.

#     Args:
#         data_name (str, optional): [description]. Defaults to 'MSL'.

#     Returns:
#         [type]: [description]
#     """
#     train_file = os.path.join('~/al/data/NASA',
#                               '{:s}_train.csv'.format(data_name))
#     train_data = pd.read_csv(train_file, index_col=0)

#     test_file = train_file.replace('train', 'test')
#     test_data = pd.read_csv(test_file, index_col=0)

#     test_label_file = train_file.replace('train', 'test_label')
#     test_label = pd.read_csv(test_label_file, index_col=0)
#     return train_data, test_data, test_label


def load_split_NASA(chan_id='T-9'):
    """Load NASA data for lstm dynamic method.

    Args:
        chan_id (str, optional): The name of file. Defaults to 'T-9'.

    Returns:
        pd.DataFrame: A tuple contains train_set and test_set.
    """
    '''
    0 row in T-10
    2 rows in P-2
    chans = ['T-9', 'T-8', 'T-5', 'T-4', 'T-3', 'T-2', 'T-13', 'T-12',
    'T-10', 'T-1', 'S-2', 'S-1', 'R-1', 'P-7', 'P-4', 'P-3', 'P-2',
     'P-15', 'P-14', 'P-11', 'P-10', 'P-1', 'M-7', 'M-6', 'M-5',
     'M-4', 'M-3', 'M-2', 'M-1', 'G-7', 'G-6', 'G-4', 'G-3', 'G-2',
     'G-1', 'F-8', 'F-7', 'F-5', 'F-4', 'F-3', 'F-2', 'F-1', 'E-9',
     'E-8', 'E-7', 'E-6', 'E-5', 'E-4', 'E-3', 'E-2', 'E-13', 'E-12',
     'E-11', 'E-10', 'E-1', 'D-9', 'D-8', 'D-7', 'D-6', 'D-5', 'D-4',
      'D-3', 'D-2', 'D-16', 'D-15', 'D-14', 'D-13', 'D-12', 'D-11',
      'D-1', 'C-2', 'C-1', 'B-1', 'A-9', 'A-8', 'A-7', 'A-6', 'A-5',
      'A-4', 'A-3', 'A-2', 'A-1']
    '''
    parent_path = os.path.join(os.path.dirname(__file__), '../../examples/data/NASA')
    train = np.load(os.path.join(parent_path, "train", chan_id + '.npy'))
    test = np.load(os.path.join(parent_path, "test", chan_id + '.npy'))
    label_file = pd.read_csv(os.path.join(parent_path, 'labeled_anomalies.csv'))
    index = label_file['chan_id'] == chan_id
    if np.sum(index) == 1:
        anomalies = eval(label_file[index]['anomaly_sequences'].to_list()[0])
        label = np.zeros(label_file['num_values'][index].values)
        for anomaly in anomalies:
            label[anomaly[0]:anomaly[1] + 1] = 1
    elif np.sum(index) == 0:
        label = np.zeros(len(test))
        print(np.sum(index), chan_id)
    elif np.sum(index) == 2:
        label = np.zeros(label_file['num_values'][index].iloc[0])
        anomalies = eval(label_file[index]['anomaly_sequences'].iloc[0])
        for anomaly in anomalies:
            label[anomaly[0]:anomaly[1] + 1] = 1

        anomalies = eval(label_file[index]['anomaly_sequences'].iloc[1])
        for anomaly in anomalies:
            label[anomaly[0]:anomaly[1] + 1] = 1
        print(np.sum(index), chan_id)

    # change the shape to train=(time*feature) and test=(time*1)
    train_set = np.concatenate([train, np.zeros(train[:, :1].shape)], axis=1)
    test_set = np.concatenate([test, label[:, np.newaxis]], axis=1)
    return pd.DataFrame(train_set), pd.DataFrame(test_set)


def load_exp_data(dataname='pm25', window_szie=15, prediction_window_size=1, fractions=[0.6,0.2,0.2], isshuffle=True, isscaler=True):
    """ 
    reading data and pro-processing, get training data, validation data and test data for model.

    Args:
        dataname (str, optional): the name of dataset, eg: 'pm25', 'bike_sharing', 'air_quality', 'metro_traffic'.
        window_size (int, optional): Number of lag observations as input. Defaults to 15.
        prediction_window_size (int, optional): Prediction window size. Defaults to 10.
        fractions: (list, optional): the training data, test data and validation data ratio, Defaults to [0.6,0.2,0.2].
        is_shuffle (bool, optional): whether to shuffle the raw data. Defaults to True.
        is_scaler: (bool, optional): whether to scale the raw data. Defaults to True.
    Returns:
        a splitted dataset(NumPy array): train_data, train_label, test_data, test_label, validation_data, validation_label
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../../examples/exp_data/')
    data_path = data_dir + dataname + '.csv'
    data = pd.read_csv(data_path)
    print('load data from {:}'.format(data_path))
    data = pd.DataFrame(data)
    raw_values = data.values
    features = raw_values.shape[-1]
    raw_values = raw_values.astype('float32')
    reframed = _series_to_supervised(raw_values, window_szie, prediction_window_size)
    # drop extra features at the forecast sequence
    for i in range(prediction_window_size):
        reframed.drop(reframed.columns[-(features+i):-(i+1)], axis=1, inplace=True)
    
    if isshuffle:
        reframed = shuffle(reframed, random_state=2)
    scaler = MinMaxScaler(feature_range=(0, 1))
    if isscaler:
        reframed = scaler.fit_transform(reframed.values)
    else:
        reframed = reframed.values
    train_num = int(reframed.shape[0] * fractions[0])
    test_num = int(reframed.shape[0] * fractions[1])
    train_values = reframed[:train_num, :]
    test_values = reframed[train_num:(train_num+test_num), :]
    val_values = reframed[(train_num+test_num):, :]
    train_data, train_label = train_values[:, :-prediction_window_size], train_values[:, -prediction_window_size:]
    test_data, test_label = test_values[:, :-prediction_window_size], test_values[:, -prediction_window_size:]
    val_data, val_label = val_values[:, :-prediction_window_size], val_values[:, -prediction_window_size:]
    # reconstruct the data-set into the data format that meets pytorch LSTM: [data_length,time-steps,features]
    train_data = train_data.reshape((train_data.shape[0], window_szie, -1))
    test_data = test_data.reshape((test_data.shape[0], window_szie, -1))
    val_data = val_data.reshape((val_data.shape[0], window_szie,-1))
    return train_data, train_label, test_data, test_label, val_data, val_label, scaler
    ''' print(os.path.split(os.path.realpath(__file__))[0])
    a,b,c,d,e,f,sc = load_exp_data('pm25',15,1,[0.6,0.2,0.2],True,True)
    print(e.shape,f.shape,type(a)) '''
