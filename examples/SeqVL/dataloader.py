import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_yahoo(filename, window_size, window_count, split_rate, use_norm=False):
    df = pd.read_csv(filename)
    if use_norm:
        mean_ = df.value.mean()
        std_ = df.value.std()
        ts = df.value.apply(lambda x: (x - mean_) / std_).values
    else:
        ts = df.value.values
    
    # df.value.plot()
    # ts.plot()
    # plt.show()
    # plt.close()
    # print(len(df))
    samples = np.array([ts[i: i + window_size + 1] for i in range(ts.shape[0] - window_size)], dtype=np.float32)
    
    seg_seqs = np.array([samples[i: i + window_count] for i in range(0, samples.shape[0] - window_count  + 1, window_count)])
    data = seg_seqs[:, :, :-1]
    labels = seg_seqs[:, :, -1]

    print('data info:')
    split_pos = int(data.shape[0] * split_rate)
    train_data = data[:split_pos]
    train_label = labels[:split_pos]
    test_data = data[split_pos:]
    test_label = labels[split_pos:]
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)

    return train_data, train_label, test_data, test_label


class Data:
    def __init__(self):
        self.data = None
        self.raw_data = None
        self.length = None
        self.supervised_data = None

    @staticmethod
    def _read_data(path):
        """
        read csv file from path
        :param path: path of file
        :return: dataframe read from csv file
        """
        data = pd.read_csv(path)
        print(data.head(5)) if data.shape[0] >= 5 else print(data)
        return data

    def _check_dims(self):
        # check at least one data column
        if len(self.raw_data.columns) <= 1:
            raise ValueError('Column of dataframe less than two')
        # check column dim equal
        self.length = len(self.raw_data[self.raw_data.columns[0]])
        for index in range(1, len(self.raw_data.columns)):
            if self.length != len(self.raw_data[self.raw_data.columns[index]]):
                raise ValueError('Column dims does not equal')

    def load_data(self, path):
        """
        load csv data
        :param path: folder path
        :return: return value and time-stamps
        """
        self.raw_data = self._read_data(path)
        self.raw_data.set_index(self.raw_data.columns[0])
        self._check_dims()
        self.data = self.raw_data

    def normalize(self, normalize_type=None):
        """
        normalize data
        :param normalize_type: 'interval', 'ratio' or 'diff'
        :return: normalize raw_data and save in data
        """
        if normalize_type == 'interval' or normalize_type is None:
            self._normalize_interval()
        elif normalize_type == 'ratio':
            self._normalize_ratio()
        elif normalize_type == 'diff':
            self._normalize_diff()
        else:
            raise NameError('normalize_type not support')

    def _normalize_interval(self):
        """
        normalize rawdata to data
        :return: turn data
        """
        result = self.raw_data.copy()
        for feature_name in self.raw_data.columns[1:]:
            max_value = self.raw_data[feature_name].max()
            min_value = self.raw_data[feature_name].min()
            result[feature_name] = (self.raw_data[feature_name] - min_value) / (max_value - min_value)
        self.data = result

    def _normalize_ratio(self):
        """
        normalize rawdata to data. count ratio one by next.
        :return:
        """
        result = self.raw_data.copy()
        for feature_name in self.raw_data.columns[1:]:
            for index in range(1, self.length):
                result[feature_name][index] -= self.raw_data[feature_name][index - 1]
        self.data = result

    def _normalize_diff(self):
        """
        normalize rawdata to data. count diff one by next
        :return:
        """
        result = self.raw_data.copy()
        for feature_name in self.raw_data.columns[1:]:
            result[feature_name][0] = 0
            for index in range(1, self.length):
                result[feature_name][index] /= self.raw_data[feature_name][index - 1]
        self.data = result

    def data2supervised(self, infer_length, pred_length, column):
        '''
        :param infer_length:
        :param pred_length:
        :return:
        '''
        data = self.data[column].values
        total_length = infer_length + pred_length - 1
        d = []
        for i in range(self.length - total_length):
            d.append(data[i:i + total_length + 1].tolist())
        d = np.array(d)
        X = d[:, :infer_length]
        y = d[:, infer_length:]
        self.supervised_data = {'X':X, 'y':y}
        print(self.supervised_data)

    def data_iterator(self, batchsize):
        """
        return data slices as an iterator
        :param batchsize:
        :return:
        """
        pass

    def load_yahoo(self, path):
        """
        load yahoo csv data
        :param path: file path
        :data format: timestamp indexed series, columns: [value, is_anomaly]
        """
        self.raw_data = self._read_data(path)
        save_list = ['timestamp', 'timestamps', 'value',
                    'anomaly', 'is_anomaly']
        self.data = self.raw_data.copy()
        drop_list = list()
        for col in self.raw_data.columns:
            if col not in save_list:
                drop_list.append(col)
        self.data.drop(columns=drop_list, inplace=True)
        
        timestamp = 'timestamp' if 'timestamp' in self.data.columns else 'timestamps'
        self.data.set_index(timestamp, inplace=True)
    
    def data_to_seqvl_format(self, window_size, window_count, split_rate):
        '''
        format the series into SeqVL input shape
        data: [sample_count, window_count, window_size]
        label: [sample_count, window_count]
        label is the forecasting result of each window
        @return: data, labels for training
        @return: test_samples, test_anomaly for AD test
        '''
        ts = self.data.value.values
        samples = np.array([ts[i: i + window_size + 1] for i in range(ts.shape[0] - window_size)], dtype=np.float32)
        anom_col = 'anomaly' if 'anomaly' in self.data.columns else 'is_anomaly'
        anomaly = self.data[anom_col].values[window_size - 1: -1]

        assert(samples.shape[0] == anomaly.shape[0])

        train_count = int(samples.shape[0] * split_rate)
        train_samples = samples[:train_count]
        test_samples = samples[train_count:, :-1]
        test_anomaly = anomaly[train_count:]

        seg_seqs = np.array([train_samples[i: i + window_count] for i in range(0, train_samples.shape[0] - window_count  + 1, window_count)])
        data = seg_seqs[:, :, :-1]
        labels = seg_seqs[:, :, -1]

        return data, labels, test_samples, test_anomaly

if __name__ == '__main__':
    # train_data, train_label, test_data, test_label = load_yahoo('ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv', 30, 300, 0.5)
    # # # print("hello")
    # print(train_data.dtype)
    # print(train_label.dtype)

    df = pd.read_csv('ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv')
    print(df.columns)
    (df.value).plot()
    df.is_anomaly.plot()
    plt.show()
    plt.close()
