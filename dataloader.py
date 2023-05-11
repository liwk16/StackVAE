import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

def norm(train_data, test_data):
    mean_ = np.mean(train_data, axis=0)
    std_ = np.std(train_data, axis=0)
    train_data = (train_data - mean_) / (std_ + 1e-4)
    test_data = (test_data - mean_) / (std_ + 1e-4)
    
    return train_data, test_data


def load_NASA(filename):
    train_data = np.load(os.path.join('NASA', 'train', filename))
    test_data = np.load(os.path.join('NASA', 'test', filename))

    return train_data, test_data

def load_SMD(filename):
    prefix = 'smd-processed'
    train_path = os.path.join(prefix, filename + '_train.pkl')
    test_path = os.path.join(prefix, filename + '_test.pkl')
    test_label_path = os.path.join(prefix, filename + '_test_label.pkl')
    x_dim = 38

    f = open(train_path, "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))
    f.close()

    f = open(test_path, 'rb')
    test_data = pickle.load(f).reshape((-1, x_dim))
    f.close()

    f = open(test_label_path, 'rb')
    test_label = pickle.load(f).reshape((-1))
    f.close()

    def preprocess(df):
        df = np.asarray(df, dtype=np.float32)

        if len(df.shape) == 1:
            raise ValueError('Data must be a 2-D array')

        if np.any(sum(np.isnan(df)) != 0):
            print('Data contains null values. Will be replaced with 0')
            df = np.nan_to_num()

        # normalize data
        df = MinMaxScaler().fit_transform(df)
        # print('Data normalized')

        return df
    
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    return train_data, test_data, test_label


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
    parent_path = os.path.join(os.path.dirname(__file__), 'NAB_data')
    # print(os.path.dirname(__file__))
    path = os.path.join(parent_path, dirname, filename)
    data = pd.read_csv(path)
    print('load data from {:}'.format(os.path.abspath(path)))
    data.index = pd.to_datetime(data['timestamp'])
    pd_data = data[['value', 'label']].astype('float')
    train_data = pd_data.iloc[:int(len(pd_data) * fraction)]
    test_data = pd_data.iloc[int(len(pd_data) * fraction):]
    return train_data, test_data
