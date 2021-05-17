import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(file_name):
    # load csv
    df = pd.read_csv(file_name)
    df = df["TT-Avg(℃)"]  # TT-Avg(�J), MT-Avg(g)

    # convert to numpy format
    data = np.array(df)

    # split train data and test data
    train_data = data[:int(len(data) * 0.9)].reshape(-1, 1)
    test_data = data[int(len(data) * 0.9) - 1:].reshape(-1, 1)
    return train_data, test_data


def data_preprocessing(data):
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data, scaler


def create_dataset(data):
    x = []
    y = []
    for i in range(1, data.shape[0]):
        x.append(data[i - 1:i, 0])
        y.append(data[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y
