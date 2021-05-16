import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from pprint import pprint
from tensorflow.keras.models import (
    Sequential,
    load_model
)
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)


def load_data(fileName):
    # load csv
    df = pd.read_csv(fileName)
    df = df["TT-Avg(�J)"]  # TT-Avg(�J), MT-Avg(g)

    # convert to numpy format
    data = np.array(df)

    # split train data and test data
    train_data = data[:int(len(data)*0.9)].reshape(-1, 1)
    test_data = data[int(len(data)*0.9)-1:].reshape(-1, 1)
    return train_data, test_data


def data_preprocessing(train_data, test_data):
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)
    return train_data, test_data, scaler


def create_dataset(data):
    x = []
    y = []
    for i in range(1, data.shape[0]):
        x.append(data[i-1:i, 0])
        y.append(data[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


def training_model():
    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(1, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model


if __name__ == "__main__":
    csv_file = "TY_climate_2017_2018.csv"

    train_data, test_data = load_data(csv_file)
    train_data, test_data, _ = data_preprocessing(train_data, test_data)

    # load data
    x_train, y_train = create_dataset(train_data)
    x_test, y_test = create_dataset(test_data)

    x_train = x_train.reshape(x_train.shape[0], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], 1, 1)

    # reshape data
    print(y_train.shape)
    y_train = y_train.reshape(y_train.shape[0], 1, 1)
    y_test = y_test.reshape(y_test.shape[0], 1, 1)

    # load model
    lstm_model = training_model()
    print(lstm_model.summary())

    # start training
    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(x_train, y_train, epochs=50, batch_size=32)

    # save model
    lstm_model.save("LSTM_001.h5")
