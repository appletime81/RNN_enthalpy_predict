import sys
import numpy as np
import pandas as pd
from pprint import pprint
np.set_printoptions(threshold=sys.maxsize)


def generate_time_series(batch_size, n_steps, seed=10):
    np.random.seed(seed)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # + noise
    return series[..., np.newaxis].astype(np.float32)


def data():
    n_steps = 50

    series = generate_time_series(10000, n_steps + 10)
    print(series.shape)
    X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
    X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
    X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

    print("X_train.shape:", X_train.shape)
    print("Y_train.shape:", Y_train.shape)

    print("X_valid.shape:", X_valid.shape)
    print("Y_valid.shape:", Y_valid.shape)

    print("X_test.shape:", X_test.shape)
    print("Y_test.shape:", Y_test.shape)
    pprint(Y_train[0])

    pprint(X_train[:, -1].shape)

    Y = np.empty((10000, n_steps, 10))  # each target is a sequence of 10D vectors
    for step_ahead in range(1, 10 + 1):
        Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
    Y_train = Y[:7000]
    Y_valid = Y[7000:9000]
    Y_test = Y[9000:]



if __name__ == "__main__":
    data()
