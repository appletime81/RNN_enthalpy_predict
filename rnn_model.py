import numpy as np
import pandas as pd

from tensorflow import keras
from pprint import pprint


def generate_time_series(batch_size, n_steps, seed=10):
    np.random.seed(seed)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # + noise
    return series[..., np.newaxis].astype(np.float32)


def gen_data_series():
    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    X_train, Y_train = series[:7000, :n_steps], series[:7000, -1]
    X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
    X_test, Y_test = series[9000:, :n_steps], series[9000:, -1]
    return X_train, X_valid, series, n_steps


def rnn_model():
    n_steps = 50

    series = generate_time_series(10000, n_steps + 10)
    X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
    X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
    X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

    Y = np.empty((10000, n_steps, 10))  # each target is a sequence of 10D vectors
    for step_ahead in range(1, 10 + 1):
        Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
    Y_train = Y[:7000]
    Y_valid = Y[7000:9000]
    Y_test = Y[9000:]
    #


    model = keras.models.Sequential(
        [
            keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
            keras.layers.SimpleRNN(20, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(10))
        ]
    )

    def last_time_step_mse(Y_true, Y_pred):
        print(Y_true[:, -1].shape)
        return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])
    model.fit(X_train, Y_train, epochs=20, verbose=1)
    print(model.evaluate(X_valid, Y_valid))
    model.save_weights("final_result.h5")


if __name__ == "__main__":
    rnn_model()
