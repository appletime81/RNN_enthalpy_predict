from load_data import (
    load_data,
    data_preprocessing,
    create_dataset,
)
from plot import predict_func
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_original_data(csv_file_name, column_name):
    df = pd.read_csv(csv_file_name)
    sub_df = df[column_name].values.reshape(-1, 1)

    sub_data, scalar = data_preprocessing(sub_df)
    sub_data_x, sub_data_y = create_dataset(sub_data)
    sub_data_x = sub_data_x.reshape(sub_data_x.shape[0], 1, 1)

    return sub_data_x, sub_data_y, scalar


def generate_predict_value(data, model_name, scalar):
    pred_values = predict_func(data, model_name, scalar)
    return pred_values


def actual_minus_pred(y_actual, y_pred):
    return np.subtract(y_actual, y_pred)


def MSE(array1, array2):
    difference_array = np.subtract(array1, array2)
    squared_array = np.square(difference_array)
    return squared_array


def plot_mse(x, y, fig, ax, color, **label):  # colors --> List
    ax.plot(x=x, y=y, color=color, label="mse_curve")
    return fig, ax


def main():
    csv_file = "TY_climate_2017_2018.csv"
    column_name = "TT-Avg(℃)"

    x_tt, y_tt, x_tt_scalar = load_original_data(csv_file, column_name)
    y_tt = y_tt.reshape(-1, 1)
    pred_tt = generate_predict_value(x_tt, "saved_models_tt_avg/LSTM_002.h5", x_tt_scalar)

    x = actual_minus_pred(y_tt, pred_tt)
    y = MSE(y_tt, pred_tt)
    print(x.shape, y.shape)
    fig, ax = plt.subplots(figsize=(25, 16))
    ax.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
