from load_data import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.style import use
use("seaborn")


def plot_all_data(input_data, predictions, fig, ax, colors, **label):  # colors --> List
    ax.plot(input_data, color=colors[0], label=label["ground_truth"])
    ax.plot(predictions, color=colors[1], label=label["predict_value"])
    return fig, ax


def predict_func(input_data, model_name, scaler):
    model = load_model(model_name)
    predictions = model.predict(input_data)
    predictions = scaler.inverse_transform(predictions)
    return predictions


def main():
    csv_file = "TY_climate_2017_2018.csv"
    column_name_tt_avg = "TT-Avg(℃)"  # column_name: TT-Avg(℃), MT-Avg(g)
    column_name_mt_avg = "MT-Avg(g)"

    # get ground truth
    df = pd.read_csv(csv_file)

    df_tt = df[column_name_tt_avg].values
    df_tt = df_tt.reshape(-1, 1)

    df_mt = df[column_name_mt_avg].values
    df_mt = df_mt.reshape(-1, 1)

    #  get all data
    all_data_tt, scaler_all_data_tt = data_preprocessing(df_tt)
    all_data_tt_x, _ = create_dataset(all_data_tt)
    all_data_tt_x = all_data_tt_x.reshape(all_data_tt_x.shape[0], 1, 1)

    all_data_mt, scaler_all_data_mt = data_preprocessing(df_mt)
    all_data_mt_x, _ = create_dataset(all_data_mt)
    all_data_mt_x = all_data_mt_x.reshape(all_data_mt_x.shape[0], 1, 1)

    colors_tt = ["tab:red", "tab:orange"]
    colors_mt = ["tab:blue", "tab:green"]

    # predict
    model_name_tt = "saved_models_tt_avg/LSTM_002.h5"
    model_name_mt = "saved_models_mt_avg/LSTM_002.h5"
    predictions_tt = predict_func(all_data_tt_x, model_name_tt, scaler_all_data_tt)
    predictions_mt = predict_func(all_data_mt_x, model_name_mt, scaler_all_data_mt)

    # plot
    labels_tt = {
        "ground_truth": "True TT-Avg(℃)",
        "predict_value": "Predicted TT-Avg(℃)"
    }
    labels_mt = {
        "ground_truth": "True MT-Avg(g)",
        "predict_value": "Predicted MT-Avg(g)"
    }
    fig, ax = plt.subplots(figsize=(16, 9))
    fig, ax = plot_all_data(df_tt, predictions_tt, fig, ax, colors_tt, **labels_tt)
    fig, ax = plot_all_data(df_mt, predictions_mt, fig, ax, colors_mt, **labels_mt)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
