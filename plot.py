from load_data import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("seaborn")


def plot_all_data(input_data, predictions, fig, ax, colors, **label):  # colors --> List
    ax.plot(input_data, color=colors[0], label=label["ground_truth"])
    ax.plot(predictions, color=colors[1], label=label["predict_value"])
    return fig, ax


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
    all_data_tt_x, all_data_tt_y = create_dataset(all_data_tt)
    all_data_tt_x = all_data_tt_x.reshape(all_data_tt_x.shape[0], 1, 1)
    all_data_tt_y = all_data_tt_y.reshape(all_data_tt_y.shape[0], 1, 1)

    all_data_mt, scaler_all_data_mt = data_preprocessing(df_mt)
    all_data_mt_x, all_data_mt_y = create_dataset(all_data_mt)
    all_data_mt_x = all_data_mt_x.reshape(all_data_mt_x.shape[0], 1, 1)
    all_data_mt_y = all_data_mt_y.reshape(all_data_mt_y.shape[0], 1, 1)

    colors_tt = ["", ""]
    colors_mt = ["", ""]




if __name__ == "__main__":
    main()
