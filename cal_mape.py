from plot import *


def mape(array1, array2):
    difference_array = np.subtract(array1, array2)
    difference_array = np.absolute(difference_array)
    difference_array = difference_array / array1
    return difference_array.mean()


def load_data():
    raw_data_file = "TY_climate_2017_2018.csv"
    column_name_tt_avg = "TT-Avg(℃)"  # column_name: TT-Avg(℃), MT-Avg(g)
    column_name_mt_avg = "MT-Avg(g)"
    column_name_8ch = "8CH"

    df = pd.read_csv(raw_data_file)

    df_tt = df[column_name_tt_avg].values
    df_tt = df_tt.reshape(-1, 1)

    df_mt = df[column_name_mt_avg].values
    df_mt = df_mt.reshape(-1, 1)

    df_8ch = df[column_name_8ch]
    list_8ch = []
    for i in range(len(df_8ch)):
        try:
            list_8ch.append(int(df_8ch[i]))
        except ValueError:
            list_8ch.append(int(df_8ch[i].replace(",", "")))
    df_8ch = np.array(list_8ch).reshape(-1, 1).astype("float32")

    #  get all data
    all_data_tt, scaler_all_data_tt = data_preprocessing(df_tt)
    all_data_tt_x, _ = create_dataset(all_data_tt)
    all_data_tt_x = all_data_tt_x.reshape(all_data_tt_x.shape[0], 1, 1)

    all_data_mt, scaler_all_data_mt = data_preprocessing(df_mt)
    all_data_mt_x, _ = create_dataset(all_data_mt)
    all_data_mt_x = all_data_mt_x.reshape(all_data_mt_x.shape[0], 1, 1)

    # predict
    model_name_tt = "saved_models_tt_avg/LSTM_002.h5"
    model_name_mt = "saved_models_mt_avg/LSTM_002.h5"
    predictions_tt = predict_func(all_data_tt_x, model_name_tt, scaler_all_data_tt)
    predictions_mt = predict_func(all_data_mt_x, model_name_mt, scaler_all_data_mt)

    # cal enthalpy
    return enthalpy(predictions_tt, predictions_mt), df_8ch[1:]


def power_forecast(file):
    pred_enth, df_8ch = load_data()
    scale = pd.read_csv(file)["scale"].values
    scale = scale[1:366].reshape(-1, 1)
    pred_power = pred_enth * scale
    mape_value = mape(pred_power, df_8ch)
    print(f"mape value: {mape_value*100}%")


if __name__ == "__main__":
    file_name = "scale.csv"
    power_forecast(file_name)
