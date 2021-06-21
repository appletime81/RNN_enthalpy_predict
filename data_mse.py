from load_data import *
from plot import (
    predict_func,
    enthalpy
)


def mse(array1, array2):
    difference_array = np.subtract(array1, array2)
    squared_array = np.square(difference_array)
    return squared_array.mean()


def main():
    # file name and column name
    csv_file = "TY_climate_2017_2018.csv"
    column_name_tt_avg = "TT-Avg(℃)"  # column_name: TT-Avg(℃), MT-Avg(g)
    column_name_mt_avg = "MT-Avg(g)"
    column_name_enthalpy = "焓值計算(kj/kg)"

    df = pd.read_csv(csv_file)
    df_tt = df[column_name_tt_avg].values
    df_tt = df_tt.reshape(-1, 1)

    df_mt = df[column_name_mt_avg].values
    df_mt = df_mt.reshape(-1, 1)

    df_enthalpy = df[column_name_enthalpy].values
    df_enthalpy = df_enthalpy.reshape(-1, 1)

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
    predictions_enthalpy = enthalpy(predictions_tt, predictions_mt)
    print(df_enthalpy[1:].shape)

    # MSE
    enthalpy_mse = mse(df_enthalpy[1:], predictions_enthalpy)
    print(f"Result: {enthalpy_mse}")


if __name__ == "__main__":
    main()
