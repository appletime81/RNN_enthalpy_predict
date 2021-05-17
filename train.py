from training_model import *
from load_data import *
from pprint import pprint
from tensorflow.keras.callbacks import TensorBoard
import os


def build_name():
    model_paths = os.listdir("saved_models")
    if len(model_paths) == 0:
        return "LSTM_001.h5"
    else:
        model_paths = sorted(model_paths, key=lambda x: int(x[-6:-3]))
        model_name = str(int(model_paths[len(model_paths) - 1][-6:-3]) + 1)
        if len(model_name) == 1:
            return "LSTM_00" + model_name + ".h5"
        elif len(model_name) == 2:
            return "LSTM_0" + model_name + ".h5"
        else:
            return "LSTM_" + model_name + ".h5"


def main():
    csv_file = "TY_climate_2017_2018.csv"
    tensorboard_call_back = TensorBoard(log_dir="./log", histogram_freq=1, write_grads=True)

    train_data, test_data, column_name = load_data(csv_file)  # column_name: TT-Avg(℃), MT-Avg(g)
    # train_data, test_data, _ = data_preprocessing(train_data, test_data)
    train_data, _ = data_preprocessing(train_data)
    test_data, _ = data_preprocessing(test_data)

    # load data
    x_train, y_train = create_dataset(train_data)
    # x_test, y_test = create_dataset(test_data)

    x_train = x_train.reshape(x_train.shape[0], 1, 1)
    # x_test = x_test.reshape(x_test.shape[0], 1, 1)

    # reshape data
    y_train = y_train.reshape(y_train.shape[0], 1, 1)
    # y_test = y_test.reshape(y_test.shape[0], 1, 1)

    # load model
    lstm_model = training_model()
    print(lstm_model.summary())

    # start training
    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[tensorboard_call_back])

    # save model
    if column_name == "TT-Avg(℃)":
        lstm_model.save(f"saved_models_tt_avg/{build_name()}")
    elif column_name == "MT-Avg(g)":
        lstm_model.save(f"saved_models_mt_avg/{build_name()}")


if __name__ == "__main__":
    main()



