from load_data import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def plot_predict_test_data(ground_truth_data, input_data, predictions):
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(ground_truth_data, color="#D2875C", label="True Value")
    ax.plot(
        range(len(input_data) + 1, len(input_data) + 1 + len(predictions)),
        predictions,
        color="#5372AB",
        label="Predicted Testing Value"
    )
    plt.legend()
    plt.show()


def plot_all_data(input_data, predictions):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(input_data, color="#D2875C", label="True Testing Value")
    plt.plot(predictions, color="#5372AB", label="Predicted Testing Value")
    plt.legend()
    plt.show()


def predict_func(input_data, model_name, scaler):
    model = load_model(model_name)
    predictions = model.predict(input_data)
    predictions = scaler.inverse_transform(predictions)
    return predictions


if __name__ == "__main__":
    csv_file = "TY_climate_2017_2018.csv"

    # load date
    train_data, test_data, column_name = load_data(csv_file)
    train_data, scaler_train = data_preprocessing(train_data)
    test_data, scaler_test = data_preprocessing(test_data)

    # get ground truth
    df = pd.read_csv(csv_file)
    df = df[column_name].values
    df = df.reshape(-1, 1)

    # create data
    x_train, y_train = create_dataset(train_data)
    x_test, y_test = create_dataset(test_data)

    # reshape
    x_train = x_train.reshape(x_train.shape[0], 1, 1)
    y_train = y_train.reshape(y_train.shape[0], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], 1, 1)
    y_test = y_test.reshape(y_test.shape[0], 1, 1)


    # predict all data
    all_data, scaler_all_data = data_preprocessing(df)
    all_data_x, all_data_y = create_dataset(all_data)
    all_data_x = all_data_x.reshape(all_data_x.shape[0], 1, 1)
    all_data_y = all_data_y.reshape(all_data_y.shape[0], 1, 1)

    # predict
    model_name = "saved_models_mt_avg/LSTM_001.h5"
    predictions = predict_func(x_test, model_name, scaler_test)

    # plot
    plot_predict_test_data(df, y_train, predictions)
    # plot_all_data(df, predictions)
