import pandas as pd

from main import *


def plot_fig(x_test, y_test, model, df, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(df, color='red', label="True Value")
    ax.plot(
        range(len(y_train) + 1, len(y_train) + 1 + len(predictions)),
        predictions,
        color="blue",
        label="Predicted Testing Value"
    )
    plt.legend()
    plt.show()

    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test_scaled, color='red', label='True Testing Value')
    plt.plot(predictions, color='blue', label='Predicted Testing Value')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    csv_file = "TY_climate_2017_2018.csv"

    df = pd.read_csv(csv_file)
    df = df["TT-Avg(�J)"].values
    df = df.reshape(-1, 1)

    train_data, test_data = load_data(csv_file)
    train_data, test_data, scaler = data_preprocessing(train_data, test_data)

    # load data
    x_train, y_train = create_dataset(train_data)
    x_test, y_test = create_dataset(test_data)

    x_train = x_train.reshape(x_train.shape[0], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], 1, 1)

    # reshape data
    y_train = y_train.reshape(y_train.shape[0], 1, 1)
    y_test = y_test.reshape(y_test.shape[0], 1, 1)

    # load model
    lstm_model = load_model("LSTM_001.h5")

    plot_fig(
        x_test=x_test,
        y_test=y_test,
        model=lstm_model,
        df=np.array(df),
        scaler=scaler
    )
