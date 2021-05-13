import pandas as pd
import numpy as np
from pprint import pprint


def load_data(fileName):
    data_series = pd.read_csv(fileName).values.tolist()
    temp_list = []
    mass_list = []

    for data_row in data_series:
        temp_list.append(data_row[7])
        mass_list.append(data_row[8])

    temp_datas = np.array(temp_list)
    mass_datas = np.array(mass_list)

    return [temp_datas, mass_datas]


if __name__ == "__main__":
    csv_file = "TY_climate_2017_2018.csv"
    x, y = load_data(csv_file)
