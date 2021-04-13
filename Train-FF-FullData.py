import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
from pandas import Series
import keras
from datetime import datetime
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from matplotlib import pyplot
from pathlib import Path

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
import numpy
from numpy import concatenate

def processData(dataframe):
    data_to_process = dataframe.copy()
    #date = data_to_process.pop('Date')
    seasons = data_to_process.pop('Seasons')
    holiday = data_to_process.pop('Holiday')
    func_day = data_to_process.pop('Functioning Day')

    new_seasons = []
    season_type = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Autumn': 4}
    for row in seasons:
        new_season = season_type[str(row)]
        new_seasons.append(new_season)

    new_holidays = []
    holiday_type = {'Holiday': 1, 'No Holiday': 2}
    for row in holiday:
        new_holiday = holiday_type[str(row)]
        new_holidays.append(new_holiday)

    new_func_days = []
    func_day_type = {'Yes': 1, 'No': 2}
    for row in func_day:
        new_day = func_day_type[str(row)]
        new_func_days.append(new_day)

    # Insertion of attributes into original dataset
    test = data_to_process.pop('Rented Bike Count')
    #data_to_process.insert(0, 'Day', days)
    #data_to_process.insert(1, 'Month', months)
    #data_to_process.insert(2, 'Years', years)
    data_to_process.insert(3, 'Seasons', new_seasons)
    data_to_process.insert(4, 'Holidays', new_holidays)
    data_to_process.insert(5, 'Functioning Day', new_func_days)
    data_to_process.insert(12, 'Rented Bike Count', test)  # 13 if date incl

    # Rename columns
    data_to_process.columns =  ['Seasons',
                                'Holidays', 'FuncDays', 'Hour', 'Temp',
                                'WindSpeed', 'Humidity', 'Visibility',
                                'DewPoint', 'SolarRad', 'Rainfall', 'Snowfall',
                                'Rented']

    # Set column types for numerical columns
    numeric_attr = ['Temp', 'WindSpeed', 'Humidity', 'Visibility', 'DewPoint',
                    'SolarRad', 'Rainfall', 'Snowfall', 'Rented']

    for col in numeric_attr:
        data_to_process[col] = pd.to_numeric(data_to_process[col], errors='coerce')

    # Set column types for original string'd columns
    categorical_attr = ['Seasons', 'Holidays', 'FuncDays', 'Hour']
    for col in categorical_attr:
        data_to_process[col] = data_to_process[col].astype("category")

    return data_to_process
