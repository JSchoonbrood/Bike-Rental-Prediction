import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
import math
import statistics
import os, sys, csv

from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot

def visualiseModel(history):
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['mean_squared_error'], label='train')
    #pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    return

def processData(dataframe):
    data_to_process = dataframe.copy()
    date = data_to_process.pop('Date')
    seasons = data_to_process.pop('Seasons')
    holiday = data_to_process.pop('Holiday')
    func_day = data_to_process.pop('Functioning Day')

    new_dates = []
    for row in date:
        row = row.replace("/","-")
        new_dates.append(row)

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

    test = data_to_process.pop('Rented Bike Count')
    data_to_process.insert(0, 'Date', new_dates)
    data_to_process.insert(1, 'Seasons', new_seasons)
    data_to_process.insert(2, 'Holidays', new_holidays)
    data_to_process.insert(3, 'Functioning Day', new_func_days)
    data_to_process.insert(13, 'Rented Bike Count', test)

    data_to_process.columns = ['Date', 'Seasons', 'Holidays', 'FuncDays',
                                'Hour', 'Temp', 'WindSpeed', 'Humidity',
                                'Visibility', 'DewPoint', 'SolarRad',
                                'Rainfall', 'Snowfall', 'Rented']

    numeric_attr = ['Temp', 'WindSpeed', 'Humidity', 'Visibility', 'DewPoint',
                    'SolarRad', 'Rainfall', 'Snowfall', 'Rented']
    for col in numeric_attr:
        data_to_process[col] = pd.to_numeric(data_to_process[col], errors='coerce')


    categorical_attr = ['Seasons', 'Holidays', 'FuncDays', 'Hour']
    for col in categorical_attr:
        data_to_process[col] = data_to_process[col].astype("category")

    data_to_process['Date'] = pd.to_datetime(data_to_process['Date'], errors='coerce')

    train_attr = data_to_process[numeric_attr]

    train_attr = pd.get_dummies(data_to_process, columns=categorical_attr)
    train_attr.pop('Date')

    return train_attr

def run():
    # Setup Directories
    if sys.platform == "linux" or sys.platform == "linux2":
        current_dir = os.path.dirname(__file__)
        fname = '/SeoulBikeData.csv'
    elif sys.platform == "win32":
        current_dir = os.path.dirname(__file__)
        fname = '\SeoulBikeData.csv'

    # Load Dataframe
    df = pd.read_csv(current_dir + fname)

    new_df = processData(df)
    #for col in new_df:
    #    new_df[col] = pd.to_numeric(new_df[col],errors='coerce')

    train_ds, test_ds =  train_test_split(new_df, test_size=0.2)
    train_ds, val_ds = train_test_split(train_ds, test_size=0.2)
    #print (train_ds)

    train_labels = train_ds.pop('Rented')
    #train_mean = np.mean(train_ds)
    #train_std = np.std(train_ds)
    #normalised_ds = 0.5 * (np.tanh(0.01 * ((train_ds - train_mean) / train_std)) + 1)

    #print (normalised_ds)

    model = Sequential()
    model.add(Dense(32, input_dim=40))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    #opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.1)
    #opt = 'adam'
    #loss = tf.keras.losses.CategoricalHinge()from_logits=True)
    #loss = tf.keras.losses.MeanSquaredError(from_logits=True)
    #loss = "mean_squared_error"
    #model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    history = model.fit(train_ds, train_labels, epochs=30, batch_size=30)

    visualiseModel(history)


run()
