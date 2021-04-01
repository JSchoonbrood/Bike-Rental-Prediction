import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys

from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from matplotlib import pyplot
from pathlib import Path

def processData(dataframe):
    data_to_process = dataframe.copy()
    date = data_to_process.pop('Date')
    seasons = data_to_process.pop('Seasons')
    holiday = data_to_process.pop('Holiday')
    func_day = data_to_process.pop('Functioning Day')

    days = []
    months = []
    years = []
    for row in date:
        date_split = row.split("/")
        days.append(int(date_split[0]))
        months.append(int(date_split[1]))
        years.append(int(date_split[2]))

    print(days)
    print(months)
    print(years)

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
    data_to_process.insert(0, 'Day', days)
    data_to_process.insert(1, 'Month', months)
    data_to_process.insert(2, 'Years', years)
    data_to_process.insert(3, 'Seasons', new_seasons)
    data_to_process.insert(4, 'Holidays', new_holidays)
    data_to_process.insert(5, 'Functioning Day', new_func_days)
    data_to_process.insert(15, 'Rented Bike Count', test)  # 13 if date incl

    # Rename columns
    data_to_process.columns =  ['Days', 'Months', 'Years', 'Seasons',
                                'Holidays', 'FuncDays', 'Hour', 'Temp',
                                'WindSpeed', 'Humidity', 'Visibility',
                                'DewPoint', 'SolarRad', 'Rainfall', 'Snowfall',
                                'Rented']

    # Set column types for numerical columns
    numeric_attr = ['Temp', 'WindSpeed', 'Humidity', 'Visibility', 'DewPoint',
                    'SolarRad', 'Rainfall', 'Snowfall', 'Rented', 'Days', 'Months', 'Years']

    for col in numeric_attr:
        data_to_process[col] = pd.to_numeric(data_to_process[col], errors='coerce')

    # Set column types for original string'd columns
    categorical_attr = ['Seasons', 'Holidays', 'FuncDays', 'Hour']
    for col in categorical_attr:
        data_to_process[col] = data_to_process[col].astype("category")

    return data_to_process

def visualiseModel(history, eval):
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(eval.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['mean_absolute_error'], label='train')
    # pyplot.plot(eval.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    return

def basicModel(train_features, train_labels, val_features, val_labels):
    model1 = tf.keras.Sequential()
    model1.add(Dense(5, input_dim=15))
    model1.add(BatchNormalization())

    model1.add(Dense(64,  activation='sigmoid'))
    model1.add(BatchNormalization())
    model1.add(Dropout(0.25))

    model1.add(Dense(64,  activation='sigmoid'))
    model1.add(BatchNormalization())
    model1.add(Dropout(0.25))

    model1.add(Dense(1))

    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    # loss = tf.keras.losses.CategoricalHinge()#from_logits=True)
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss = "mean_absolute_error"
    model1.compile(optimizer=opt, loss=loss, metrics=['mean_absolute_error'])

    history = model1.fit(train_features, train_labels, epochs=100,
                         batch_size=30)
    eval = model1.evaluate(val_features, val_labels)

    visualiseModel(history, eval)

def run():
    current_dir = Path(os.path.dirname(__file__))
    dataset_path = os.path.join(current_dir, 'SeoulBikeData.csv')

    dataframe = pd.read_csv(dataset_path)

    df = processData(dataframe)

    train_df, test_df = train_test_split(df, test_size = 0.1)
    train_df, val_df = train_test_split(train_df, test_size=0.1)

    train_features = train_df.copy()
    train_labels = train_features.pop("Rented")

    val_features = val_df.copy()
    val_labels = val_features.pop("Rented")

    test_features = test_df.copy()
    test_labels = test_features.pop("Rented")

    basicModel(train_features, train_labels, val_features, val_labels)

run()
