from datetime import datetime
from pathlib import Path
from math import sqrt

import os
import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from keras import backend as K
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import get_custom_objects


from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def process_data(dataframe):
    """Process data to a useable format.

    Args:
        dataframe (any): Pass in the dataframe

    Returns:
        any: Processed dataframe
    """
    modified_dataframe = dataframe.copy()
    modified_dataframe.columns = ['Date', 'Rented', 'Hour', 'Temp', 'Humidity',
                                  'WindSpeed', 'Visbility', 'DewPointTemp',
                                  'SolarRad', 'Rainfall', 'Snowfall',
                                  'Seasons', 'Holiday', 'FuncDay']

    date = modified_dataframe.pop('Date')
    hour = modified_dataframe.pop('Hour')
    encoded_dates = []
    index = 0

    # Normalise datetime into useable format
    for row in date:
        date_split = row.split('/')
        new_date = ' '.join(
            (date_split[2], date_split[1], date_split[0], str(hour[index])))
        strp_date = datetime.strptime(new_date, '%Y %m %d %H')
        encoded_dates.append(str(strp_date))
        index += 1

    modified_dataframe.insert(0, 'Time', encoded_dates)
    modified_dataframe.set_index('Time', inplace=True)

    seasons = modified_dataframe.pop('Seasons')
    holiday = modified_dataframe.pop('Holiday')
    func_day = modified_dataframe.pop('FuncDay')

    # Encode seasons (str) to integers
    integer_seasons = []
    season_type = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Autumn': 4}
    for row in seasons:
        new_season = season_type[str(row)]
        integer_seasons.append(new_season)
    modified_dataframe.insert(9, 'Seasons', integer_seasons)

    # Encode holiday (str) to integers
    integer_holiday = []
    holiday_type = {'Holiday': 1, 'No Holiday': 2}
    for row in holiday:
        new_holiday = holiday_type[str(row)]
        integer_holiday.append(new_holiday)
    modified_dataframe.insert(10, 'Holidays', integer_holiday)

    # Encode func_day (str) to integers
    integer_func_day = []
    func_day_type = {'Yes': 1, 'No': 2}
    for row in func_day:
        new_day = func_day_type[str(row)]
        integer_func_day.append(new_day)
    modified_dataframe.insert(11, 'FuncDays', integer_func_day)

    modified_dataframe.to_csv('modified.csv')

    return modified_dataframe


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Convert timeseries dataframe to supervised
    Args:
        data (any): Processed dataframe
        n_in (int, optional): How many hours before the current to use for prediction. Defaults to 1.
        n_out (int, optional): How many future predictions. Defaults to 1.
        dropnan (bool, optional): Drops NaN columns. Defaults to True.

    Returns:
        any: Return dataframe
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # Concatenate data
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_huber_loss_fn(**huber_loss_kwargs):
    """A function to return huberloss
    """
    def custom_huber_loss(y_true, y_pred):
        """Returns tensorflows huberloss with parameters already passed

        Args:
            y_true (_type_): y true values
            y_pred (_type_): y prediction values

        Returns:
            any: Huberloss function
        """
        return tf.compat.v1.losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)
    return custom_huber_loss


def modified_sigmoid(x):
    """Returns a modified sigmoid curve

    Args:
        x (any): Tensor or variable taken from model.

    Returns:
        any: A tensor
    """
    return (K.sigmoid(x)-0.1)


class HaltCallback(tf.keras.callbacks.Callback):
    """A class to automatically halt training when val_loss has reached a specific criteria.

    Args:
        tf (_type_): Inherits tensorflow callback
    """

    def on_epoch_end(self, logs={}):
        """Halts training when condition met.

        Args:
            epoch: self, inherit's class attributes.
            logs (dict, optional): Log data from training. Defaults to {}.
        """
        if (logs.get('val_loss') <= 0.001):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True


def run():
    """Run function, handles everything in correct order and executes training model.
    """
    current_dir = Path(os.path.dirname(__file__))
    dataset_path = os.path.join(current_dir, 'SeoulBikeData.csv')
    dataframe = pd.read_csv(dataset_path)
    processed_df = process_data(dataframe)
    values = processed_df.values

    # Normalizes values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # Convert series data to supervised
    reframed = series_to_supervised(scaled, 12, 1)
    reframed.drop(reframed.columns[[-1, -2, -3, -4, -
                  5, -6, -7, -8, -9, -10, -11]], axis=1, inplace=True)
    reframed_values = reframed.values

    # 80% Training, 20% Testing split
    n_train_hours = 292*24
    train = reframed_values[:n_train_hours, :]
    test = reframed_values[n_train_hours:, :]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    
    # Define modified sigmoid
    get_custom_objects().update(
        {'ModifiedSigmoid': Activation(modified_sigmoid)})
    
    # Add training stops
    # training_stop_callback = HaltCallback()
    training_stop_callback2 = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min', baseline=None)
    
    # Add model checkpoint for best model based on lowest val_loss
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss',
                         mode='min', save_best_only=True)
    
    # Design LSTM Network
    model = Sequential()
    model.add(LSTM(33, input_shape=(
        train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(5, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation(modified_sigmoid, name='ModifiedSigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=get_huber_loss_fn(delta=0.1), optimizer=opt)

    # Track training history
    history = model.fit(train_x, train_y, epochs=1000, batch_size=30, validation_data=(
        test_x, test_y), verbose=2, shuffle=False, callbacks=[training_stop_callback2, mc])

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    model.save(current_dir)

    # Reload model for testing
    saved_model = load_model('best_model.h5', custom_objects={'custom_huber_loss': get_huber_loss_fn(
        delta=0.1), 'modified_sigmoid': Activation(modified_sigmoid)})

    # Make a prediction
    yhat = saved_model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))

    # Invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_x[:, -11:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # Invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_x[:, -11:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    
    # Calculate errors
    mse = mean_squared_error(inv_y, inv_yhat)
    mae = mean_absolute_error(inv_y, inv_yhat)
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    # Plot Predictions against Actual
    pyplot.plot(inv_yhat, label='[Prediction]')
    pyplot.plot(inv_y, label='[Actual]')
    pyplot.legend()
    pyplot.show()
    
    rng = np.random.RandomState(0)
    colors = rng.rand(len(inv_yhat))
    pyplot.scatter(inv_yhat, inv_y, c=colors, alpha=1)
    pyplot.xlabel('Ground Truth')
    pyplot.ylabel('Predictions')
    pyplot.show()
    
    # Display Errors
    print('Test RMSE: %.3f' % rmse)
    print('Test MSE: %.3f' % mse)
    print('Test MAE: %.3f' % mae)

run()
