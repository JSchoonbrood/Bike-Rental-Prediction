from pathlib import Path
from math import sqrt

import os
import numpy as np
import pandas as pd

import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import get_custom_objects

from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from scripts.process_df import process_data
from scripts.supervised_df import series_to_supervised
from scripts.huberloss import get_huber_loss_fn
from scripts.sigmoidcurve import modified_sigmoid
from scripts.dynamic_lr import LearningRateReducerCb
from scripts.callback import HaltCallback

def run():
    """Run function, handles everything in correct order and executes training model.
    """
    current_dir = Path(os.path.dirname(__file__))
    dataset_path = os.path.join(current_dir, 'SeoulBikeData.csv')
    dataframe = pd.read_csv(dataset_path)
    processed_df = process_data(dataframe, ['DewPointTemp'])
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
    training_stop_callback = HaltCallback()
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
    model.compile(loss=get_huber_loss_fn(delta=0.1), optimizer='adam')

    # Track training history
    history = model.fit(train_x, train_y, epochs=1000, batch_size=30, validation_data=(
        test_x, test_y), verbose=2, shuffle=False, callbacks=[LearningRateReducerCb(), training_stop_callback2, mc])

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
    inv_yhat = np.concatenate((yhat, test_x[:, -10:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # Invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_x[:, -10:]), axis=1)
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
