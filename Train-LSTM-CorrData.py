import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pathlib import Path
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import seaborn as sn

def processData(dataframe):
	modified_dataframe = dataframe.copy()
	modified_dataframe.columns = ['Date', 'Rented', 'Hour', 'Temp', 'Humidity',
								  'WindSpeed', 'Visbility', 'DewPointTemp',
								  'SolarRad', 'Rainfall', 'Snowfall',
								  'Seasons', 'Holiday', 'FuncDay']

	date = modified_dataframe.pop('Date')
	hour = modified_dataframe.pop('Hour')
	encoded_dates = []
	index = 0
	for row in date:
		date_split = row.split('/')
		new_date = ' '.join((date_split[2], date_split[1], date_split[0], str(hour[index])))
		strp_date = datetime.strptime(new_date, '%Y %m %d %H')
		encoded_dates.append(str(strp_date))
		index += 1

	modified_dataframe.insert(0, 'Time', encoded_dates)
	modified_dataframe.set_index('Time', inplace=True)

	seasons = modified_dataframe.pop('Seasons')
	integer_seasons = []
	season_type = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Autumn': 4}
	for row in seasons:
		new_season = season_type[str(row)]
		integer_seasons.append(new_season)
	modified_dataframe.insert(9, 'Seasons', integer_seasons)

	holiday = modified_dataframe.pop('Holiday')
	integer_holiday = []
	holiday_type = {'Holiday': 1, 'No Holiday': 2}
	for row in holiday:
		new_holiday = holiday_type[str(row)]
		integer_holiday.append(new_holiday)
	modified_dataframe.insert(10, 'Holidays', integer_holiday)

	func_day = modified_dataframe.pop('FuncDay')
	integer_func_day = []
	func_day_type = {'Yes': 1, 'No': 2}
	for row in func_day:
		new_day = func_day_type[str(row)]
		integer_func_day.append(new_day)
	modified_dataframe.insert(11, 'FuncDays', integer_func_day)
	modified_dataframe.pop('DewPointTemp')
	modified_dataframe.to_csv('modified.csv')

	df_corr = modified_dataframe.corr()

	#f = pyplot.figure(figsize=(19, 15))
	#pyplot.matshow(df_corr, fignum=f.number)
	#pyplot.xticks(range(modified_dataframe.select_dtypes(['number']).shape[1]), modified_dataframe.select_dtypes(['number']).columns, fontsize=14, rotation=45)
	#pyplot.yticks(range(modified_dataframe.select_dtypes(['number']).shape[1]), modified_dataframe.select_dtypes(['number']).columns, fontsize=14)
	#cb = pyplot.colorbar()
	#cb.ax.tick_params(labelsize=14)
	#pyplot.title('Correlation Matrix', fontsize=16);
	#pyplot.show()

	return modified_dataframe

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def get_huber_loss_fn(**huber_loss_kwargs):
	def custom_huber_loss(y_true, y_pred):
		return tf.compat.v1.losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)
	return custom_huber_loss

def modified_sigmoid(x):
	return (K.sigmoid(x)-0.1)

class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss') <= 0.001):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")
            self.model.stop_training = True

class LearningRateReducerCb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.995
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

def run():
	current_dir = Path(os.path.dirname(__file__))
	dataset_path = os.path.join(current_dir, 'SeoulBikeData.csv')

	dataframe = pd.read_csv(dataset_path)
	print(dataframe)
	df = processData(dataframe)
	print(df)

	values = df.values

	scaler = MinMaxScaler(feature_range=(0, 1)) #normalizes values between 0 and 1
	scaled = scaler.fit_transform(values)

	reframed = series_to_supervised(scaled, 12, 1) #first 1: consider 1hr before. #second 1: how many predictions in future
	reframed.drop(reframed.columns[[-1,-2,-3,-4,-5,-6,-7, -8, -9, -10]], axis=1, inplace=True)
	print (reframed.head())
	reframed_values = reframed.values

	n_train_hours = 300*24

	train = reframed_values[:n_train_hours, :]
	test = reframed_values[n_train_hours:, :]

	train_x, train_y = train[:, :-1], train[:, -1]
	test_x, test_y = test[:, :-1], test[:, -1]

	train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
	test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

	print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
	get_custom_objects().update({'ModifiedSigmoid': Activation(modified_sigmoid)})
	trainingStopCallback = haltCallback()
	trainingStopCallback2 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min', baseline=None)
	mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
	# design network
	model = Sequential()
	model.add(LSTM(33, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
	model.add(LSTM(20, return_sequences=True))
	model.add(LSTM(10, return_sequences=True))
	model.add(LSTM(5, return_sequences=False))
	model.add(Dense(1))
	model.add(Activation(modified_sigmoid, name='ModifiedSigmoid'))
	model.compile(loss=get_huber_loss_fn(delta=0.1), optimizer='adam')

	history = model.fit(train_x, train_y, epochs=1000, batch_size=30, validation_data=(test_x, test_y), verbose=2, shuffle=False, callbacks=[LearningRateReducerCb(), trainingStopCallback2, mc])

	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()

	model.save('E:\Github\INNS-Assessment\Model')

	saved_model = load_model('best_model.h5', custom_objects={'custom_huber_loss':get_huber_loss_fn(delta=0.1), 'modified_sigmoid':Activation(modified_sigmoid)})

	# make a prediction
	yhat = saved_model.predict(test_x)
	test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))

	# invert scaling for forecast
	inv_yhat = concatenate((yhat, test_x[:, -10:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]

	# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = concatenate((test_y, test_x[:, -10:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]

	mse = mean_squared_error(inv_y, inv_yhat)
	mae = mean_absolute_error(inv_y, inv_yhat)
	rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

	pyplot.plot(inv_yhat, label='[Prediction]')
	pyplot.plot(inv_y, label='[Actual]')
	pyplot.legend()
	pyplot.show()

	rng = np.random.RandomState(0)
	colors = rng.rand(len(inv_yhat))
	pyplot.scatter(inv_yhat, inv_y, c=colors ,alpha = 1)
	pyplot.xlabel('Ground Truth')
	pyplot.ylabel('Predictions')
	pyplot.show();
	print('Test RMSE: %.3f' % rmse)
	print('Test MSE: %.3f' % mse)
	print('Test MAE: %.3f' % mae)

run()
