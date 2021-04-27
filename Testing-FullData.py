import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
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
import visualkeras

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

def run():
	current_dir = Path(os.path.dirname(__file__))
	dataset_path = os.path.join(current_dir, 'modified.csv')

	df = pd.read_csv(dataset_path)
	df.set_index('Time', inplace=True)

	values = df.values

	scaler = MinMaxScaler(feature_range=(0, 1)) #normalizes values between 0 and 1
	scaled = scaler.fit_transform(values)

	reframed = series_to_supervised(scaled, 12, 1) #first 1: consider 1hr before. #second 1: how many predictions in future
	reframed.drop(reframed.columns[[-1,-2,-3,-4,-5,-6,-7, -8, -9, -10, -11]], axis=1, inplace=True)
	reframed_values = reframed.values

	n_train_hours = 292*24

	train = reframed_values[:n_train_hours, :]
	test = reframed_values[n_train_hours:, :]

	train_x, train_y = train[:, :-1], train[:, -1]
	test_x, test_y = test[:, :-1], test[:, -1]

	train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
	test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

	saved_model = load_model('162RMSE.h5', custom_objects={'custom_huber_loss':get_huber_loss_fn(delta=0.1), 'modified_sigmoid':Activation(modified_sigmoid)})

	visualkeras.graph_view(saved_model).show()

	# make a prediction
	yhat = saved_model.predict(test_x)
	test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))

	# invert scaling for forecast
	inv_yhat = concatenate((yhat, test_x[:, -11:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]

	# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = concatenate((test_y, test_x[:, -11:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]

	rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
	mse = mean_squared_error(inv_y, inv_yhat)
	mae = mean_absolute_error(inv_y, inv_yhat)

	pyplot.plot(inv_yhat, label='[Predicted Value]')
	pyplot.plot(inv_y, label='[Actual Value]')
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
