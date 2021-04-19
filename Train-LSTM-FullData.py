import pandas as pd
import os

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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

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

	modified_dataframe.to_csv('modified.csv')
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

def run():
	current_dir = Path(os.path.dirname(__file__))
	dataset_path = os.path.join(current_dir, 'SeoulBikeData.csv')

	dataframe = pd.read_csv(dataset_path)

	df = processData(dataframe)
	print (df)

	values = df.values

	scaler = MinMaxScaler(feature_range=(0, 1)) #normalizes values between 0 and 1
	scaled = scaler.fit_transform(values)

	reframed = series_to_supervised(scaled, 48, 1) #first 1: consider 1hr before. #second 1: how many predictions in future
	reframed.drop(reframed.columns[[-1,-2,-3,-4,-5,-6,-7, -8, -9, -10, -11]], axis=1, inplace=True)
	print (reframed.head())
	reframed_values = reframed.values

	n_train_hours = 200*24

	train = reframed_values[:n_train_hours, :]
	test = reframed_values[n_train_hours:, :]

	train_x, train_y = train[:, :-1], train[:, -1]
	test_x, test_y = test[:, :-1], test[:, -1]

	train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
	test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

	print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

	# design network
	model = Sequential()
	model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
	model.add(LSTM(25))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	history = model.fit(train_x, train_y, epochs=150, batch_size=30, validation_data=(test_x, test_y), verbose=2, shuffle=False)
	# plot history
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()


	# make a prediction
	yhat = model.predict(test_x)
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
	# calculate RMSE
	rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
	print('Test RMSE: %.3f' % rmse)


run()
