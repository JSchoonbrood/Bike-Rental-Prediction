import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pathlib import Path

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
	print (modified_dataframe)
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

	reframed = series_to_supervised(scaled, 168, 1) #first 1: consider 1hr before. #second 1: how many predictions in future
	print (reframed.head())

	initial_number = 168
	for i in range(len(reframed)):
		modified_number = initial_number-1
		variable_name = "var1(t-" + modified_number + ")"
		reframed.drop(variable_name)

run()