"""Contains function to handle processing raw data.
"""

from datetime import datetime


def process_data(dataframe, drop_columns=[]):
    """Process data to a useable format.

    Args:
        dataframe (any): Pass in the dataframe
        drop_columns (list<str>): A list of strings of column names

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

    if len(drop_columns) > 0:
        for column in drop_columns:
            try:
                modified_dataframe.pop(column)
            except KeyError:
                print("An error has occured, column name",
                      column, "is not valid")

    modified_dataframe.to_csv('modified.csv')

    # df_corr = modified_dataframe.corr()
    #f = pyplot.figure(figsize=(19, 15))
    #pyplot.matshow(df_corr, fignum=f.number)
    #pyplot.xticks(range(modified_dataframe.select_dtypes(['number']).shape[1]), modified_dataframe.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    #pyplot.yticks(range(modified_dataframe.select_dtypes(['number']).shape[1]), modified_dataframe.select_dtypes(['number']).columns, fontsize=14)
    #cb = pyplot.colorbar()
    # cb.ax.tick_params(labelsize=14)
    #pyplot.title('Correlation Matrix', fontsize=16);
    # pyplot.show()

    return modified_dataframe
