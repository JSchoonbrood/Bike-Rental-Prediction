"""Contains function to convert dataframe from series based to supervised
"""

import pandas as pd

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