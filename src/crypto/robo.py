import sys
sys.path.insert(0, sys.path[0].removesuffix('/src/crypto'))
from pycaret.classification import ClassificationExperiment
from src.utils import *
from src.calcEMA import calc_RSI
from sklearn.model_selection import train_test_split
import pandas as pd
from binance.client import Client
import datetime


# Variables
datadir = './data'
label = 'status'
stop_loss = 2.0
regression_times = 24 * 30  # horas
# numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi']
numeric_features = ['close']
currency = 'USDT'


def load_model(model_name):
    return ClassificationExperiment.load(model_name)


def predict_model(tail=-1, numeric_features=['close'], regression_times=24 * 30):
    symbols = pd.read_csv('symbol_list.csv')
    for symbol in symbols['symbol']:
        df = get_data(symbol + currency, save_database=False, interval='1h', tail=tail)
        df = calc_RSI(df)
        df = regresstion_times(df, numeric_features, regression_times)


def regresstion_times(df_database, numeric_features=['close'], regression_times=24 * 30):
    new_numeric_features = []
    new_numeric_features.append(numeric_features)
    for nf in numeric_features:
        for i in range(1, regression_times + 1):
            col = nf + "_" + str(i)
            df_database[col] = df_database[nf].shift(i)
            new_numeric_features.append(col)

    df_database.dropna(inplace=True)
    df_database = df_database.round(2)
    return df_database, new_numeric_features


def get_max_date(df_database):
    max_date = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')
    if df_database is not None:
        max_date = pd.to_datetime(df_database['open_time'].max(), unit='ms')
    return max_date


def get_database(symbol, tail=-1):
    database_name = get_database_name(symbol)

    df_database = pd.DataFrame()
    if os.path.exists(database_name):
        df_database = pd.read_csv(database_name, sep=';')
    if tail > 0:
        df_database = df_database.tail(tail).copy()
    return df_database


def get_database_name(symbol):
    return datadir + '/' + symbol + '/' + symbol + '.csv'


def download_data(save_database=True):
    symbols = pd.read_csv('symbol_list.csv')
    for symbol in symbols['symbol']:
        get_data(symbol, save_database)


def get_klines(symbol, interval='1h', max_date='2010-01-01', limit=1000):
    client = Client()
    klines = client.get_historical_klines(symbol=symbol, interval=interval, start_str=max_date, limit=limit)
    df_klines = pd.DataFrame(data=klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    return df_klines


def get_data(symbol, save_database=True, interval='1h', tail=-1):
    database_name = get_database_name(symbol)
    df_database = get_database(symbol, tail)
    max_date = get_max_date(df_database)

    print('Downloading data for symbol: ' + symbol)
    # Interval for Kline data (e.g., '1h' for 1-hour candles)
    while (max_date < datetime.datetime.now()):
        # Fetch Kline data
        print('Max date: ', max_date)
        df_klines = get_klines(symbol, interval=interval, max_date=max_date.strftime('%Y-%m-%d'))
        df_database = pd.concat([df_klines, df_database])
        df_database = df_database.sort_values('open_time')
        df_database['symbol'] = symbol

        max_date = get_max_date(df_database)

        if save_database:
            if not os.path.exists(database_name):
                os.makedirs(database_name)
            df_database.to_csv(database_name, sep=';', index=False)
            print('Database updated at ', database_name)
    return df_database


if __name__ == '__main__':
    download_data()
