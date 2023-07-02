from src.calcEMA import *
import os
import pytz
import datetime
import pandas as pd
import yfinance as yf
import gc
import sys
if sys.path[0].endswith('/src'):
    sys.path.insert(0, sys.path[0].removesuffix('/src'))
print('Path:', sys.path)


def process(load_cache=True):
    data_file = sys.path[0] + '/src/data/ibov.csv'
    print('Carregando Dataset...')
    dataset = load_dataset(load_cache, data_file)
    print('Iniciando Calculo RSI e EMAs...')

    emas_dataset = pd.DataFrame()
    for symbol in get_tickers():
        emas_dataset = pd.concat([emas_dataset, run_calc_emas(
            dataset[dataset['symbol'] == symbol], 'adj_close')])
        gc.collect()
    print('Lista ordenada por *Ações com Desconto*:', emas_dataset.index.max())
    print_descontados(emas_dataset)

    print('Atualizando cache...')
    emas_dataset.sort_values(['date', 'symbol'], ascending=[True, True], inplace=True)
    emas_dataset.to_csv(data_file, sep=';')

    return emas_dataset


def load_dataset(load_cache=True, data_file='./src/data/ibov.csv') -> pd.DataFrame:
    symbols = get_tickers()
    dataset: pd.DataFrame = None
    if (load_cache and os.path.exists(data_file)):
        # dataset = pd.read_json(data_file, orient='records', date_unit='s')
        dataset = pd.read_csv(data_file, sep=';',
                              index_col='date', parse_dates=True)
        dataset['date_time'] = pd.to_datetime(dataset['date_time'])
        dataset['date_import'] = pd.to_datetime(dataset['date_import'])

        if datetime.datetime.now() > dataset.index.max():
            print('Atualizando dataset. Data atual:',
                  datetime.datetime.now().strftime('%Y-%m-%d'))
            print('Atualizando dataset. Última data:',
                  dataset.index.max().strftime('%Y-%m-%d'))
            data = download_data(dataset.index.max().strftime(
                '%Y-%m-%d'), list(dataset['symbol'].drop_duplicates()))
            dataset = pd.concat([dataset, convert_downloaded_data(data)])
            gc.collect()
            dataset.drop_duplicates(
                subset=['date_time', 'symbol'], keep='first', inplace=True)
    else:
        data = download_data('2013-01-01', symbols)
        dataset = convert_downloaded_data(data)
    # print(dataset.info())
    return dataset


def get_tickers() -> list:
    filename = sys.path[0] + '/src/data/tickers_list_to_analisys.csv'
    print('Tickers List File:', filename)
    tickers = pd.read_csv(filename)
    tickers['symbol'] += '.SA'
    return list(tickers['symbol'])


def get_symbols_from_tickers_history(tickers_history: pd.DataFrame) -> pd.DataFrame:
    symbols = []
    for symbol, _ in tickers_history.columns:
        symbols.append(symbol)
    symbols = list(set(symbols))
    return symbols


def convert_downloaded_data(tickers_history: pd.DataFrame) -> pd.DataFrame:
    symbols = get_symbols_from_tickers_history(tickers_history)

    new_df = pd.DataFrame()
    for s in symbols:
        aux = tickers_history[s].copy()
        aux['symbol'] = s
        new_df = pd.concat([new_df, aux], axis=0)
        gc.collect()
    new_df.dropna(how='any', axis=0, inplace=True)
    new_df.rename(columns={'Adj Close': 'adj_close', 'Close': 'close', 'High': 'high',
                  'Low': 'low', 'Open': 'open', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    new_df.index.name = 'date'
    new_df['date_time'] = pd.to_datetime(new_df.index)
    new_df['date_import'] = pd.to_datetime(datetime.datetime.now(tz=pytz.UTC))
    gc.collect()
    return new_df


def download_data(start_date='', tickers=[]) -> pd.DataFrame:
    if start_date == '':
        year = datetime.datetime.today().year
        start_date = str(year) + '-01-01'

    print('Baixando dados [start_date]: ' + start_date)
    print('Symbols: ', tickers)
    data = yf.download(tickers, start=start_date,
                       threads=20, group_by='ticker', ignore_tz=True)
    return data


def print_descontados(df: pd.DataFrame):
    filter = df[df.index == df.index.max()][['open', 'high', 'low', 'close',
                                             'adj_close', 'volume', 'symbol', 'ema_200p', 'ema_200p_diff', 'rsi']]
    print('>>> 50 Ações mais descontadas:')
    print(filter.sort_values(by='ema_200p_diff', ascending=True).head(50).round(2))
