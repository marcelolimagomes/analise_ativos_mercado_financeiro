import sys
if sys.path[0].endswith('/src'):
    sys.path.insert(0, sys.path[0].removesuffix('/src'))
print('Path:', sys.path)

import yfinance as yf
import pandas as pd
import datetime
import pytz
import os

from src.calcEMA import *

def process(load_cache = True):
    data_file = sys.path[0] + '/src/data/ibov.json'
    print('Carregando Dataset...')
    dataset = load_dataset(load_cache, data_file)     
    print('Iniciando Calculo RSI e EMAs...')

    emas_dataset = pd.DataFrame()
    for symbol in get_tickers():
        emas_dataset = pd.concat([emas_dataset, run_calc_emas(dataset[dataset['symbol'] == symbol], 'adj_close')])
    
    print('Lista ordenada por *Ações com Desconto*:', emas_dataset.index.max())
    print_descontados(emas_dataset)

    if not os.path.exists(data_file):
        emas_dataset.to_json(
            data_file,
            orient='records',
            date_unit='s',
            date_format='epoch')

    return emas_dataset

def load_dataset(load_cache = True, data_file = './src/data/ibov.json') -> pd.DataFrame:
    symbols = get_tickers()    
    if ( load_cache and os.path.exists(data_file)):
        dataset = pd.read_json(data_file, orient='records', date_unit='s')
        dataset.index = pd.to_datetime(dataset['date_time'])
        dataset['date_import'] = pd.to_datetime(dataset['date_import'])
        dataset.index.name = 'date'
    else:
        data = download_data('2013-01-01', symbols)
        dataset = convert_downloaded_data(data)
    print(dataset.info())
    return dataset

def get_tickers() -> list:
    filename = sys.path[0] + '/src/data/tickers_list_to_analisys.csv'
    print('Tickers List File:', filename)
    tickers = pd.read_csv(filename)
    tickers['symbol'] += '.SA'
    return list(tickers['symbol'])


def convert_downloaded_data(tickers_history: pd.DataFrame) -> pd.DataFrame:
    symbols = []
    for symbol, _ in tickers_history.columns:
        symbols.append(symbol)
    symbols = list(set(symbols))

    new_df = pd.DataFrame()
    for s in symbols:
        aux = tickers_history[s].copy()
        aux['symbol'] = s
        new_df = pd.concat([new_df, aux], axis=0)
    new_df.dropna(how='any', axis=0, inplace=True)
    new_df.rename(columns={'Adj Close': 'adj_close', 'Close': 'close', 'High': 'high',
                  'Low': 'low', 'Open': 'open', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    new_df.index.name = 'date'
    new_df['date_time'] = pd.to_datetime(new_df.index)
    new_df['date_import'] = pd.to_datetime(datetime.datetime.now(tz=pytz.UTC))
    return new_df


def download_data(start_date='', tickers=[]) -> pd.DataFrame:
    if start_date == '':
        year = datetime.datetime.today().year
        start_date = str(year) + '-01-01'

    print('Baixando dados [start_date]: ' + start_date)
    print('Symbols: ', tickers)
    data = yf.download(tickers, start=start_date,
                       threads=20, group_by='ticker')
    return data

def print_descontados(df: pd.DataFrame):
    filter = df[df.index == df.index.max()]
    print('>>> 50 Ações mais descontadas:')
    print(filter.sort_values(by='ema_200p_diff', ascending=True).head(50).round(2))