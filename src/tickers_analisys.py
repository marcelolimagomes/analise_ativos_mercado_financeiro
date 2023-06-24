import yfinance as yf
import pandas as pd
import datetime
import pytz
import src.calcEMA as calc


def process():
    symbols = get_tickers()
    data = download_data('2013-01-01', symbols)
    new_data = convert_downloaded_data(data)

    print('Start calc RSI and EMAs...')
    new_data = calc.run_calc_emas(new_data, 'adj_close')

    new_data.to_json(
        './src/data/ibov.json',
        orient='records',
        date_unit='s',
        date_format='epoch')
    # print(new_data.head())


def get_tickers() -> list:
    tickers = pd.read_csv('./src/data/tickers_list_to_analisys.csv')
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

    new_df.rename(columns={'Adj Close': 'adj_close', 'Close': 'close', 'High': 'high',
                  'Low': 'low', 'Open': 'open', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    new_df.index.name = 'date'
    new_df['date_time'] = pd.to_datetime(new_df.index)
    new_df['s_datetime'] = new_df.index.strftime('%Y%m%d%H%M')
    new_df['date_import'] = datetime.datetime.now(tz=pytz.UTC)
    # print('new_df>> \n', new_df)
    return new_df


def download_data(start_date='', tickers=[]) -> pd.DataFrame:
    if start_date == '':
        year = datetime.datetime.today().year
        start_date = str(year) + '-01-01'

    print('Downloading data start_date: ' + start_date)
    print('Symbols: ', tickers)
    data = yf.download(tickers, start=start_date,
                       threads=20, group_by='ticker')
    return data
