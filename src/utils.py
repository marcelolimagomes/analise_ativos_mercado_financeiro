import os
import pandas as pd
from pycaret.regression.oop import RegressionExperiment
import plotly.express as px
import gc

all_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore', 'symbol']

data_numeric_fields = ['open', 'high', 'low', 'volume', 'close']

date_features = ['open_time']

use_cols = date_features + data_numeric_fields

numeric_features = data_numeric_fields + ['rsi']


def increment_time(interval='1h'):
    match(interval):
        case '1min':
            return pd.Timedelta(minutes=1)
        case '3min':
            return pd.Timedelta(minutes=3)
        case '5min':
            return pd.Timedelta(minutes=5)
        case '15min':
            return pd.Timedelta(minutes=15)
        case '30min':
            return pd.Timedelta(minutes=30)
        case '1h':
            return pd.Timedelta(hours=1)
        case '2h':
            return pd.Timedelta(hours=2)
        case '4h':
            return pd.Timedelta(hours=4)
        case '6h':
            return pd.Timedelta(hours=6)
        case '8h':
            return pd.Timedelta(hours=8)
        case '12h':
            return pd.Timedelta(hours=12)
        case '1d':
            return pd.Timedelta(days=1)
        case '3d':
            return pd.Timedelta(days=3)
        case '1w':
            return pd.Timedelta(weeks=1)
        case '1M':
            return pd.Timedelta(days=30)


def date_parser(x):
    return pd.to_datetime(x, unit='ms')


def read_data(dir, sep=';', all_cols=None, use_cols=use_cols) -> pd.DataFrame:
    filenames = []

    for file in os.listdir(dir):
        if file.endswith(".csv"):
            filenames.append(os.path.join(dir, file))

    parse_dates = ['open_time']
    dataframes = []

    for filename in filenames:
        print('Start reading file: ', filename)
        df = pd.read_csv(filename, names=all_cols, parse_dates=parse_dates,
                         date_parser=date_parser, sep=sep, decimal='.', usecols=use_cols)
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.sort_values(['open_time'], inplace=True)
    combined_df.reset_index(inplace=True, drop=True)
    return combined_df


def rotate_label(df, rows_to_rotate=-1, label='label_shifted', dropna=False):
    new_label = label + '_' + str(rows_to_rotate)
    df[new_label] = df[label].shift(rows_to_rotate)
    if dropna:
        df.dropna(inplace=True)

    return new_label, df


def setup_model(
        data: pd.DataFrame,
        label: str,
        train_size=0.7,
        numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
        date_features=['open_time'],
        use_gpu=False,
        regressor_estimator='lr',
        apply_best_analisys=False,
        fold=3,
        sort='MAE',
        verbose=False) -> [RegressionExperiment, any]:

    re = RegressionExperiment()

    setup = re.setup(data,
                     train_size=train_size,
                     target=label,
                     numeric_features=numeric_features,
                     date_features=date_features,
                     create_date_columns=["hour", "day", "month"],
                     fold_strategy='timeseries',
                     fold=fold,
                     session_id=123,
                     normalize=True,
                     use_gpu=use_gpu,
                     verbose=verbose,
                     )
    best = regressor_estimator
    if apply_best_analisys:
        print('Applying best analisys...') if verbose else None
        best = setup.compare_models(sort=sort, verbose=True, exclude=['lightgbm'])

    print(f'Creating model Best: [{best}]') if verbose else None
    model = setup.create_model(best, verbose=False)
    model_name_file = str(model)[0:str(model).find('(')] + '_' + label
    print(f'Saving model {model_name_file}') if verbose else None
    setup.save_model(model, model_name_file)

    return setup, model


def predict(setup: RegressionExperiment,
            model: any,
            predict_data: pd.DataFrame = None,
            numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
            date_features=['open_time'],
            verbose=False) -> RegressionExperiment:

    print('predict.setup: \n', setup) if verbose else None
    print('predict.model: \n', model) if verbose else None
    print('predict.predict_data: \n', predict_data) if verbose else None
    print('predict.numeric_features: \n', numeric_features) if verbose else None
    print('predict.date_features: \n', date_features) if verbose else None

    predict = None
    if predict_data is None:
        predict = setup.predict_model(model, verbose=verbose)
    else:
        predict = setup.predict_model(model, data=predict_data[date_features + numeric_features], verbose=verbose)

    return predict


def forecast(data: pd.DataFrame,
             fh: int = 1,
             train_size=0.7,
             interval='1h',
             numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
             date_features=['open_time'],
             regressor_estimator='lr',
             apply_best_analisys=False,
             use_gpu=False,
             fold=3,
             ):
    list_models = {}

    _data = data.copy()
    test_data = data.tail(1).copy().reset_index(drop=True)
    print('numeric_features: ', numeric_features)

    open_time = test_data['open_time']
    df_result = pd.DataFrame()
    for i in range(1, fh + 1):
        df_predict = pd.DataFrame()
        open_time = open_time + increment_time(interval)
        df_predict['open_time'] = open_time
        print(f'Applying predict No: {i} for open_time: {df_predict["open_time"].values}')

        for label in numeric_features:
            if label not in list_models:
                print('Training model for label:', label)
                target, train_data = rotate_label(_data, -1, label, True)
                setup, model = setup_model(train_data, target, train_size=train_size, fold=fold,
                                           regressor_estimator=regressor_estimator, use_gpu=use_gpu, apply_best_analisys=apply_best_analisys)
                train_data.drop(columns=target, inplace=True)
                list_models[label] = {'setup': setup, 'model': model}
                print('Training model Done!')

            _setup = list_models[label]['setup']
            _model = list_models[label]['model']

            df = predict(_setup,
                         _model,
                         test_data if i == 1 else df_result.tail(1).copy(),
                         numeric_features,
                         date_features)

            print('Label:', label, 'Predict Label:', df['prediction_label'].values[0])
            df_predict[label] = df['prediction_label']
            gc.collect()

        df_result = pd.concat([df_result, df_predict], axis=0)
        gc.collect()

    return df_result.sort_values('open_time').reset_index(drop=True)


def shift_test_data(predict_data: pd.DataFrame, label='close', columns=[], verbose=False):
    print('Shifting: \n', predict_data.tail(1)[columns]) if verbose else None
    _test_data = predict_data[columns].tail(1).copy().shift(1, axis='columns')
    _test_data.drop(columns=label, inplace=True)
    _test_data['open_time'] = predict_data['open_time']
    print('Shifted: \n', _test_data.tail(1)) if verbose else None
    return _test_data


def forecast2(data: pd.DataFrame,
              label: str = 'close',
              fh: int = 1,
              train_size=0.7,
              interval='1h',
              numeric_features=['open', 'high', 'low', 'volume', 'close', 'rsi'],
              date_features=['open_time'],
              regressor_estimator='lr',
              apply_best_analisys=False,
              use_gpu=False,
              fold=3,
              regression_times=1,
              sort='MAE',
              verbose=False,
              ):

    _data = data.copy()
    for i in range(1, regression_times + 1):
        _label, _data = rotate_label(_data, i, label)
        numeric_features.append(_label)
    _data.dropna(inplace=True)

    print('numeric_features: ', numeric_features) if verbose else None

    open_time = data.tail(1)['open_time']
    df_result = pd.DataFrame()
    setup = None
    model = None
    for i in range(1, fh + 1):
        if model is None:
            print('Training model for label:', label) if verbose else None
            setup, model = setup_model(_data, label, train_size, numeric_features, date_features,
                                       use_gpu, regressor_estimator, apply_best_analisys, fold, sort, verbose)
            print('Training model Done!') if verbose else None

        open_time = open_time + increment_time(interval)
        print(f'Applying predict No: {i} for open_time: {open_time}') if verbose else None
        predict_data = shift_test_data(_data.tail(1).copy() if i == 1 else df_result.tail(1).copy(), label=label, columns=[label] + numeric_features)
        predict_data['open_time'] = open_time

        df_predict = predict(setup, model, predict_data, numeric_features, date_features, verbose)
        df_predict['close'] = df_predict['prediction_label']

        gc.collect()

        df_result = pd.concat([df_result, df_predict], axis=0)
        gc.collect()

    return df_result.sort_values('open_time').reset_index(drop=True), model, setup


def calc_diff(predict_data, validation_data, regressor):
    start_date = predict_data["open_time"].min()  # strftime("%Y-%m-%d")
    end_date = predict_data["open_time"].max()  # .strftime("%Y-%m-%d")
    # now = datetime.now().strftime("%Y-%m-%d")

    predict_data.index = predict_data['open_time']
    validation_data.index = validation_data['open_time']

    filtered_data = validation_data.loc[(validation_data['open_time'] >= start_date) & (validation_data['open_time'] <= end_date)].copy()
    filtered_data['prediction_label'] = predict_data['prediction_label']
    filtered_data['diff'] = ((filtered_data['close'] - filtered_data['prediction_label']) / filtered_data['close']) * 100
    filtered_data.drop(columns=['open_time'], inplace=True)
    filtered_data.round(2)
    return filtered_data


def plot_predic_model(predict_data, validation_data, regressor):
    start_date = predict_data["open_time"].min()  # strftime("%Y-%m-%d")
    end_date = predict_data["open_time"].max()  # .strftime("%Y-%m-%d")

    filtered_data = calc_diff(predict_data, validation_data, regressor)

    fig1 = px.line(filtered_data, x=filtered_data.index, y=['close', 'prediction_label'], template='plotly_dark', range_x=[start_date, end_date])
    fig2 = px.line(filtered_data, x=filtered_data.index, y=['diff'], template='plotly_dark', range_x=[start_date, end_date])
    fig1.show()
    fig2.show()
    return filtered_data
