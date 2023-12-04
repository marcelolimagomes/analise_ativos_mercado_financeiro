import sys
sys.path.insert(0, sys.path[0].removesuffix('/src/crypto'))
sys.path.insert(0, sys.path[0].removesuffix('/src'))
print(sys.path)

from src.utils import *
from src.calcEMA import calc_RSI
from pycaret.regression import *
# Variables
datadir = './data'
test_dir = './test'
label = 'close'
regression_times = 24 * 14
days_to_forecasting = 7
use_cols = ['open_time', 'close']


def read_train_data():
    data = read_data(datadir, use_cols=use_cols)
    # data = calc_RSI(data)
    # data.dropna(inplace=True)
    print(data.info())
    print(data.tail(1))
    return data


def read_validation_data():
    validation_data = read_data(test_dir, use_cols=use_cols)
    validation_data.dropna(inplace=True)
    print(validation_data.info())
    print(validation_data.tail(1))
    return validation_data


def predict_model(data, validation_data, features):
    regressor_list = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr',
                      'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 'catboost']

    # regressor_list = ['ard', 'ransac', 'tr', 'kr', 'svm', 'mlp', 'lightgbm']

    diff = {}
    for regs in regressor_list:
        print('Calculating: ', regs)
        predict_data, _ = forecast2(data=data.copy(), fh=24 * days_to_forecasting, label=label,
                                    numeric_features=features.copy(), regression_times=regression_times, regressor_estimator=regs)
        diff[regs] = calc_diff(predict_data, validation_data, regs)['diff'].std()
        print(regs, ': - Desvio padrão: ', diff[regs])

    df_result = pd.DataFrame.from_dict(diff, orient='index', columns=['diff']).sort_values('diff')
    df_result.to_csv('diff.csv')
    print(df_result)


def main():
    print(use_cols)
    data = read_train_data()
    validation_data = read_validation_data()

    features = data.columns.drop(['open_time', 'close']).tolist()
    print(features)

    predict_model(data, validation_data, features)


if __name__ == '__main__':
    main()
