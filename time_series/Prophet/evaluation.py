from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
import time
from transform import Transform
from model import MyProphet
from metrics import MAPE
from utils import without_keys


class CrossValidation:
    def __init__(self, params, cv):
        self.params = params
        self.cv = cv

    def fit(self, df):
        params_transform = Transform.params_transform(self.params['name'], df, self.cv)
        transform, inv_transform = Transform.transform_by_name(self.params['name'])
        kwargs = without_keys(self.params, ['name'])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            results = []
            for fold in range(1, self.cv + 1):
                res = analysis(df, fold, transform, inv_transform,
                               params_transform, **kwargs)
                results.append(res)
            return results


class GridSearchCV:
    def __init__(self, params, cv=4):
        self.params = params
        self.cv = cv

    def fit(self, df):
        columns = list(np.append(list(self.params.keys()), ['params']))
        for fold in range(1, self.cv + 1):
            columns.append(f'fold{fold}')
        columns.append('mean')

        self.cv_results_ = pd.DataFrame(columns=columns)
        n = len(ParameterGrid(self.params))
        k = 0
        for params in ParameterGrid(self.params):
            start_time = time.time()
            res = CrossValidation(params, self.cv).fit(df)
            row = pd.DataFrame(columns=columns)
            for key, value in params.items():
                row.at[0, key] = value
            row.at[0, 'params'] = params
            for fold in range(1, self.cv + 1):
                row.at[0, f'fold{fold}'] = res[fold - 1]
            row.at[0, 'mean'] = np.mean(res)
            self.cv_results_ = self.cv_results_.append(row, ignore_index=True)
            k += 1
            print(f'{k} out of {n} done')
            print('Time: ', time.time() - start_time)


def analysis(df, fold, transform, inv_transform, params_transform, TEST_SIZE=70, plot=False, **kwargs):
    # kwargs:
    # changepoint_prior_scale
    # changepoint_range

    df = df.copy()
    df = df[:df.shape[0] - TEST_SIZE * (fold - 1)]
    TRAIN_SIZE = df.shape[0] - TEST_SIZE
    df_train = df[:TRAIN_SIZE]
    df_test = df[TRAIN_SIZE:]
    df_train.y = transform(df_train.y)

    model = MyProphet(**kwargs).fit(df_train)
    df_pred = model.predict(model.make_future_dataframe(periods=TEST_SIZE, freq='H'))
    df_pred.y = inv_transform(df_pred.y, **params_transform)
    df_train.y = inv_transform(df_train.y, **params_transform)

    if (plot):
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
        ax1.plot(df_train.index, df_train.y)
        ax1.plot(df_pred.index[TRAIN_SIZE:], df_pred.y[TRAIN_SIZE:])
        ax1.legend(['train', 'predict'])

        ax2.plot(df_train.index, df_train.y)
        ax2.plot(df_test.index, df_test.y)
        ax2.legend(['train', 'test'])

    return MAPE(df_test.y, df_pred.y[TRAIN_SIZE:])

