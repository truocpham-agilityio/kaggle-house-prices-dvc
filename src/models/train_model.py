# -*- coding: utf-8 -*-
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor

from src.data import load_data, load_params
from src.models.metrics import rmse


def train_model(train_path, results_dir, model_dir):
    """Train model and predict survival on Kaggle test set"""

    results_dir = Path(results_dir).resolve()
    assert (os.path.isdir(results_dir)), NotADirectoryError

    model_dir = Path(model_dir).resolve()
    assert (os.path.isdir(model_dir)), NotADirectoryError

    # load data
    train_df = load_data(train_path, sep=',', header=0, index_col='Id')

    # load params
    params = load_params()
    target_class = params['target_class']
    model_params = params['model_params']

    target = np.log1p(train_df.pop(target_class))

    # loop through model_params keys as single regressor
    for regressor in model_params.keys():
        if regressor.lower() == 'xgb_regressor':
            scaler = make_pipeline(RobustScaler(), XGBRegressor(**model_params[regressor]))
        elif regressor.lower() == 'svr':
            scaler = make_pipeline(RobustScaler(), SVR(**model_params[regressor]))
        elif regressor.lower() == 'lasso':
            scaler = make_pipeline(RobustScaler(), Lasso(**model_params[regressor]))
        elif regressor.lower() == 'kernel_ridge':
            scaler = make_pipeline(RobustScaler(), KernelRidge(**model_params[regressor]))
        elif regressor.lower() == 'ridge':
            scaler = make_pipeline(RobustScaler(), Ridge(**model_params[regressor]))
        elif regressor.lower() == 'elastic_net':
            scaler = make_pipeline(RobustScaler(), ElasticNet(**model_params[regressor]))
        elif regressor.lower() == 'gradient_boosting_regressor':
            scaler = make_pipeline(RobustScaler(), GradientBoostingRegressor(**model_params[regressor]))
        else:
            raise NotImplementedError

        # fit train set
        scaler.fit(train_df, target)

        # predict
        target_pred = scaler.predict(train_df.values)

        # save estimator as pickle file
        with open(model_dir.joinpath(f'{regressor}.pkl'), 'wb') as file:
            pickle.dump(scaler, file)

        # save metrics
        rmse_score = {'rmse': rmse(target, target_pred)}
        metrics = json.dumps(rmse_score)
        with open(results_dir.joinpath(f'{regressor}_metrics.json'), 'w') as writer:
            writer.writelines(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', dest='train_path',
                        required=True, help='Train CSV file')
    parser.add_argument('-rd', '--results-dir', dest='results_dir',
                        default=Path('./results').resolve(),
                        required=False, help='Metrics output directory')
    parser.add_argument('-md', '--model-dir', dest='model_dir',
                        default=Path('./models').resolve(),
                        required=False, help='Model output directory')
    args = parser.parse_args()

    # train model
    train_model(args.train_path, args.results_dir, args.model_dir)
