# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from src.data import load_data, load_params


def predict_model(test_path, results_dir, model_dir):
    """Use trained models to make predictions"""

    results_dir = Path(results_dir).resolve()
    assert (os.path.isdir(results_dir)), NotADirectoryError

    model_dir = Path(model_dir).resolve()
    assert (os.path.isdir(model_dir)), NotADirectoryError

    # load data
    test_df = load_data(test_path, sep=',', header=0)

    # load params
    params = load_params()
    target_class = params['target_class']

    test_df.pop(target_class)
    id = test_df.pop('Id')

    # load estimators
    files = os.listdir(model_dir)
    for file_name in files:
        model_filepath = model_dir.joinpath(file_name)
        assert (os.path.isfile(model_filepath)), FileNotFoundError
        with open(model_filepath, 'rb') as model_file:
            estimator = pickle.load(model_file)

            # predict
            pred = np.expm1(estimator.predict(test_df.values))

            # write submission to csv
            submission = pd.DataFrame()
            submission['Id'] = id
            submission['SalePrice'] = pred
            submission.to_csv(f'{results_dir}/{file_name}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-te', '--test', dest='test_path',
                        required=True, help='Test CSV file')
    parser.add_argument('-rd', '--results-dir', dest='results_dir',
                        default=Path('./submission').resolve(),
                        required=False, help='Submissions output directory')
    parser.add_argument('-md', '--model-dir', dest='model_dir',
                        default=Path('./models').resolve(),
                        required=False, help='Model output directory')
    args = parser.parse_args()

    # predict model
    predict_model(args.test_path, args.results_dir, args.model_dir)
