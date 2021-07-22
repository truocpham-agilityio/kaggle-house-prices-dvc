# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from src.data import is_missing, load_data, load_params, save_as_csv


def replace_nan(train_path, test_path, output_dir):
    """Replace NaN values"""

    output_dir = Path(output_dir).resolve()

    assert (os.path.isdir(output_dir)), NotADirectoryError(output_dir)

    # load data
    train_df, test_df = load_data([train_path, test_path], sep=',', header=0, index_col='Id')

    # load params
    params = load_params()

    # concatenate df
    df = pd.concat([train_df, test_df], ignore_index=True)

    # fill NaNs with the default strategy is mean
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(params['ignore_cols'])
    imputer = SimpleImputer(missing_values=np.NaN, strategy=params['imputation']['method'])
    for col in num_cols:
        # fit imputing with numerical column
        imputer = imputer.fit(df[[col]])

        # assign imputed value for numerical column
        df[col] = imputer.transform(df[[col]]).ravel()

    # make sure no missing values
    assert (not is_missing(df, df.columns)), AssertionError

    # return datasets to train and test
    n_train = train_df.shape[0]
    train_df = df[:n_train]
    test_df = df[n_train:]

    # save data
    save_as_csv([train_df, test_df],
                [train_path, test_path],
                output_dir,
                replace_text='.csv',
                suffix='_nan_imputed.csv',
                na_rep='nan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', dest='train_path',
                        required=True, help='Train CSV data file path')
    parser.add_argument('-te', '--test', dest='test_path',
                        required=True, help='Test CSV data file path')
    parser.add_argument('-o', '--output_dir', dest='output_dir',
                        required=False, help='Output directory')

    args = parser.parse_args()

    replace_nan(args.train_path, args.test_path, output_dir=args.output_dir)
