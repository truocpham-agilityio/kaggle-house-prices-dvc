# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np

from scipy.stats import skew

from src.data import load_data, load_params, save_as_csv


def skew_data(train_path, test_path, output_dir):
    """Skewness values with threshold"""

    output_dir = Path(output_dir).resolve()
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # load data
    train_df, test_df = load_data([train_path, test_path], sep=',', header=0)

    # load params
    params = load_params()

    # concatenate df
    df = pd.concat([train_df, test_df], ignore_index=True)

    # cache the Id
    index_df = df['Id']

    if params['skew']['is_transform']:
        df = df.drop(params['ignore_cols'], axis=1)
        numeric_df = df.select_dtypes(exclude=['object'])
        skewness = numeric_df.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= params['skew']['threshold']].index
        df[skewness_features] = np.log1p(df[skewness_features])
        df = pd.get_dummies(df)

    # insert the Id at the beginning
    df.insert(loc=0, column='Id', value=index_df)

    # return datasets to train and test
    n_train = train_df.shape[0]
    train_df = df[:n_train]
    test_df = df[n_train:]

    # save data
    save_as_csv([train_df, test_df],
                [train_path, test_path],
                output_dir,
                replace_text='_categorized.csv',
                suffix='_skewed.csv',
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

    skew_data(args.train_path, args.test_path, output_dir=args.output_dir)
