# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

from src.data import load_data, load_params, save_as_csv


def remove_outliers(train_path, test_path, output_dir):
    """Remove outliers values"""

    output_dir = Path(output_dir).resolve()
    assert (os.path.isdir(output_dir)), NotADirectoryError(output_dir)

    # load data
    train_df, test_df = load_data([train_path, test_path], sep=',', header=0)

    # load params
    params = load_params()

    # optionally remove outliers
    target_class = params['target_class']
    if params['is_drop_outlier']:
        train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (
            train_df[target_class] < 300000)].index, inplace=True)
        train_df.drop(train_df[(train_df['GarageArea'] > 800) & (
            train_df[target_class] > 700000)].index, inplace=True)
        train_df.drop(train_df[(train_df['TotalBsmtSF'] > 6000) & (
            train_df[target_class] < 200000)].index, inplace=True)

    # save data
    save_as_csv([train_df, test_df],
                [train_path, test_path],
                output_dir,
                replace_text='.csv',
                suffix='_outliers_removed.csv',
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

    remove_outliers(args.train_path, args.test_path, output_dir=args.output_dir)
