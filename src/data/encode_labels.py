# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from src.data import load_data, load_params, save_as_csv


def encode_labels(train_path, test_path, output_dir, remove_nan=False):
    """Encode categorical labels as numeric, save the processed data"""

    output_dir = Path(output_dir).resolve()
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # load data
    train_df, test_df = load_data([train_path, test_path], sep=',', header=0)

    # load params
    params = load_params()

    # concatenate df
    df = pd.concat([train_df, test_df], ignore_index=True)

    # convert binary 0/1 classification
    lb = LabelBinarizer()
    binary_to_numerical_cols = params['encoding']['binary_to_numerical_cols']
    for col in binary_to_numerical_cols:
        df[col] = lb.fit_transform(df[col])

    # covert numerical to categorical
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    df['OverallCond'] = df['OverallCond'].astype(str)
    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)

    # label encoding to numeric
    cat_df = df.select_dtypes(include=['object'])
    for col in cat_df.columns.values:
        # fill missing value
        df[col].fillna('None', inplace=True)

        # label encode
        le = LabelEncoder()
        col_enc = str(col) + '_label'
        le_labels = le.fit_transform(df[col])
        df[col_enc] = le_labels

        # one hot encode
        ohe = OneHotEncoder()
        arr_enc = ohe.fit_transform(df[[col_enc]]).toarray()
        labels_enc = list(le.classes_)
        ohe_enc_df = pd.DataFrame(arr_enc, columns=labels_enc)

        # add encoded attributes to categorical dataframe
        df[labels_enc] = ohe_enc_df[labels_enc]

    # remove nan (if applicable)
    if remove_nan:
        df = df.dropna(axis=0, how='any')

    # return datasets to train and test
    n_train = train_df.shape[0]
    train_df = df[:n_train]
    test_df = df[n_train:]

    # save data
    save_as_csv([train_df, test_df],
                [train_path, test_path],
                output_dir,
                replace_text='_nan_imputed.csv',
                suffix='_categorized.csv',
                na_rep='nan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', dest='train_path',
                        required=True, help='Train CSV data file path')
    parser.add_argument('-te', '--test', dest='test_path',
                        required=True, help='Test CSV data file path')
    parser.add_argument('-o', '--output_dir', dest='output_dir',
                        required=False, help='Output directory')
    parser.add_argument('-r', '--remove-nan', dest='remove_nan',
                        default=False, required=False,
                        help='Remove nan rows from training dataset')

    args = parser.parse_args()

    encode_labels(args.train_path, args.test_path,
                  output_dir=args.output_dir, remove_nan=args.remove_nan)
