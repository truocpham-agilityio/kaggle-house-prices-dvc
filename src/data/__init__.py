# -*- coding: utf-8 -*-
import os
from pathlib import Path

import yaml
import pandas as pd


def load_data(data_path, sep=',', header=None, index_col=None) -> object:
    """Helper function to load train and test files
     as well as optional param loading

    Args:
        data_path (str or list of str): path to csv file
        sep (str):
        index_col (str):
        header (int):

    Returns:
        object:
    """

    # if single path as str, convert to list of str
    if type(data_path) is str:
        data_path = [data_path]

    # loop ever filepath in list and read file
    output_df = [pd.read_csv(el, sep=sep, header=header, index_col=index_col) for el in data_path]

    # if single file as input, return single df not a list
    if len(output_df) == 1:
        output_df = output_df[0]

    return output_df


def load_params(filepath='params.yaml') -> dict:
    """Helper function to load params.yaml

    Args:
        filepath (str): filename or full filepath to yaml file with parameters

    Returns:
        dict: dictionary of parameters
    """

    assert (os.path.isfile(filepath)), FileNotFoundError

    # read params.yaml
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)

    return params


def save_as_csv(df, filepath, output_dir,
                replace_text='.csv', suffix='_processed.csv', na_rep='nan', output_path=False):
    """Helper function to format the new filename and save output

    Args:
        df (object): dataset for processing
        filepath (str): full filepath to dataset
        output_dir (str): output dir name or full output dir path to save the outputs
        replace_text (str): default the replaced text is .csv
        suffix (str): default the new suffix text is _processed.csv
        nan_rep (str): the pattern to replace in exporting csv, default is nan
        output_path (bool): the flag to return the output dir path

    Returns:
        void:
    """

    # if single path as str, convert to list of str
    if not isinstance(df, list):
        df = [df]

    if isinstance(filepath, str):
        filepath = [filepath]

    # list lengths must be equal
    assert (len(df) == len(filepath)), AssertionError

    output_dir = Path(output_dir).resolve()
    assert (os.path.isdir(output_dir)), NotADirectoryError(output_dir)

    for temp_df, temp_path in zip(df, filepath):
        # set output filenames
        save_fname = os.path.basename(temp_path.replace(replace_text, suffix))

        # save updated dataframes
        save_filepath = output_dir.joinpath(save_fname)
        temp_df.to_csv(save_filepath, na_rep=na_rep)
        if output_path:
            return save_filepath


def is_missing(df, columns) -> bool:
    """Helper function to check missing values on dataset

    Args:
        df (object): dataset for processing
        columns (array): the columns list

    Returns:
        bool: the flag to mark dataset still have missing values or not
    """
    for column in columns:
        if df[column].isnull().values.any():
            return True
        return False
