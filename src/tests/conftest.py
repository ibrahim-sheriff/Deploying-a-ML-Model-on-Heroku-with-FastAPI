"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the conftest data used with pytest module
"""
import os
import pytest
import pandas as pd
import great_expectations as ge
from sklearn.model_selection import train_test_split

import config
from pipeline.data import get_clean_data


@pytest.fixture(scope='session')
def data():

    if not os.path.exists(config.DATA_DIR):
        pytest.fail(f"Data not found at path: {config.DATA_DIR}")

    X_df, y_df = get_clean_data(config.DATA_DIR)
    X_df['salary'] = y_df
    X_df['salary'] = X_df['salary'].map({1: '>50k', 0: '<=50k'})
    
    df = ge.from_pandas(X_df)

    return df


@pytest.fixture(scope='session')
def sample_data():

    if not os.path.exists(config.DATA_DIR):
        pytest.fail(f"Data not found at path: {config.DATA_DIR}")

    data_df = pd.read_csv(config.DATA_DIR, nrows=10)

    # chaning column names to use _ instead of -
    columns = data_df.columns
    columns = [col.replace('-', '_') for col in columns]
    data_df.columns = columns
    
    # make all characters to be lowercase in string columns
    data_df = data_df.applymap(
        lambda s: s.lower() if type(s) == str else s)

    data_df['salary'] = data_df['salary'].map({'>50k': 1, '<=50k': 0})

    y_df = data_df.pop('salary')
    X_df = data_df

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.3, random_state=config.RANDOM_STATE, stratify=y_df
    )

    return X_train, X_test, y_train, y_test
