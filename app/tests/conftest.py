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

from ..pipeline import config


@pytest.fixture(scope='session')
def data():

    if not os.path.exists(config.DATA_DIR):
        pytest.fail(f"Data not found at path: {config.DATA_DIR}")

    df = ge.read_csv(config.DATA_DIR)

    return df


@pytest.fixture(scope='session')
def sample_data():

    if not os.path.exists(config.DATA_DIR):
        pytest.fail(f"Data not found at path: {config.DATA_DIR}")

    df = pd.read_csv(config.DATA_DIR, nrows=10)

    df['salary'] = df['salary'].map({'>50K': 0, '<=50K': 1})

    y = df.pop('salary')
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test
