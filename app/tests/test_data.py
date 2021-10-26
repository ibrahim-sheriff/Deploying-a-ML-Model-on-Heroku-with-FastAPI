"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the test functions for dataset
"""
import great_expectations as ge


def test_columns_exist(data: ge.dataset.PandasDataset):

    columns = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'salary'
    ]

    for column in columns:
        assert data.expect_column_to_exist(
            column)['success'], f"{column} does not exist"


def test_column_dtypes(data: ge.dataset.PandasDataset):

    columns = {
        'age': 'int64',
        'workclass': 'object',
        'fnlgt': 'int64',
        'education': 'object',
        'education-num': 'int64',
        'marital-status': 'object',
        'occupation': 'object',
        'relationship': 'object',
        'race': 'object',
        'sex': 'object',
        'capital-gain': 'int64',
        'capital-loss': 'int64',
        'hours-per-week': 'int64',
        'native-country': 'object',
        'salary': 'object'
    }

    for column, dtype in columns.items():
        assert data.expect_column_values_to_be_of_type(
            column, dtype)['success'], f"{column} should be of type {dtype}"


def test_education_num_column(data: ge.dataset.PandasDataset):

    assert data.expect_column_values_to_be_between(
        'education-num', 1, 17
    )['success'], "education-num column includes unknown category"


def test_marital_status(data: ge.dataset.PandasDataset):

    categs = [
        'Never-married',
        'Married-civ-spouse',
        'Divorced',
        'Married-spouse-absent',
        'Separated',
        'Married-AF-spouse',
        'Widowed'
    ]

    assert data.expect_column_distinct_values_to_equal_set(
        'marital-status', categs
    )['success'], "marital-status column includes unknown category"


def test_label_salary(data: ge.dataset.PandasDataset):

    assert data.expect_column_distinct_values_to_equal_set(
        'salary', ['<=50K', '>50K']
    )['success'], "salary column includes more than two classes"


def test_hours_per_week_range(data: ge.dataset.PandasDataset):
    data.expect_column_values_to_be_between
    assert data.expect_column_values_to_be_between(
        'hours-per-week', 1, 99
    )['success'], "hours-per-week column is not within range of 1 and 99"
