"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the function to fetch the data from local directory
"""
import pandas as pd


def get_clean_data(path):

    data_df = pd.read_csv(path)

    # remove duplicates
    data_df = data_df[~data_df.duplicated()]

    data_df['salary_df'] = data_df['salary_df'].map({'>50K': 0, '<=50K': 1})

    y_df = data_df.pop('salary_df')
    x_df = data_df

    return x_df, y_df
