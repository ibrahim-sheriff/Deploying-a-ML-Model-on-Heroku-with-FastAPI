"""
Author: Ibrahim Sherif
Date: October, 2021
This script provides function for validation on a slice of dataset
"""
import sys
import logging
import pandas as pd

from pipeline.model import inference_model
from pipeline.evaluate import compute_metrics


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def slice_metrics(column, X, y_true, y_pred):
    """
    Calculates metrics on a slice of data for a specific column

    Args:
        column (str): Column name representing a feature
        X (pandas dataframe): data features
        y_true ([type]): data true labels
        y_pred ([type]): data predicted labels

    Returns:
        pandas dataframe: Dataframe with metrics for each category
    """
    df = pd.concat([X[column].copy(), y_true], axis=1)
    df['salary_pred'] = y_pred

    metrics = []
    for categ in df[column].unique():
        prec, rec, f1 = compute_metrics(
            df[df[column] == categ]['salary_pred'],
            df[df[column] == categ]['salary']
        )
        metrics.append([categ, prec, rec, f1])
        # print(f"[INFO] {categ}: Precision = {prec:.3f}, Recall = {rec:.3f}, F1 = {f1:.3f}")

    return pd.DataFrame(
        metrics,
        columns=[
            'Category',
            'Precision',
            'Recall',
            'F1'])


def evaluate_slices(file, model_pipe, column, X, y, split):
    """
    Evaluting model on a slice of data for a specific column
    and data split and saving the results to a file

    Args:
        file (file): file object
        model_pipe (sklearn pipeline/model): sklearn model or pipeline
        column (str): Column name representing a feature
        X (pandas dataframe): data features
        y (pandas series): data labels
        split (str): train or test split

    Returns:
        None
    """
    logging.info(f"Evaluating {column} on slice of {split} data")

    y_pred = inference_model(model_pipe, X)
    slice_df = slice_metrics(column, X, y, y_pred)

    print(f"Model evaluation on {column} slice of train data", file=file)
    print(slice_df.to_string(index=False), file=file)
    print("", file=file)
