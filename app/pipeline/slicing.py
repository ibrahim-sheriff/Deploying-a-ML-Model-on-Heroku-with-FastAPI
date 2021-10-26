"""
Author: Ibrahim Sherif
Date: October, 2021
This script provides function for validation on a slice of dataset
"""
import config
import pandas as pd
from data import get_clean_data
from model import get_model_pipeline, train_model, inference_model, compute_metrics


def validate_slice(column, X, y_true, y_pred):
    
    df = pd.concat([X[column].copy(), y_true], axis=1)
    df['salary_pred'] = y_pred

    metrics = []
    for categ in df[column].unique():
        prec, rec, f1 = compute_metrics(
            df[df[column] == categ]['salary_pred'],
            df[df[column] == categ]['salary']
        )
        metrics.append([categ, prec, rec, f1])
        #print(f"[INFO] {categ}: Precision = {prec:.3f}, Recall = {rec:.3f}, F1 = {f1:.3f}")
        
    
    return pd.DataFrame(metrics, columns=['Category', 'Precision', 'Recall', 'F1'])
