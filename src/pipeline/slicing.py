"""
Author: Ibrahim Sherif
Date: October, 2021
This script provides function for validation on a slice of dataset
"""
import pandas as pd

from pipeline.model import inference_model
from pipeline.evaluate import compute_metrics


def slice_metrics(column, X, y_true, y_pred):
    
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

def evaluate_slices(file, model_pipe, column, X, y, split):
            
    print(f"[INFO] Evaluating {column} on slice of {split} data")
    
    y_pred = inference_model(model_pipe, X)
    slice_df = slice_metrics(column, X, y, y_pred)
    
    print(f"Model evaluation on {column} slice of train data", file=file)
    print(slice_df.to_string(index=False), file=file)
    print("", file=file)
    