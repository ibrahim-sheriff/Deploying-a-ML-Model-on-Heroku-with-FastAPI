"""
Author: Ibrahim Sherif
Date: October, 2021
This script used to train and save the model
"""
import joblib
from sklearn.model_selection import train_test_split

import config
from data import get_clean_data
from slicing import validate_slice
from model import get_model_pipeline, train_model, inference_model, compute_metrics


def train(data_path, model, param_grid, feats):

    print("[INFO] Loading and getting clean data")
    X, y = get_clean_data(data_path)

    print("[INFO] Splitting data to train and test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.RANDOM_STATE, stratify=y)

    print("[INFO] Load model pipeline")
    model_pipe = get_model_pipeline(model, feats)

    print("[INFO] Train model")
    model_pipe = train_model(model_pipe, X_train, y_train, param_grid)

    print("[INFO] Running inference")
    y_train_pred = inference_model(model_pipe, X_train)
    y_test_pred = inference_model(model_pipe, X_test)

    print("[INFO] Evaluating model")
    print("Train metrics")
    print("Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}".format(
        *compute_metrics(y_train_pred, y_train)))

    print("Test metrics")
    print("Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}".format(
        *compute_metrics(y_test_pred, y_test)))

    with open(config.EVAL_DIR, 'w') as file:
        print("Evaluation on train data", file=file)
        print("Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}".format(
            *compute_metrics(y_train_pred, y_train)), file=file)

        print("Evaluation on test data", file=file)
        print("Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}".format(
            *compute_metrics(y_test_pred, y_test)), file=file)
        
        
    print("[INFO] Evaluating slices")
    with open(config.SLICE_DIR, 'w') as file:
        
        slice_columns = ['sex', 'race']
        for col in slice_columns:
            
            #print(f"[INFO] Metrics for {col} in train data")
            slice_df = validate_slice(col, X_train, y_train, y_train_pred)
            print(f"Model evaluation on {col} slice of train data", file=file)
            print(slice_df.to_string(header=False, index=False), file=file)
            
            
            #print(f"[INFO] Metrics for {col} in test data")
            slice_df = validate_slice(col, X_test, y_test, y_test_pred)
            print(f"Model evaluation on {col} slice of test data", file=file)
            print(slice_df.to_string(header=False, index=False), file=file)
            
            print("", file=file)

    
    return model_pipe


def run():

    model = train(
        config.DATA_DIR,
        config.MODEL,
        config.PARAM_GRID,
        config.FEATURES)
    joblib.dump(model, config.MODEL_DIR)


if __name__ == "__main__":
    run()
