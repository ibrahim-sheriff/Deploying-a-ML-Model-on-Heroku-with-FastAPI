"""
Author: Ibrahim Sherif
Date: October, 2021
This script used to train and save the model
"""
from sklearn.model_selection import train_test_split

from pipeline.model import get_model_pipeline, train_model


def train(model, X_train, y_train, param_grid, feats):

    print("[INFO] Creating model pipeline")
    model_pipe = get_model_pipeline(model, feats)

    print(f"[INFO] Training {model.__class__.__name__} model")
    model_pipe = train_model(model_pipe, X_train, y_train, param_grid)

    return model_pipe
