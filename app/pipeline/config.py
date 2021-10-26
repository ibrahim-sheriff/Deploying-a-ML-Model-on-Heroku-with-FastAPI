"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the config data for training model pipeline and running tests 
related to the pipeline
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 17
__MAIN_DIR = os.path.abspath('../')


#MODEL = RandomForestClassifier(random_state=RANDOM_STATE)
MODEL = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
PARAM_GRID = None

if isinstance(MODEL, RandomForestClassifier):
    PARAM_GRID = {
        'model__n_estimators': list(range(50, 150, 25)),
        'model__max_depth': list(range(2, 11, 2)),
        'model__min_samples_leaf': list(range(1, 51, 5)),
    }
elif isinstance(MODEL, LogisticRegression):
    PARAM_GRID = {
        'model__C': np.linspace(0.1, 10, 5)
    }

FEATURES = {
    'categorical': [
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'workclass',
        'native-country'
    ],
    'numeric': [
        'age',
        'fnlgt',
        'capital-gain',
        'capital-loss',
        'hours-per-week'
    ],
    'drop': ['education']
}

__DATA_FILE = 'edited_census.csv'
__MODEL_FILE = 'pipe_' + MODEL.__class__.__name__
__EVAL_FILE = 'model_evaluation.txt'
__SLICE_FILE = 'slice_output.txt'

DATA_DIR = os.path.join(__MAIN_DIR, 'data', __DATA_FILE)
MODEL_DIR = os.path.join(__MAIN_DIR, 'models', __MODEL_FILE)
EVAL_DIR = os.path.join(__MAIN_DIR, 'metrics', __EVAL_FILE)
SLICE_DIR = os.path.join(__MAIN_DIR, 'metrics', __SLICE_FILE)
