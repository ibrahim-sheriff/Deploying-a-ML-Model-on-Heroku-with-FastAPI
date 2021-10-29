"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the model functions needed to build, train and evaluate the model
"""
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import fbeta_score, precision_score, recall_score


def get_model_pipeline(model, feats):

    try:
        assert isinstance(model, (LogisticRegression, RandomForestClassifier))
    except AssertionError:
        raise Exception(
            "Model should be RandomForestClassifier or LogisticRegression")

    if isinstance(model, RandomForestClassifier):
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=1000)
    elif isinstance(model, LogisticRegression):
        encoder = OneHotEncoder(handle_unknown='ignore')

    # categorical feature preprocessor
    categ_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        encoder
    )

    # numerical feature preprocessor
    numeric_preproc = StandardScaler()

    # features preprocessor
    feats_preproc = ColumnTransformer([
        ('drop', 'drop', feats['drop']),
        ('categorical', categ_preproc, feats['categorical']),
        ('numerical', numeric_preproc, feats['numeric'])
    ],
        remainder='passthrough'
    )

    # model pipeline
    model_pipe = Pipeline([
        ('features_preprocessor', feats_preproc),
        ('model', model)
    ])

    return model_pipe


def train_model(model, X_train, y_train, param_grid):

    g_search = GridSearchCV(
        model,
        param_grid,
        scoring='f1',
        cv=StratifiedKFold(),
        error_score='raise',
        n_jobs=4
    )

    _ = g_search.fit(X_train, y_train)

    return g_search.best_estimator_

def inference_model(model, X):

    preds = model.predict(X)
    return preds
