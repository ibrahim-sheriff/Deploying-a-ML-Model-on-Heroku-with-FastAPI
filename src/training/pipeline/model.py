from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score


def get_model_pipeline(model, categ_feats, numeric_feats, drop_feats):
    
    if isinstance(model, RandomForestClassifier):
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=1000)
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
            ('drop', 'drop', drop_feats),
            ('categorical', categ_preproc, categ_feats),
            ('numerical', numeric_preproc, numeric_feats)
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
        n_jobs=3
    )
    
    cv_scores = g_search.fit(X_train, y_train)
    
    return g_search.best_estimator_, cv_scores
    
    
def compute_metrics(y_true, y_pred):    
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    return f1, precision, recall


def inference_model(model, X):
    
    preds = model.predict(X)
    return preds
