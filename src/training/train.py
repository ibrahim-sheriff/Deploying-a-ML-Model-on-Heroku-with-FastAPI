import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from pipeline.data import get_clean_data
from pipeline.model import get_model_pipeline, train_model, inference_model, compute_metrics

def train(data_path, model, param_grid, feats):
    
    print("[INFO] Loading and getting clean data")
    X, y = get_clean_data(data_path)
    
    print("[INFO] Splitting data to train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17, stratify=y)
    
    print("Load model pipeline")
    model_pipe = get_model_pipeline(model, feats['categorical'], feats['numeric'], feats['drop'])
    
    print("Train model")
    model_pipe, _ = train_model(model_pipe, X_train, y_train, param_grid)
    
    print("Running inference")
    y_train_pred = inference_model(model_pipe, X_train)
    y_test_pred = inference_model(model_pipe, X_test)
    
    print("Train metrics")
    print("Precision = {}, Recall = {}, F1 = {}".format(*compute_metrics(y_train_pred, y_train)))
    
    print("Test metrics")
    print("Precision = {}, Recall = {}, F1 = {}".format(*compute_metrics(y_test_pred, y_test)))
    
    return model_pipe
    
    
def run():
    data_path = '../../data/edited_census.csv'
    model_save_path = '../../models/rf_model.joblib'
    
    param_grid = {
        'model__n_estimators': list(range(10, 21, 1)),
        'model__max_depth': list(range(2, 11, 1))
    }
    
    feats = {
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
    
    model = RandomForestClassifier(random_state=17)
    
    model = train(data_path, model, param_grid, feats)
    
    joblib.dump(model, model_save_path)



if __name__ == "__main__":
    run()
     