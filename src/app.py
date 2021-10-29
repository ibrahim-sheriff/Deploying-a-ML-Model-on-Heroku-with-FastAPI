import yaml
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Body

from config import MODEL_DIR, APP_CONFIG
from api.schemas import Person, FeatureInfo


model = joblib.load(MODEL_DIR)

with open(APP_CONFIG) as fp:
    config = yaml.safe_load(fp)

app = FastAPI()

@app.get("/")
async def greetings():
    return "Greetings and salutations everybody"

@app.get("/feature_info/{feature_name}")
async def feature_info(feature_name: FeatureInfo):
    
    info = config['features_info'][feature_name]
    return info

@app.post("/predict/")
async def predict(person: Person = Body(..., examples=config['post_examples'])):
    
    person = person.dict()
    features = np.array([person[f] for f in config['features_info'].keys()]).reshape(1, -1)
    df = pd.DataFrame(features, columns=config['features_info'].keys())
    
    pred_label = int(model.predict(df))
    pred_probs = float(model.predict_proba(df)[:, 1])
    pred = '>50k' if pred_label == 1 else '<=50k'
    
    return {'Label': pred_label, 'Prob': pred_probs, 'Salary': pred}
