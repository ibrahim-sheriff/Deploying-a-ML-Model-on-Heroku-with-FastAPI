import pandas as pd


def get_clean_data(path):
    
    df = pd.read_csv(path)
    
    # remove duplicates
    df = df[~df.duplicated()]
        
    df['salary'] = df['salary'].map({'>50K': 0, '<=50K': 1})
    
    y = df.pop('salary')
    X = df
    
    return X, y
