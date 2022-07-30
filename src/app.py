import uvicorn
from fastapi import FastAPI
import pandas as pd
from joblib import load
from io import BytesIO
import requests

mLink = 'https://github.com/Pierre0201/fastapi/blob/main/src/ressource/clf.joblib?raw=true'
mfile = BytesIO(requests.get(mLink).content)
clf = load(mfile)

# 2. Create the app object
app = FastAPI()

path = 'https://raw.githubusercontent.com/Pierre0201/fastapi/main/src/ressource/'
test_df = pd.read_csv(path+'submission_kernel02.csv')
feats = [f for f in test_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, stranger'}

@app.get('/predict')
def predict():
    y = clf.predict_proba(test_df[feats].loc[test_df['SK_ID_CURR']==100141], num_iteration=clf.best_iteration_)[:,1]
    return {
        'prediction': y[0]
        }

@app.get('/prediction/')
def get_prediction(json_credit: dict = Body({})):
    """
    Calculates the probability of default for a credit application.  
    Args:  
    - credit data (json).  
    Returns:  
    - probability of default (dict).
    """
    df_one_credit = pd.Series(json_credit).to_frame().transpose()
    probability = clf.predict_proba(df_one_credit, num_iteration=clf.best_iteration_)[:, 1][0]
    return {'probability': probability}


@app.post('/prediction/')
def get_prediction(json_credit: dict = Body({})):
    """
    Calculates the probability of default for a credit application.  
    Args:  
    - credit data (json).  
    Returns:  
    - probability of default (dict).
    """
    df_one_credit = pd.Series(json_credit).to_frame().transpose()
    probability = clf.predict_proba(df_one_credit, num_iteration=clf.best_iteration_)[:, 1][0]
    return {'probability': probability}
