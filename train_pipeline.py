import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from Functions import (seed_everything,
                       preprocess,
                       manual_cross,
                       objective)
from Pipelines import pipe_baseline,pipe_lgbm

try:
    RANDOM_SEED = 42
    seed_everything(RANDOM_SEED)

    #load Data
    df = pd.read_csv(r'Data/train_sampled_red.csv', index_col='id')
    x = df.drop('claim', axis=1)
    y= df.claim
    #split the Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= RANDOM_SEED)

    pipe_baseline.fit_transform(x_train,y_train)
    pred_baseline = pipe_baseline.predict_proba(x_test)
    score_baseline = roc_auc_score(y_test,pred_baseline[:,1])

    pipe_lgbm.fit_transform(x_train,y_train)
    pred_lgbm = pipe_lgbm.predict_proba(x_test)
except Exception as err:
    print(err)
