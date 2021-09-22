import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Functions import (seed_everything,
                       preprocess,
                       manual_cross,
                       objective)
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)

lgbm_params = {
    'num_leaves': trial.suggest_int('num_leaves', 10, 2000),
    'max_depth': trial.suggest_int('max_depth', 7, 12),
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 500, 2000),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    'n_estimators': trial.suggest_int('n_estimators', 100, 900),
    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),
    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 0.9),
    'bagging_freq': trial.suggest_int('bagging_freq', 2, 10),
    'random_state': 42,
    'num_threads': -1}

#load data
df = pd.read_csv(r'Data/train_sampled_red.csv',index_col='id')
x = df.drop('claim', axis=1)
y= df.claim
#split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_seed=RANDOM_SEED)

