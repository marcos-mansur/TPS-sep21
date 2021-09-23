import pandas as pd
import numpy as np
import os, random, time
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
RANDOM_SEED = 42

#=================== Pre processing ==================
class preprocess(BaseEstimator, TransformerMixin):
    """Creates a columns with the sum of amount of missing value per row
    and the standart deviation of each row"""

    def __init__(self,sum_missing: bool=True, st_dev: bool=True):
        self.sum_missing = sum_missing
        self.st_dev = st_dev
        pass

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, df: pd.DataFrame):
        x = df.copy()
        if self.sum_missing:
            x['sum_missing'] = x.isna().sum(axis=1)
        if self.st_dev:
            x['std'] = x.std(axis=1)
        return x

#========================= Cross Validation =========================
def manual_cross(X:pd.DataFrame, y:pd.Series, N_SPLITS:int=5, model_params:dict=None):
    """ Manually coded cross-validation

    :param X: features
    :param y: target
    :param N_SPLITS: amount of splits
    :param params: dictionary of params for the model
    :return: list of folds scores """

    X = X.reset_index()
    y = y.reset_index().iloc[:, 1]
    strat_kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    scores = np.empty(N_SPLITS)
    for idx, (train_idx, test_idx) in enumerate(strat_kf.split(X, y)):
        print("=" * 12 + f"Training fold {idx}" + 12 * "=")
        start = time.time()

        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]
        eval_set = [(X_val, y_val)]

        lgbm_clf = LGBMClassifier(**model_params)
        lgbm_clf.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=200,
            eval_metric="binary_logloss",  # ROC_AUC?
            verbose=False,
        )

        preds = lgbm_clf.predict_proba(X_val)
        fold_score = roc_auc_score(y_val, preds[:, 1])
        scores[idx] = fold_score
        runtime = time.time() - start
        print(f"Fold {idx} finished with score: {fold_score:.5f} in {runtime:.2f} seconds.\n")
    print(f'Mean score of folds {scores.mean()}')
    return scores

#=========================== Fix Seed =========================
def seed_everything(seed=RANDOM_SEED):
    """ set random seed"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

#=================== Optimization ========================

def objective(trial, data, target):
    """ objetive function to be optmized by optuna """

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

    # split the Data
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.3)

    # the model
    model = LGBMRegressor(**lgbm_params)
    # fit the Data
    model.fit(train_x, train_y)
    # predict
    pred = model.predict(valid_x)
    # score
    roc_auc = roc_auc_score(valid_y, pred)
    return roc_auc

