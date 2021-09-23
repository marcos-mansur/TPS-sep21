import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from Functions import preprocess
from lightgbm import LGBMClassifier
import numpy as np

RANDOM_SEED= 42


best_lgbm_params= {
    'num_leaves': 635,
    "max_depth": 11,
    'min_data_in_leaf': 690,
    'learning_rate': 0.02270155819774826,
    'n_estimators': 806,
    'lambda_l1': 21,
    'lambda_l2': 11,
    'min_gain_to_split': 2.0362294407676766,
    'bagging_fraction': 0.6496364626925781,
    'bagging_freq': 2,
    'random_state': 42}


pipe_baseline = make_pipeline(LGBMClassifier(random_state=RANDOM_SEED))

pipe_preprocess = make_pipeline(preprocess(sum_missing=True, st_dev=True),
                                SimpleImputer(missing_values=np.nan,
                                              strategy='most_frequent',
                                              add_indicator=True))
pipe_lgbm = make_pipeline(pipe_preprocess,
                          LGBMClassifier(**best_lgbm_params))


# AGORA VAI