import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from Functions import preprocess
from lightgbm import LGBMClassifier
RANDOM_SEED= 42

pipe_baseline = make_pipeline(LGBMClassifier(random_state=RANDOM_SEED))

pipe_preprocess = make_pipeline(preprocess(sum_missing=True, st_dev=True),
                                SimpleImputer(missing_values=np.nan,
                                              strategy='most_frequent',
                                              add_indicator=True))
pipe_lgbm = make_pipeline(pipe_preprocess,
                          LGBMClassifier(**best_params_lgbm))


# AGORA VAI