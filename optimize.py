import optuna
from Functions import objective
from lightgbm import LGBMClassifier
import joblib

# load data
df = pd.read_csv('Data/train_sampled_reduced', index='id')  # loading sampled of 20% of df
x = df.drop('claim', axis=1)
y = df.claim

# parameters search space
lgbm_space = {
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
    'random_state': RANDOM_SEED,
    'num_threads': -1}

# define model
model = LGBMClassifier(**lgbm_space)

# turn off log messages
#optuna.logging.set_verbosity(optuna.logging.WARNING)
# create Study
study= optuna.create_study(direction='maximize')
# load and resume previous Study
#Study = joblib.load("Study/Study.pkl")
# optimize
study.optimize(lambda trial: objective(trial,x,y,model,lgbm_space),  n_trials=100)

#print results
print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# save Study to be resumed later
joblib.dump(study, "Study/Study.pkl")