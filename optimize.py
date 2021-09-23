import optuna
from Functions import objective
from lightgbm import LGBMClassifier
import joblib
import pandas as pd

RANDOM_SEED =42
# load data
df = pd.read_csv('Data/train_sampled_reduced', index='id')  # loading sampled of 20% of df
x = df.drop('claim', axis=1)
y = df.claim

# parameters search space


# turn off log messages
#optuna.logging.set_verbosity(optuna.logging.WARNING)
# create Study
study= optuna.create_study(direction='maximize')
# load and resume previous Study
#Study = joblib.load("Study/Study.pkl")
# optimize
study.optimize(lambda trial: objective(trial,x,y),  n_trials=100)

#print results
print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# save Study to be resumed later
joblib.dump(study, "Study/Study.pkl")