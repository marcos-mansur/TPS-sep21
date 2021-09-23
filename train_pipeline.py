import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Functions import (seed_everything,
                       preprocess,
                       manual_cross,
                       objective)
RANDOM_SEED = 42
seed_everything(RANDOM_SEED)



#load data
df = pd.read_csv(r'Data/train_sampled_red.csv',index_col='id')
x = df.drop('claim', axis=1)
y= df.claim
#split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_seed=RANDOM_SEED)

