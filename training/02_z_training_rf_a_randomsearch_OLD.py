import pandas as pd
import numpy as np
import sys
import importlib

from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline

sys.path.insert(0, './..')
from utils import data_manage_utils, train_utils

importlib.reload(train_utils)
importlib.reload(data_manage_utils)

INPUT_FILES = "B"


X_train = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/X_train_df.pkl")
y_train = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/y_train.pkl")
X_test = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/X_test_df.pkl")
y_test = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/y_test.pkl")
scaler = data_manage_utils.load_scaler_from_sav("./processed_files/NEW/scaler.sav")

X_train_scale = scaler.transform(X_train)
sm = SMOTE(random_state=42)

max_depth_arr = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth_arr.append(None)
# Create the random grid
random_grid = {'n_estimators': [int(x) for x in np.linspace(start=100, stop=2000, num=20)],
               'max_depth': max_depth_arr,
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'max_features': ['sqrt', 'log2'],
               'bootstrap': [True, False]}

print("Random grid: ")
print(random_grid)

# START VARIABLES
verbosity = 10
nr_jobs = 5
scorer = make_scorer(f1_score, average='macro', greater_is_better=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
nr_runs = 200
# END VARIABLES

imba_pipeline = make_pipeline(SMOTE(random_state=42), RandomForestClassifier(random_state=42))

new_params = {'randomforestclassifier__' + key: random_grid[key] for key in random_grid}
print(new_params)
rand_search = RandomizedSearchCV(imba_pipeline, n_iter=nr_runs, param_distributions=new_params, cv=kf,
                                 scoring=scorer, return_train_score=True, n_jobs=nr_jobs,
                                 verbose=verbosity)
rand_search.fit(X_train_scale, y_train)

best_params = rand_search.best_params_
score = rand_search.best_score_
mean_fit_time = np.mean(rand_search.cv_results_.get("mean_fit_time"))
if mean_fit_time > 60:
    mean_fit_time = f"{(mean_fit_time/60):.2f} mins"
else:
    mean_fit_time = f"{mean_fit_time:.2f} secs"
print(f"Best params: {best_params}")
print(f"Best score: {scorer}")
print(f"Mean fit time: {mean_fit_time}")

time, time_string = data_manage_utils.print_time(time_format="%Y_%m_%d-%H%M")
out_dir = "../training/training_results/RF/" + time_string
data_manage_utils.save_search_params(out_dir=out_dir, param_dict=best_params)
with open("../training/training_results/RF/" + time_string + "/score.txt", "wb") as f:
    f.write(scorer)
    f.write(score)
    f.close()
