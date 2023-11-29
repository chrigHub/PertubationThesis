import numpy as np
import sys
import importlib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

sys.path.insert(0, '../..')
from main.utils import train_utils, data_manage_utils

importlib.reload(train_utils)
importlib.reload(data_manage_utils)

X_train = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/X_train_df.pkl")
y_train = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/y_train.pkl")
X_test = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/X_test_df.pkl")
y_test = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/y_test.pkl")
scaler = data_manage_utils.load_scaler_from_sav("./processed_files/NEW/scaler.sav")

X_train_scale = scaler.transform(X_train)
sm = SMOTE(random_state=42)

# Create the random grid
random_grid = {'n_estimators': np.arange(100, 1600, 100),
               'max_depth': np.arange(10, 110, 10),
               'max_samples': np.arange(0.1, 1.1, 0.1),
               'min_samples_split': [2, 3, 4]}

print("Random grid: ")
print(random_grid)

result_scores = train_utils.custom_random_search(X=X_train_scale, y=y_train, model_func=RandomForestClassifier,
                                                 random_grid=random_grid, nr_iter=200, n_folds=3, sampler=sm, n_jobs=8,
                                                 verbosity=10)
print(result_scores)

best_param_tuple = data_manage_utils.find_best_scores(result_scores, score_name="balanced_accuracy")
best_params = best_param_tuple[0]
time, time_string = data_manage_utils.print_time(time_format="%Y_%m_%d-%H%M")
out_dir = "../training/training_results/RF/" + time_string
data_manage_utils.save_search_params(out_dir=out_dir, param_dict=best_params)
