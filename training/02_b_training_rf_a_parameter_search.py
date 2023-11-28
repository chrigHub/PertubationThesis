import importlib
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV

sys.path.insert(0, "./..")
from utils import data_manage_utils, train_utils

importlib.reload(data_manage_utils)
importlib.reload(train_utils)

# DEFINING VARIABLES
DATA_SPEC = "B"
CLF_TYPE = "KNN"
PARAM_EST_TYPE = "SH"
PARAMS_DICT = {
    "RF": {
        "n_estimators": [int(x) for x in np.linspace(start=25, stop=1600, num=20)],
        "max_depth": [1, 2, 3, None],
        "max_features": ["sqrt", 0.3, 0.6, 0.9, 1.0],
        "min_samples_split": [2, 0.1, 0.2, 0.3],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    },
    "KNN": {
        "n_neighbors": [int(x) for x in np.linspace(start=5, stop=60, num=20)],
        "weights": ["uniform", "distance"],
        "leaf_size": [int(x) for x in np.linspace(start=10, stop=100, num=20)],
        "p": [1,2]
    }
}
BASE_CLF_OPT_DICT = {
    "RF": {
        "clf": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "sh_params": {
            "min_resources": 25,
            "resource": "n_estimators",
            "max_resources": 1600,
            "verbose": 1
        },
        "eg_params": {
            "verbose": 10
        }
    },
    "KNN": {
        "clf": KNeighborsClassifier(algorithm="ball_tree"),
        "sh_params": {
            "min_resources": 100,
            "resource": "n_samples",
            "max_resources": "auto",
            "verbose": 1
        },
        "eg_params": {
            "verbose": 10
        }
    }
}
PARAM_GRID = PARAMS_DICT.get(CLF_TYPE)
if PARAM_EST_TYPE == "SH":
    resource = BASE_CLF_OPT_DICT.get(CLF_TYPE).get("sh_params").get("resource")
    if resource in PARAM_GRID.keys():
        PARAM_GRID.pop(BASE_CLF_OPT_DICT.get(CLF_TYPE).get("sh_params").get("resource"))

BASE_CLF = BASE_CLF_OPT_DICT.get(CLF_TYPE).get("clf")
base_clf_params = BASE_CLF.get_params()
PARAM_EST_PARAMS_GEN = {
    "estimator": BASE_CLF,
    "param_grid": PARAM_GRID,
    "cv": 5,
    "scoring": "accuracy",
    "n_jobs": 5
}

PARAM_EST_DICT = {
    "SH": HalvingGridSearchCV(factor=2, **PARAM_EST_PARAMS_GEN, **BASE_CLF_OPT_DICT.get(CLF_TYPE).get("sh_params")),
    "EG": GridSearchCV(**PARAM_EST_PARAMS_GEN, **BASE_CLF_OPT_DICT.get(CLF_TYPE).get("eg_params"))
}

METHOD_NAME_DICT = {
    "SH": "Successive Halving",
    "EG": "Exhaustive Grid"
}

print("\n\n\n================= Starting script =================")
data_folder = "./processed_files/" + DATA_SPEC
run_time = data_manage_utils.print_time("%Y_%m_%d-%H%M")[1]
output_folder = "../training/training_results/" + CLF_TYPE + "/" + run_time
print(f"Data folder: {data_folder}")
print(f"Output folder: {output_folder}")

print("\nLoading data ...")
X_train = data_manage_utils.load_numpy_from_pickle(data_folder + "/X_train_df.pkl")
print(f"Shape of X_train: {X_train.shape}")
y_train = data_manage_utils.load_numpy_from_pickle(data_folder + "/y_train_df.pkl")
print(f"Shape of y_train: {y_train.shape}")
X_test = data_manage_utils.load_numpy_from_pickle(data_folder + "/X_test_df.pkl")
print(f"Shape of X_test: {X_test.shape}")
y_test = data_manage_utils.load_numpy_from_pickle(data_folder + "/y_test_df.pkl")
print(f"Shape of X_test: {y_test.shape}")
scaler = data_manage_utils.load_scaler_from_sav(data_folder + "/scaler.sav")
data_dict = {
    "data_folder": data_folder,
    "X_train shape": X_train.shape,
    "y_train shape": y_train.shape,
    "X_test shape": X_test.shape,
    "y_test shape": y_test.shape
}

print("\nScaling data ...")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_numpy = y_train.to_numpy().ravel()
y_test_numpy = y_test.to_numpy().ravel()
print(f"Shape of y_train_numpy: {y_train.shape}")
print(f"Shape of y_test_numpy: {y_test.shape}")
print(f"X_train_scaled: σ = {np.std(X_train_scaled):.6f}, μ = {np.mean(X_train_scaled):.6f}")
print(f"X_test_scaled: σ = {np.std(X_test_scaled):.6f}, μ = {np.mean(X_test_scaled):.6f}")

print(f"\nPreparing {METHOD_NAME_DICT.get(PARAM_EST_TYPE)} ...")
param_est = PARAM_EST_DICT.get(PARAM_EST_TYPE)

print(f"{METHOD_NAME_DICT.get(PARAM_EST_TYPE)} Parameters:\n {param_est}")

clf_str = re.sub(' {2,}', ' ', str(BASE_CLF).replace("\n", ""))
# REMOVED DUE TO NO USE
# search_method_str = re.sub(' {2,}', ' ', str(sh).replace("\n", ""))

search_method_dict = {
    "method_type": str(type(param_est)),
    "method_params": data_manage_utils.params_to_string_dict(param_est.get_params())
}

param_dict = {
    "base_clf": clf_str,
    "param_grid": PARAM_GRID,
    "search_method": search_method_dict
}

print(f"\nPerforming {METHOD_NAME_DICT.get(PARAM_EST_TYPE)} ...")
start, start_str = data_manage_utils.print_time()
print(f"Starting at {start_str}\n")
param_est.fit(X_train_scaled, y_train_numpy)
end, end_str = data_manage_utils.print_time()
print(f"Ending at {end_str} after {end - start} time")
if PARAM_EST_TYPE == "SH":
    print(f"Number of resources: {param_est.n_resources_}")
    print(f"Number of candidates: {param_est.n_candidates_}")
print(f"Best params: {param_est.best_params_}")
print(f"Best score: {param_est.best_score_}")
best_param_dict = param_est.best_params_
base_clf_params.update(best_param_dict)

print("\nCreating result DataFrame ...")
result_cv_df = pd.DataFrame(param_est.cv_results_)
if not os.path.exists(output_folder):
    print(f"Creating folder '{output_folder}'")
    os.makedirs(output_folder)
output_path_results = f"{output_folder}/cv_results.pkl"
result_cv_df.to_pickle(output_path_results)

result_dict = {
    "time_start": start_str,
    "time_end": end_str,
    "time_needed": str(end - start),
    "clf_name": BASE_CLF.__class__.__name__,
    "clf_module": BASE_CLF.__class__.__module__,
    "best_params": base_clf_params,
    "score_metric": PARAM_EST_PARAMS_GEN.get("scoring"),
    "score_value": param_est.best_score_,
    "cv_result_location": output_path_results
}

if PARAM_EST_TYPE == "SH":
    result_dict.update({
        "n_resources": param_est.n_resources_,
        "n_candidates": param_est.n_candidates_,
    })

final_dict = {
    "data_info": data_dict,
    "param_info": param_dict,
    "result_info": result_dict
}

print("\nSaving result dict ...")
settings_filename = "settings.json"
print(f"Location: {output_folder}/{settings_filename}")
data_manage_utils.save_search_params(output_folder, final_dict, settings_filename)
print("================= Script ended =================")
