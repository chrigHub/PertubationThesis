import joblib
import numpy as np
import sys
import importlib
import json
import os
from sklearn.metrics import accuracy_score

sys.path.insert(0, '../../../..')
from main.utils import general_utils, train_utils, data_manage_utils

importlib.reload(general_utils)
importlib.reload(data_manage_utils)
importlib.reload(train_utils)

# DEFINING VARIABLES
PATH_STR = None
# PATH_STR = "./training_results/RF/2023_09_20-2108"
if not PATH_STR:
    PATH_STR = general_utils.find_latest_folder("./training_results")

print("\n\n\n================= Starting script =================")

print(f"\nLoading from path : {PATH_STR}")

print(f"\nLoading param search data ...")

with open(PATH_STR + "/settings.json", "r") as f:
    results = json.load(f)
result_info = results.get("result_info")

data_info = results.get("data_info")
data_folder = data_info.get("data_folder")
print(f"Loading training data from: {data_folder}")
X_train = data_manage_utils.load_numpy_from_pickle(os.path.join(data_folder, "X_train_df.pkl"))
y_train = data_manage_utils.load_numpy_from_pickle(os.path.join(data_folder, "y_train_df.pkl"))
X_test = data_manage_utils.load_numpy_from_pickle(os.path.join(data_folder, "X_test_df.pkl"))
y_test = data_manage_utils.load_numpy_from_pickle(os.path.join(data_folder, "y_test_df.pkl"))
scaler = data_manage_utils.load_scaler_from_sav(os.path.join(data_folder, "scaler.sav"))
print(f"\nScale and ravel data ... ")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()
print(f"Shape of y_train_numpy: {y_train.shape}")
print(f"Shape of y_test_numpy: {y_test.shape}")
print(f"X_train_scaled: σ = {np.std(X_train_scaled):.6f}, μ = {np.mean(X_train_scaled):.6f}")
print(f"X_test_scaled: σ = {np.std(X_test_scaled):.6f}, μ = {np.mean(X_test_scaled):.6f}")
print(f"First row of X_train: \n{X_train_scaled[0][:]}")

best_params = result_info.get("best_params")
clf_method = getattr(importlib.import_module(result_info.get("clf_module")), result_info.get("clf_name"))
print(f"\nBest params: \n{best_params}")
print(f"Clf method: \n{clf_method}")

clf = clf_method(**best_params)
print(f"\nUsing classifier: {clf}")

print(f"\n\nStarting training ...")
start, start_str = data_manage_utils.print_time()
print(f"Starting train fit at {start_str}")
clf.fit(X_train_scaled, y_train)
end, end_str = data_manage_utils.print_time()
time_needed = end - start
print(f"Ending train fit at {end_str} after {time_needed}")

print(f"\n\nStarting evaluation ...")
y_train_pred = clf.predict(X_train_scaled)
y_test_pred = clf.predict(X_test_scaled)
train_acc = accuracy_score(y_train, y_train_pred)

train_scores = train_utils.evaluate(y_train, y_train_pred)
test_scores = train_utils.evaluate(y_test, y_test_pred)
print(f"Train metrics: \n{train_scores}")
print(f"Test metrics: \n{test_scores}")

clf_results = {
    "model_path": PATH_STR,
    "data_path": data_folder,
    "clf_name": clf.__class__.__name__,
    "clf_module": clf.__class__.__module__,
    "clf_params": clf.get_params(),
    "scores_train": train_scores,
    "scores_test": test_scores,
    "fit_time": str(time_needed)
}

print("\n\nSaving result dict ...")
settings_filename = "train_results.json"
print(f"Location: {PATH_STR}/{settings_filename}")
data_manage_utils.save_search_params(PATH_STR, clf_results, settings_filename)

print("\n\nSaving model ...")
out_model_name = "model.joblib"
out_model_path = os.path.join(PATH_STR, out_model_name)
joblib.dump(clf, out_model_path, compress=3)
print(f"Uncompressed model size: {np.round(os.path.getsize(out_model_path) / 1024 / 1024, 2)} MB")
print("================= Script ended =================")
