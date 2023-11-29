import pandas as pd
import numpy as np
import datetime
import os
import joblib
import importlib
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.insert(0, '../../..')
from main.utils import data_manage_utils

importlib.reload(data_manage_utils)

path = None
# path = "./training_results/RF/2022_11_04-2350"
data_folder = "B"
if path is None:
    path = [x[0] for x in os.walk("training_results/RF")][-1:][0]
print(path)
param_dict = data_manage_utils.load_params_from_txt(path + "/params_final.txt")
print(param_dict)
param_dict = {str(key)[len("randomforestclassifier__"):]: param_dict[key] for key in param_dict}
param_dict.update({"verbose": 10})
print(param_dict)

X_train = data_manage_utils.load_numpy_from_pickle("./processed_files/" + data_folder + "/X_train_df.pkl")
y_train = data_manage_utils.load_numpy_from_pickle("./processed_files/" + data_folder + "/y_train.pkl").ravel()
X_test = data_manage_utils.load_numpy_from_pickle("./processed_files/" + data_folder + "/X_test_df.pkl")
y_test = data_manage_utils.load_numpy_from_pickle("./processed_files/" + data_folder + "/y_test.pkl").ravel()
scaler = data_manage_utils.load_scaler_from_sav("./processed_files/" + data_folder + "/scaler.sav")

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

start, start_string = data_manage_utils.print_time()
print("Start time: " + start_string)
clf = RandomForestClassifier(**param_dict)

clf.fit(X_train_scaled, y_train)

end, end_string = data_manage_utils.print_time()
end = datetime.datetime.now()
print("End time: " + end_string)
print("Time elapsed: " + str(end - start))

print("Predicting train labels.")
y_pred_train = clf.predict(X_train_scaled)
out_pred_train_name = "y_train_pred.pkl"
out_pred_train_path = os.path.join(path, out_pred_train_name)
y_pred_train_df = pd.DataFrame(y_pred_train, columns=["y_pred_train_rf"])
y_pred_train_df.to_pickle(out_pred_train_path)
print("Done predicting train labels.")

print("Start predicting test labels")
y_pred_test = clf.predict(X_test_scaled)
out_pred_test_name = "y_test_pred.pkl"
out_pred_test_path = os.path.join(path, out_pred_test_name)
y_pred_test_df = pd.DataFrame(y_pred_test, columns=["y_pred_test_rf"])
y_pred_test_df.to_pickle(out_pred_test_path)
print("Done predicting test labels.")

print("Saving model to .joblib")
out_model_name = "model.joblib"
out_model_path = os.path.join(path, out_model_name)
joblib.dump(clf, out_model_path, compress=3)
print(f"Uncompressed Random Forest: {np.round(os.path.getsize(out_model_path) / 1024 / 1024, 2)} MB")
