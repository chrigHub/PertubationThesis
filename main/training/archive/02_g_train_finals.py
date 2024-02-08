import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import os
import sys
import re
import ast

sys.path.insert(0, '../../..')
from main.utils import data_manage_utils

X_train_df = pd.read_pickle("processed_files/NEW/X_train_df.pkl")
y_train_df = pd.read_pickle("processed_files/NEW/y_train_df.pkl")
X_test_df = pd.read_pickle("processed_files/NEW/X_test_df.pkl")
y_test_df = pd.read_pickle("processed_files/NEW/y_test_df.pkl")
# scaler = data_manage_utils.load_scaler_from_sav("./processed_files/NEW/scaler.sav")

path_rf = [x[0] for x in os.walk("training_results/RF")][-1:][0]
path_svc = [x[0] for x in os.walk("training_results/SVC")][-1:][0]
path_nb = [x[0] for x in os.walk("training_results/NB")][-1:][0]
path_mlp = [x[0] for x in os.walk("training_results/MLP")][-1:][0]


def tryval(val):
    try:
        val = ast.literal_eval(val)
    except:
        pass
    return val


def load_dict_from_txt(path):
    param_dict = {}
    path = path + "/params_final.txt"
    with open(path, "r") as f:
        for line in f:
            line = re.sub('[ ]', '', line)[:-1]
            line = line.split(":")
            _d = {line[0][line[0].find('__')+2:]: tryval(line[1])}
            param_dict.update(_d)
    return param_dict


param_dict_rf = load_dict_from_txt(path_rf)
param_dict_svc = load_dict_from_txt(path_svc)
param_dict_nb = load_dict_from_txt(path_nb)
param_dict_mlp = load_dict_from_txt(path_mlp)

scaler = StandardScaler()
smote = SMOTE(random_state=42)
X_train_smote = X_train_df.copy()
X_train_smote, y_train_smote = smote.fit_resample(X_train_smote, y=y_train_df.to_numpy())
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_train_scaled = scaler.transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)


def make_preds(X_train, X_test, clf, path):
    print("Start predicting train labels.")
    y_train_pred = clf.predict(X_train)
    train_pred_path = os.path.join(path, "y_train_pred.pkl")
    y_train_pred_df = pd.DataFrame(y_train_pred, columns=["y_train_pred"])
    y_train_pred_df.to_pickle(train_pred_path)
    print("Done predicting train labels.")

    print("Start predicting test labels.")
    y_test_pred = clf.predict(X_test)
    test_pred_path = os.path.join(path, "y_test_pred.pkl")
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=["y_test_pred"])
    y_test_pred_df.to_pickle(test_pred_path)
    print("Done predicting test labels.")

    print("Saving model to .joblib")
    out_model_name = "model.joblib"
    out_model_path = os.path.join(path, out_model_name)
    joblib.dump(clf, out_model_path, compress=3)
    print(f"Uncompressed Model: {np.round(os.path.getsize(out_model_path) / 1024 / 1024, 2)} MB")


start, start_string = data_manage_utils.print_time()
print("Start time: " + start_string)

print("At RandomForestClassifier 1/4")
#clf_rf = RandomForestClassifier(**param_dict_rf)
#clf_rf.fit(X_train_smote_scaled, y_train_smote)
#make_preds(X_train_scaled, X_test_scaled, clf_rf, path_rf)
print("At SVC 2/4")
clf_svc = SVC(**param_dict_svc)
clf_svc.fit(X_train_smote_scaled, y_train_smote)
make_preds(X_train_scaled, X_test_scaled, clf_svc, path_svc)
print("At GaussianNB 3/4")
clf_nb = GaussianNB(**param_dict_nb)
clf_nb.fit(X_train_smote_scaled, y_train_smote)
make_preds(X_train_scaled, X_test_scaled, clf_nb, path_nb)
print("At MLPClassifier 4/4")
clf_mlp = MLPClassifier(**param_dict_mlp)
clf_mlp.fit(X_train_smote_scaled, y_train_smote)
make_preds(X_train_scaled, X_test_scaled, clf_mlp, path_mlp)

end, end_string = data_manage_utils.print_time()
print("End time: " + end_string)
print("Time elapsed: " + str(end - start))
