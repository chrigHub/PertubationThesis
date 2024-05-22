import numpy as np
import pandas as pd
import datetime
import joblib
import importlib
from sklearn.metrics import balanced_accuracy_score
import os
import sys

# Variables
MODEL_SPEC = "RF"
DATA_SPEC = "B"

ROOT_PATH = os.path.abspath("../../.")
INPUT_FOLDER = os.path.join(ROOT_PATH, "data/training/training_results", MODEL_SPEC)
OUTPUT_FOLDER = os.path.join(ROOT_PATH, "data/perturbation/pert_output", MODEL_SPEC)
DATA_FOLDER = os.path.join(ROOT_PATH, "data/preprocessing", DATA_SPEC)

sys.path.insert(0, ROOT_PATH)
from main.utils import train_utils, data_manage_utils

importlib.reload(train_utils)
importlib.reload(data_manage_utils)

# PERT PARAMS
# // The higher the most reduced //
OPT_THRESHOLD = 1

# FOLDER SELECTION
path = [x[0] for x in os.walk(INPUT_FOLDER)][-1:][0] + "/"
print(f"Using model from path: {path}")
model = joblib.load(path + "model.joblib")


X_test = pd.read_pickle(os.path.join(DATA_FOLDER, "X_test_df.pkl"))
y_test = pd.read_pickle(os.path.join(DATA_FOLDER, "y_test_df.pkl"))
scaler = data_manage_utils.load_scaler_from_sav(os.path.noin(DATA_FOLDER, "scaler.sav"))

y_pred = data_manage_utils.load_numpy_from_pickle(path + "y_test_pred.pkl")
X_test_scaled = data_manage_utils.load_numpy_from_pickle(os.path.noin(DATA_FOLDER, "X_test_scaled.pkl"))

y_pred_2 = model.predict(X_test_scaled)
print(f"Balanced accuracy score of model: {balanced_accuracy_score(y_test, y_pred_2)}")
print(f"Shape of true labels: {y_test.shape} \n Shape of pred labels: {y_pred.shape}")


def pert_resolution(val, res):
    arr = []
    for i in np.arange(res / 10, res, res / 10):
        low_val = val - i
        high_val = val + i
        arr.append(low_val)
        arr.append(high_val)
    return arr


def pert_percent(val, perc):
    arr = []
    for i in range(1, 10, 1):
        low_val = val * (1 - ((perc / 1000) * i))
        high_val = val * (1 + ((perc / 1000) * i))
        arr.append(low_val)
        arr.append(high_val)
    return arr


def pert_percent_int(val, perc):
    arr = []
    for i in range(1, int(val * (perc / 100)), 1):
        low_val = val - i
        high_val = val + i
        arr.append(low_val)
        arr.append(high_val)
    return arr


def pert_cat(val, cats):
    arr = []
    for c in cats:
        if val != c:
            arr.append(c)
    return arr


def pert_ordinal_n(val, n):
    arr = []
    for v in range(1, n + 1):
        low_val = val - v
        high_val = val + v
        arr.append(low_val)
        arr.append(high_val)
    return arr


# MaxError would be higher. However not critical for perturbation so resolution was taken.
pert_temp = {
    "col": "TEMP(C)",
    "func": pert_resolution,
    "param": 0.1,
    "level": 3,
    "info": "Sensor resolution is only 0.1 degrees"
}
pert_felt_temp = {
    "col": "FELT_TEMP(C)",
    "func": pert_resolution,
    "param": 0.1,
    "level": 3,
    "info": "Sensor resolution is only 0.1 degrees"
}
pert_wind_speed = {
    "col": "WIND_SPEED(KMH)",
    "func": pert_percent,
    "param": 5,
    "level": 3,
    "info": "Sensor accuracy is 5%"
}
pert_humidity = {
    "col": "REL_HUMIDITY(PERCENT)",
    "func": pert_percent,
    "param": 5,
    "level": 3,
    "info": "Unknown"
}
pert_wind_drct = {
    "col": "WIND_DRCT(DEG)",
    "func": pert_resolution,
    "param": 10,
    "level": 3,
    "info": " 10 Degree resolution"
}
pert_flight_time = {
    "col": "CRS_ELAPSED_TIME(MINS)",
    "func": pert_percent_int,
    "param": 5,
    "level": 1,
    "info": "TBI"
}
pert_flight_dist = {
    "col": "DISTANCE(KM)",
    "func": pert_percent,
    "param": 5,
    "level": 1,
    "info": "TBI"
}
pert_nr_prev_flights = {
    "col": "NR_PREV_ARR_FLIGHTS(1HR)",
    "func": pert_percent_int,
    "param": 10,
    "level": 1,
    "info": "TBI"
}
pert_nr_engines = {
    "col": "NR_ENGINES",
    "func": pert_ordinal_n,
    "param": 2,
    "level": 1,
    "info": "TBI"
}
pert_jet_yn = {
    "col": "JET(YN)",
    "func": pert_cat,
    "param": [0, 1],
    "level": 1,
    "info": "TBI"
}
pert_winglets_yn = {
    "col": "WINGLETS(YN)",
    "func": pert_cat,
    "param": [0, 1],
    "level": 1,
    "info": "TBI"
}
pert_approach_speed = {
    "col": "APPROACH_SPEED(KMH)",
    "func": pert_percent,
    "param": 5,
    "level": 1,
    "info": "TBI"
}
pert_wingspan = {
    "col": "WINGSPAN(M)",
    "func": pert_percent,
    "param": 5,
    "level": 1,
    "info": "TBI"
}
pert_length = {
    "col": "LENGTH(M)",
    "func": pert_percent,
    "param": 5,
    "level": 1,
    "info": "TBI"
}
pert_tail_height = {
    "col": "TAIL_HEIGHT(M)",
    "func": pert_percent,
    "param": 5,
    "level": 1,
    "info": "TBI"
}
pert_mtow = {
    "col": "MTOW(KG)",
    "func": pert_percent,
    "param": 5,
    "level": 1,
    "info": "TBI"
}
pert_vsbty = {
    "col": "VISIBILITY(MILES)",
    "func": pert_ordinal_n,
    "param": 3,
    "level": 1,
    "info": "TBI"
}
pert_rw_1028 = {
    "col": "10/28",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_rw_09r27l = {
    "col": "09R/27L",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_rw_09l27r = {
    "col": "09L/27R",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_rw_08r26l = {
    "col": "08R/26L",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_rw_08l26r = {
    "col": "08L/26R",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_br = {
    "col": "EVENT_BR",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_dz = {
    "col": "EVENT_DZ",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_fg = {
    "col": "EVENT_FG",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_fu = {
    "col": "EVENT_FU",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_gr = {
    "col": "EVENT_GR",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_gs = {
    "col": "EVENT_GS",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_hz = {
    "col": "EVENT_HZ",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_ic = {
    "col": "EVENT_IC",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_sg = {
    "col": "EVENT_SG",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_ts = {
    "col": "EVENT_TS",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_sn = {
    "col": "EVENT_SN",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_event_ra = {
    "col": "EVENT_RA",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 2,
    "info": "TBI"
}
pert_1hour_pert = {
    "col": "1HOUR_PRECIPITATION(INCH)",
    "func": pert_resolution,
    "param": 0.01,
    "level": 3,
    "info": "Sensor resolution is 0.01 inch"
}
pert_sea_level_pressure = {
    "col": "SEA_LEVEL_PRESSURE(MILLIBAR)",
    "func": pert_resolution,
    "param": 0.005,
    "level": 3,
    "info": "Sensor resolution is 170 millibar"
}
pert_dep_delay = {
    "col": "DEP_DELAY(MINS)",
    "func": pert_percent_int,
    "param": 0.005,
    "level": 3,
    "info": "Sensor resolution is 170 millibar"
}
options = [pert_temp, pert_wind_speed, pert_humidity, pert_wind_drct, pert_flight_time, pert_nr_prev_flights,
           pert_winglets_yn, pert_mtow, pert_vsbty, pert_event_sn, pert_event_ra, pert_rw_1028, pert_rw_09r27l,
           pert_rw_09l27r, pert_rw_08r26l, pert_rw_08l26r, pert_event_ts, pert_event_br, pert_event_ic, pert_event_dz,
           pert_event_fg, pert_event_ra, pert_event_sn, pert_sea_level_pressure, pert_dep_delay]

arr = []
for o in options:
    arr.append(o.get("col"))
x_len = len(X_test.columns)
diff = set(X_test.columns) - set(arr)
neg_diff = set(arr) - set(X_test.columns)
print(str(x_len - len(diff)) + "/" + str(x_len))
print(f"diff: {diff}. neg-diff: {neg_diff}")

X_test["y_true"] = y_test

def filter_options(option, threshold):
        return option["level"] >= threshold

def perturbate(data: pd.DataFrame, option_threshold: int, options: list, verbosity=1):
    assert type(data) == pd.DataFrame, f"got type {type(data)} for param 'data'. Expected  pd.DataFrame"
    if option_threshold:
        assert type(option_threshold) == int, f"got type {type(option_threshold)} for param 'mode'. Expected str"
    assert type(options) == list, f"got type {type(options)} for param 'options'. Expected list"
    print(options)
    options = list(filter(lambda opt: opt["level"] >= option_threshold, options))
    print(options)
    data_copy = data.copy()
    data_copy["pert_id"] = None
    df_cols = data_copy.columns
    pert_lst = []
    cc = 1
    for opt in options:
        print("Starting run " + str(cc) + "/" + str(len(options)))
        c = 1
        for i, row in data_copy.iterrows():
            if verbosity > 1:
                print(f"{cc}/{str(len(options))}:({c}/{len(data_copy)})")
            row_pert_name = opt["col"] + "<" + str(i) + ">"
            row["pert_id"] = row_pert_name
            pert_lst.append(list(row))
            # row_df = pd.DataFrame(row)
            # pert_df = pert_df.append(row_df.T,ignore_index=True)
            for count, val in enumerate(opt["func"](row[opt["col"]], opt["param"]), start=1):
                row[opt["col"]] = val
                row["pert_id"] = row_pert_name
                pert_lst.append(list(row))
                # row_df = pd.DataFrame(row)
                # pert_df = pert_df.append(row_df.T,ignore_index=True)
            c += 1
        end, end_string = data_manage_utils.print_time(time_format="%Y_%m_%d %H:%M:%S")
        end = datetime.datetime.now()
        print("Ending run " + str(cc) + "/" + str(len(options)) + " at: " + end_string)
        cc += 1
    print("Converting list to pd.DataFrame...")
    pert_df = pd.DataFrame(pert_lst, columns=df_cols)
    pert_end, pert_end_string = data_manage_utils.print_time()
    print("Ending perturbation at: " + pert_end_string)
    print(f"Time elapsed: {pert_end - pert_start}")
    return pert_df


pert_start, pert_start_string = data_manage_utils.print_time()
print("Starting perturbation at: " + pert_start_string)
X_pert = perturbate(X_test.iloc[:], OPT_THRESHOLD, options)
print("Predicting on created dataframe")
X_pert['y'] = model.predict(scaler.transform(X_pert[X_pert.columns[:-2]]))
pert_end, pert_end_string = data_manage_utils.print_time()
print("Ending perturbation at: " + pert_end_string)
print(f"Time elapsed: {pert_end - pert_start}")
time, time_string = data_manage_utils.print_time(time_format="%Y_%m_%d-%H%M")

outdir = os.path.join(ROOT_PATH,"data/perturbation/pert_output", MODEL_SPEC)
if not os.path.exists(outdir):
    os.mkdir(outdir)
outdir = outdir + '/' + time_string
if not os.path.exists(outdir):
    os.mkdir(outdir)


fullname = os.path.join(outdir, "pert_out_df.pkl")
print(f"Shape of df: {X_pert.shape}")
X_pert.to_pickle(fullname)
