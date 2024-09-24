import argparse

import numpy as np
import pandas as pd
import datetime
import joblib
import importlib
from sklearn.metrics import accuracy_score
import os
import sys

ROOT_PATH = os.path.abspath("../../.")

sys.path.insert(0, ROOT_PATH)
from main.utils import train_utils, data_manage_utils

importlib.reload(train_utils)
importlib.reload(data_manage_utils)

# PERT PARAMS
# // The higher the most reduced //
OPT_THRESHOLD = 1


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


def pert_extreme(val, tup):
    arr = []
    if val == tup[0]:
        return [tup[1]]
    return []


# MaxError would be higher. However, not critical for perturbation so resolution was taken.
pert_temp = {
    "col": "TEMP(C)",
    "func": pert_percent,
    "param": 0.75,
    "level": 3,
    "info": "Sensor tolerance is 0.75%"
}
pert_dew_temp = {
    "col": "DEWPOINT_TEMP(C)",
    "func": pert_percent,
    "param": 0.9,
    "level": 3,
    "info": "Sensor tolerance is 0.9%"
}
pert_wind_speed = {
    "col": "WIND_SPEED(KMH)",
    "func": pert_percent,
    "param": 3.7,
    "level": 3,
    "info": "Sensor accuracy is 3.7%"
}
pert_humidity = {
    "col": "REL_HUMIDITY(PERCENT)",
    "func": pert_percent,
    "param": 2,
    "level": 3,
    "info": "Based on calculation with temperature and dew point temperature worst case tolerance is 2%."
}
pert_wind_drct = {
    "col": "WIND_DRCT(DEG)",
    "func": pert_resolution,
    "param": 5,
    "level": 3,
    "info": " 10 Degree resolution"
}
pert_elapsed_time = {
    "col": "CRS_ELAPSED_TIME(MINS)",
    "func": pert_percent_int,
    "param": 5,
    "level": 1,
    "info": "Standard perturbation value. Default guess"
}
pert_nr_prev_flights = {
    "col": "NR_PREV_ARR_FLIGHTS(1HR)",
    "func": pert_percent_int,
    "param": 5,
    "level": 1,
    "info": "Standard perturbation value. Default guess"
}
pert_winglets_yn = {
    "col": "WINGLETS(YN)",
    "func": pert_cat,
    "param": [0, 1],
    "level": 1,
    "info": "Default for 2-value based perturbation. Not high priority"
}
pert_approach_speed = {
    "col": "APPROACH_SPEED(KMH)",
    "func": pert_percent,
    "param": 1.4,
    "level": 1,
    "info": "1.4% average closest distance to other unique values. Chosen since approach speed is not strictly cardinal."
}

pert_tail_height = {
    "col": "TAIL_HEIGHT(M)",
    "func": pert_percent,
    "param": 4,
    "level": 1,
    "info": "4% average closest distance to other unique values. Chosen since tail height is not strictly cardinal."
}

pert_parking_area = {
    "col": "PARKING_AREA(SQM)",
    "func": pert_percent,
    "param": 6,
    "level": 1,
    "info": "6% average closest distance to other unique values. Chosen since parking area is not strictly cardinal."
}

pert_vsbty = {
    "col": "VISIBILITY(MILES)",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 1,
    "info": "According to documentation value might be inaccurate by 1."
}

pert_event_br = {
    "col": "EVENT_BR",
    "func": pert_extreme,
    "param": (2, 3),
    "level": 2,
    "info": "No value for event 3 found. Might influence data on occurrence."
}
pert_event_dz = {
    "col": "EVENT_DZ",
    "func": pert_extreme,
    "param": (2, 3),
    "level": 2,
    "info": "No value for event 3 found. Might influence data on occurrence."
}
pert_event_fg = {
    "col": "EVENT_FG",
    "func": pert_extreme,
    "param": (2, 3),
    "level": 2,
    "info": "No value for event 3 found. Might influence data on occurrence."
}
pert_event_fu = {
    "col": "EVENT_FU",
    "func": pert_extreme,
    "param": (2, 3),
    "level": 2,
    "info": "No value for event 3 found. Might influence data on occurrence."
}
pert_event_gr = {
    "col": "EVENT_GR",
    "func": pert_extreme,
    "param": (2, 3),
    "level": 2,
    "info": "No value for event 3 found. Might influence data on occurrence."
}
pert_event_gs = {
    "col": "EVENT_GS",
    "func": pert_extreme,
    "param": (2, 3),
    "level": 2,
    "info": "No value for event 3 found. Might influence data on occurrence."
}
pert_event_hz = {
    "col": "EVENT_HZ",
    "func": pert_extreme,
    "param": (2, 3),
    "level": 2,
    "info": "No value for event 3 found. Might influence data on occurrence."
}
pert_event_ic = {
    "col": "EVENT_IC",
    "func": pert_extreme,
    "param": (2, 3),
    "level": 2,
    "info": "No value for event 3 found. Might influence data on occurrence."
}

pert_event_ts = {
    "col": "EVENT_TS",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 1,
    "info": "Swap to neighbours."
}
pert_event_sn = {
    "col": "EVENT_SN",
    "func": pert_extreme,
    "param": (2, 3),
    "level": 2,
    "info": "No value for event 3 found. Might influence data on occurrence."
}
pert_event_ra = {
    "col": "EVENT_RA",
    "func": pert_ordinal_n,
    "param": 1,
    "level": 1,
    "info": "Swap to neighbours."
}
pert_1hour_pert = {
    "col": "1HOUR_PRECIPITATION(INCH)",
    "func": pert_resolution,
    "param": 0.02,
    "level": 3,
    "info": "Sensor resolution is 0.02 inch"
}
pert_sea_level_pressure = {
    "col": "SEA_LEVEL_PRESSURE(MILLIBAR)",
    "func": pert_resolution,
    "param": 0.7,
    "level": 3,
    "info": "Sensor resolution is 0.7 millibar"
}
pert_dep_delay = {
    "col": "DEP_DELAY(MINS)",
    "func": pert_ordinal_n,
    "param": 5,
    "level": 1,
    "info": "Default Switch. Low Priority."
}
pert_runway_error = {
    "col": "RUNWAY_ERROR(PERC)",
    "func": pert_resolution,
    "param": 0.2,
    "level": 3,
    "info": "Only values in 0.2 steps are permitted."
}

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

    pert_start, pert_start_string = data_manage_utils.print_time()
    print("Starting perturbation at: " + pert_start_string)

    data_copy = data.copy()
    data_copy["pert_id"] = None
    data_copy["level"] = None
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
            row["level"] = opt["level"]
            pert_lst.append(list(row))
            # row_df = pd.DataFrame(row)
            # pert_df = pert_df.append(row_df.T,ignore_index=True)
            for count, val in enumerate(opt["func"](row[opt["col"]], opt["param"]), start=1):
                row[opt["col"]] = val
                row["pert_id"] = row_pert_name
                row["level"] = opt["level"]
                pert_lst.append(list(row))
                # row_df = pd.DataFrame(row)
                # pert_df = pert_df.append(row_df.T,ignore_index=True)
            c += 1
        end, end_string = data_manage_utils.print_time(time_format="%Y_%m_%d %H:%M:%S")
        print("Ending run " + str(cc) + "/" + str(len(options)) + " at: " + end_string)
        cc += 1
    print("Converting list to pd.DataFrame...")
    pert_df = pd.DataFrame(pert_lst, columns=df_cols)
    pert_end, pert_end_string = data_manage_utils.print_time()
    print("Ending perturbation at: " + pert_end_string)
    print(f"Time elapsed: {pert_end - pert_start}")
    return pert_df


def check_column_diff(X: pd.DataFrame, options: list):
    arr = []
    for o in options:
        arr.append(o.get("col"))
    x_len = len(X.columns)
    diff = set(X.columns) - set(arr)
    neg_diff = set(arr) - set(X.columns)
    print(str(x_len - len(diff)) + "/" + str(x_len))
    print(f"diff: {diff}. neg-diff: {neg_diff}")


def save_pert_data(data: pd.DataFrame, out_path: str):
    fullname = os.path.join(out_path, "pert_out_df.pkl.gz")
    print(f"Shape of df: {data.shape}")
    print(f"Compressing perturbation outcome as .gzip")
    data.to_pickle(fullname, compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--folder")
    args = parser.parse_args()

    # LOAD MODEL DATA
    path = os.path.join(ROOT_PATH, "data/training/training_results", args.model, args.folder)
    print(f"Using model from path: {path}")

    # Check if model is built
    model_file_path = os.path.join(path, "model.joblib")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Cannot find model through path '{model_file_path}'.")

    # Load model from joblib
    model = joblib.load(model_file_path)

    # Load search settings params from json
    settings_path = os.path.join(path, "estimation_settings.json")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Cannot find settings file through path '{settings_path}'.")
    data_folder = data_manage_utils.find_data_path_by_settings_file(settings_path, ROOT_PATH)

    X_train, y_train, X_test, y_test = data_manage_utils.load_processed_data_by_folder(data_folder)

    scale_path = os.path.join(data_folder, "scaler.sav")
    do_scale = False
    if os.path.exists(scale_path):
        print("Model input is being scaled.")
        do_scale = True
        scaler = data_manage_utils.load_scaler_from_sav(scale_path)
        X_test_scaled = scaler.transform(X_test)

    y_pred = data_manage_utils.load_numpy_from_pickle(os.path.join(path, "y_test_pred.pkl"))

    # PREDICT BASED ON MODEL
    if do_scale:
        y_pred_check = model.predict(X_test_scaled)
    else:
        y_pred_check = model.predict(X_test)
    print(f"Accuracy score of model: {accuracy_score(y_test, y_pred_check)}")
    print(f"Shape of true labels: {y_test.shape} \n Shape of pred labels: {y_pred.shape}")

    # OPTION SELECTION
    options = [pert_dep_delay, pert_elapsed_time, pert_nr_prev_flights, pert_approach_speed, pert_tail_height,
               pert_parking_area, pert_winglets_yn, pert_temp, pert_dew_temp, pert_humidity, pert_wind_drct,
               pert_wind_speed, pert_1hour_pert, pert_sea_level_pressure, pert_vsbty, pert_event_br, pert_event_dz,
               pert_event_fg, pert_event_fu, pert_event_gr, pert_event_gs, pert_event_hz, pert_event_ic, pert_event_ra,
               pert_event_sn, pert_event_ts, pert_runway_error]

    # IDENTIFY NON SUITABLE OPTIONS
    idxs = []
    for i, o in enumerate(options):
        if o.get("col") in X_test.columns:
            idxs.append(i)
    options = [options[i] for i in idxs]

    # COLUMN CHECKING
    check_column_diff(X=X_test, options=options)

    # PERFORM PERTURBATION
    X_test["y_true"] = y_test
    pert_data = perturbate(X_test.iloc[:], OPT_THRESHOLD, options)
    print("Predicting on created dataframe")

    # PREDICT ON PERTURBATED DATA
    if do_scale:
        pert_data['y'] = model.predict(scaler.transform(pert_data[pert_data.columns[:-3]]))
    else:
        pert_data['y'] = model.predict(pert_data[pert_data.columns[:-3]])

    # SAVE DATA
    save_pert_data(data=pert_data, out_path=path)


if __name__ == "__main__":
    main()
