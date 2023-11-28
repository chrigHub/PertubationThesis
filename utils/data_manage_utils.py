import joblib
import numpy as np
import pandas as pd
import datetime
import glob
import os
import re
import ast
import json


def validate_time(date_string, format: str = '%m-%d-%Y %H:%M:%S'):
    ret = True
    try:
        time = datetime.datetime.strptime(date_string, format)
        return ret, time
    except ValueError:
        ret = False
        return ret, None


def col_stats_to_string(df: pd.DataFrame, attr_names: [] = []):
    if not attr_names:
        attr_names = df.columns
    ret = ""
    for attr in attr_names:
        ret = ret + "\n" + 20 * "=" + attr + 20 * "="
        ret = ret + "\n" + "Datatype of attribute: " + str(df[attr].dtype)
        nr_of_unique = len(df[attr].unique())
        nr_of_null = df[attr].isna().sum()
        ret = ret + "\n" + "Number of null values: " + str(nr_of_null)
        ret = ret + "\n" + "Number of unique values: " + str(nr_of_unique)
        ret = ret + "\n" + "Doubled values: " + str(len(df) - nr_of_null - nr_of_unique)
        if nr_of_unique < 30:
            ret = ret + "\n" + "Unique values: " + str(df[attr].unique())
        if df[attr].dtype == 'O':
            is_time, _ = validate_time(df[attr][df[attr].notna()].iloc[0])
            if is_time:
                date_series = df[attr].apply(lambda x: validate_time(x)[1])
                ret = ret + "\n" + "Range: [" + str(date_series.min()) + ";" + str(date_series.max()) + "]"
            else:
                ret = ret + "\n" + "Range: char (" + str(df[attr].str.len().min()) + "-" + str(
                    df[attr].str.len().max()) + ")"
        else:
            ret = ret + "\n" + "Range: [" + str(df[attr].min()) + ";" + str(df[attr].max()) + "]"
    return ret


def read_csv_from_subfolder(path):
    if path:
        all_files = glob.glob(path)
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, on_bad_lines='skip')
            li.append(df)
        return pd.concat(li, axis=0, ignore_index=True)
    else:
        return None


def read_table_from_subfolder(path, delim: str = ','):
    if path:
        all_files = glob.glob(path)
        li = []
        for filename in all_files:
            df = pd.read_table(filename, delimiter=delim)
            li.append(df)
        return pd.concat(li, axis=0, ignore_index=True)
    else:
        return None


def read_pickle_from_subfolder(path):
    if path:
        all_files = glob.glob(path)
        li = []
        for filename in all_files:
            df = pd.read_pickle(filename)
            li.append(df)
        return pd.concat(li, axis=0, ignore_index=True)
    else:
        return None


def read_excel_from_subfolder(path):
    if path:
        all_files = glob.glob(path)
        li = []
        for filename in all_files:
            df = pd.read_excel(filename)
            li.append(df)
        return pd.concat(li, axis=0, ignore_index=True)
    else:
        return None


def save_numpy_to_pickle(array: np.ndarray, filepath: str):
    with open(filepath, 'wb') as f:
        np.save(f, array)


def load_numpy_from_pickle(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        arr = np.load(f, allow_pickle=True)
    return arr


def load_scaler_from_sav(filepath: str):
    return joblib.load(filepath)


def print_time(time_format: str = "%Y_%m_%d %H:%M") -> (datetime, str):
    time = datetime.datetime.now()
    return time, time.strftime(time_format)


def find_best_scores(dict_list: list, score_name: str):
    assert type(score_name) == str, f"str type expected, got: {type(score_name)}"
    assert type(dict_list) == list, f"dict type expected, got: {type(dict_list)}"
    assert dict_list, "parameter 'dict_list' is empty"

    for count, entry in enumerate(dict_list):
        assert type(entry) == tuple, f"tuple type expected, got: {type(entry)}"
        if count == 0:
            best_tuple = entry
        elif entry[1].get(score_name) > best_tuple[1].get(score_name):
            best_tuple = entry
    return best_tuple


def save_search_params(out_dir: str, param_dict: dict, filename: str = "params.txt"):
    assert type(param_dict) == dict, f"dict type expected, got: {type(param_dict)}"
    filename = os.path.join(out_dir, filename)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        assert os.path.exists(out_dir), f"Folder {out_dir} could not be created"
    with open(filename, 'w') as data:
        data.write(json.dumps(param_dict))


def params_to_string_dict(params: dict):
    assert type(params) == dict, f"dict type expected, got: {type(params)}"
    ret = {}
    for k, v in params.items():
        ret.update({k : str(v)})
    return ret


def tryval(val):
    try:
        val = ast.literal_eval(val)
    except:
        pass
    return val


def load_params_from_txt(filepath: str):
    param_dict = {}
    with open(filepath, "r") as f:
        for line in f:
            line = re.sub('[ ]', '', line)[:-1]
            line = line.split(":")
            _d = {line[0]: tryval(line[1])}
            param_dict.update(_d)
    return param_dict
