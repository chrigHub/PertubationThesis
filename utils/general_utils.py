import os
import glob
import numpy as np
import pandas as pd


def find_latest_folder(parent_dir: str) -> str:
    assert type(parent_dir) == str, f"parent_dir expected type 'str', got: {type(parent_dir)}"
    list_of_files = glob.glob(parent_dir + '/*')
    latest_folder = max(list_of_files, key=os.path.getmtime)
    list_of_files = glob.glob(latest_folder + "/*")
    latest_folder = max(list_of_files, key=os.path.getmtime)
    return latest_folder
