import os.path
import numpy as np
import argparse
import sys
import pandas as pd
import joblib
import importlib

import sklearn.ensemble
from sklearn.metrics import accuracy_score

ROOT_PATH = "../../."
DATA_FOLDER = os.path.join(ROOT_PATH, "data/training/training_results")
SEPERATOR = 20 * "=" + "{}" + 20 * "="

sys.path.insert(0, ROOT_PATH)
from main.utils import data_manage_utils


def create_classifier_instance(module_name, clf_name, params):
    """
    Dynamically imports the specified module and creates an instance of the specified classifier.

    :param module_name: The module containing the model.
    :param clf_name: The name of the model class.
    :param params: The dictionary of parameters for the model
    :return: An instance of the specified classifier.
    """
    try:
        module = importlib.import_module(module_name)
        clf_class = getattr(module, clf_name)
        return clf_class(**params)
    except ModuleNotFoundError:
        raise ImportError(f"Module {module_name} not found.")
    except AttributeError:
        raise ImportError(f"Class {clf_name} not found in module {module_name}.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the classifier instance: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    parser.add_argument("--folder")
    args = parser.parse_args()

    # Creating path to file.
    folder = os.path.join(DATA_FOLDER, args.type, args.folder)
    settings_path = os.path.join(folder, "estimation_settings.json")

    # Check if file exists
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Path '{settings_path}' does not exist. Program will end.")

    dict = data_manage_utils.read_search_params(settings_path)

    # Load training data
    data_info = dict.get("data_info")
    data_folder = data_info.get("data_folder")
    X_train, y_train, X_test, y_test = data_manage_utils.load_processed_data_by_folder(data_folder)

    # Load model instance and parameters
    result_info = dict.get("result_info")
    clf_name = result_info.get("clf_name")
    module_name = result_info.get("clf_module")
    model_params = result_info.get("best_params")

    model = create_classifier_instance(module_name=module_name, clf_name=clf_name, params=model_params)

    # Scale if scaler.sav is in folder
    scaler_path = os.path.join(data_folder, "scaler.sav")
    if os.path.exists(scaler_path):
        print("Scaling Data...")
        scaler = data_manage_utils.load_scaler_from_sav(scaler_path)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Convert to numpy
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.to_numpy()
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.to_numpy()
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.to_numpy()
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.to_numpy()

    # Train model on train data
    print(f"Fitting model\n{model}")
    model.fit(X_train, y_train)

    # Print accuracy on evaluation for checking
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Predicted accuracy score is: {acc:.4f}")

    # Outputting model file as joblib
    print(f"Saving model as .joblib...")
    out_model_path = os.path.join(folder, "model.joblib")
    joblib.dump(model, out_model_path, compress=("zlib", 3))
    print(f"Model Size: {np.round(os.path.getsize(out_model_path) / 1024 / 1024, 2)} MB")


if __name__ == "__main__":
    main()
