import os.path
import numpy as np
import argparse
import sys
import pandas as pd
import importlib

ROOT_PATH = "../../."
SEPERATOR = 20 * "=" + "{}" + 20 * "="

sys.path.insert(0, ROOT_PATH)
from data import DataHolder
from estimation import ParamEstimationManager
from main.utils import data_manage_utils


def load_scale_and_print_data(data_path):
    data_holder = DataHolder(data_path)
    data_holder.scale_data()
    data_holder.print_data_shapes()
    data_holder.print_data_stats()
    return data_holder


def perform_estimation(X_train, y_train, args):
    # Transforming parameter inputs to dict
    est_params = {
        "cv": args.est_cv,
        "scoring": args.est_scoring,
        "n_jobs": args.est_njobs,
        "verbose": args.est_verbose
    }

    # Creating estimator instance
    estimation_manager = ParamEstimationManager(clf_type=args.clf_type, est_type=args.est_type, est_params=est_params)
    print(SEPERATOR.format("Starting " + estimation_manager.est_alg_name))

    # Fitting estimator
    result_doc_dict = estimation_manager.fit(X_train, y_train)

    print(SEPERATOR.format("Ending " + estimation_manager.est_alg_name))
    return estimation_manager, result_doc_dict


#def predict_clf(data_holder: DataHolder, estimation_manager: ParamEstimationManager):
#    print(SEPERATOR.format("START TRAINING"))
#
#    X_train, y_train, X_test, y_test = data_holder.get_data_as_numpy_scaled()
#    est = estimation_manager.param_estimator
#
#    y_train_pred = est.predict(X_train)
#    y_test_pred = est.predict(X_test)
#
#    print(SEPERATOR.format("EMD TRAINING"))
#    return y_train_pred, y_test_pred


def create_est_output_files(args: dict, data_holder: DataHolder, estimation_manager: ParamEstimationManager,
                            result_doc_dict: dict, y_train_pred: np.ndarray, y_test_pred: np.ndarray):
    # Create storage folder
    run_time = data_manage_utils.print_time("%Y_%m_%d-%H%M")[1]
    output_folder = os.path.join(ROOT_PATH, "data/training/training_results")
    output_folder = os.path.join(output_folder, args.clf_type, run_time)
    if not os.path.exists(output_folder):
        print(f"Creating folder '{output_folder}'")
        os.makedirs(output_folder)

    print("\nSaving cv_results ...")
    # Write cv_results to output folder
    cv_results_df = pd.DataFrame(estimation_manager.param_estimator.cv_results_)
    cv_results_path = os.path.join(output_folder, "cv_results.pkl")
    cv_results_df.to_pickle(cv_results_path)

    print("\nSaving result dict ...")
    final_dict = {
        "result_info": result_doc_dict,
        "data_info": data_holder.get_data_doc_dict(),
        "est_info": estimation_manager.get_param_doc_dict()
    }

    print("\nSaving search param json file...")
    # Write result_doc_dict to output folder
    data_manage_utils.save_search_params(output_folder, final_dict, "estimation_settings.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataspec")
    parser.add_argument("--clf_type")
    parser.add_argument("--est_type")
    parser.add_argument("--est_cv", type=int)
    parser.add_argument("--est_scoring")
    parser.add_argument("--est_njobs", type=int)
    parser.add_argument("--est_verbose", type=int)
    args = parser.parse_args()

    # Loading data
    data_path = os.path.join(ROOT_PATH, "data/preprocessing/processed_files")
    data_path = os.path.join(data_path, args.dataspec)
    data_holder = load_scale_and_print_data(data_path)
    X_train, y_train, X_test, y_test = data_holder.get_data_as_numpy_scaled()

    # Perform estimation
    estimation_manager, result_doc_dict = perform_estimation(X_train, y_train, args)

    # Create documentation files
    create_est_output_files(args, data_holder, estimation_manager, result_doc_dict)


if __name__ == "__main__":
    main()
