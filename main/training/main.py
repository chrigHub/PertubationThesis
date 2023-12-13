import os.path
import data
import estimation
import argparse
import sys
import importlib
import importlib
from data import DataHolder
from estimation import ParamEstimationManager

ROOT_PATH = "../.././"
SEPERATOR = 20 * "=" + "{}" + 20 * "="

importlib.reload(data)
importlib.reload(estimation)


def load_scale_and_print_data(data_path):
    data_holder = DataHolder(data_path)
    data_holder.scale_data()
    data_holder.print_data_shapes()
    data_holder.print_data_stats()
    return data_holder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataspec")
    parser.add_argument("--clf_type")
    parser.add_argument("--est_type")
    parser.add_argument("--est_cv")
    parser.add_argument("--est_scoring")
    parser.add_argument("--est_njobs")
    parser.add_argument("--est_verbose")
    args = parser.parse_args()

    # Transforming parameter inputs to dict
    est_params = {
        "cv": args.est_cv,
        "scoring": args.est_scoring,
        "n_jobs": args.est_njobs,
        "verbose": args.est_verbose
    }

    # Loading data
    data_path = os.path.join(ROOT_PATH, "data/preprocessing/processed_files")
    data_path = os.path.join(data_path, args.dataspec)
    data_holder = load_scale_and_print_data(data_path)
    X_train, y_train, X_test, y_test = data_holder.get_data_as_numpy_scaled()

    # Creating estimator instance
    estimation_manager = ParamEstimationManager(clf_type=args.clf_type, est_type=args.est_type, est_params=est_params)
    print(SEPERATOR.format("Starting " + estimation_manager.est_alg_name))
    estimator = estimation_manager.param_estimator
    print(estimator)


if __name__ == "__main__":
    main()
