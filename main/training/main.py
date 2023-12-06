import os.path
import DataHolder
import ParamEstimationManager
import argparse
from DataHolder import DataHolder
from ParamEstimationManager import ParamEstimationManager

ROOT_PATH = "../.././"


def load_and_scale_data(data_path):
    data_holder = DataHolder(data_path)
    data_holder.scale_data()
    data_holder.print_data_shapes()
    data_holder.print_data_stats()
    return data_holder

def configure_param_search():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv")
    parser.add_argument("--dataspec")
    parser.add_argument("--clftype")
    parser.add_argument("--estalg")
    parser.add_argument("-v", "--verbose")
    args = parser.parse_args()

    #Transforming parameter inputs to dict
    est_params = {
        "cv" : args.est_cv,
        "scoring" : args.est_scoring,
        "n_jobs" : args.est_njobs,
        "verbose" : args.verbose
    }

    #Loading data
    data_path = os.path.join(ROOT_PATH, "data/preprocessing/processed_files")
    data_path = os.path.join(data_path, args.dataspec)
    data_holder = load_and_scale_data(data_path)

    #Creating estimator instance
    estimationManager = ParamEstimationManager(clf_type = args.clf_type, est_type = args.est_type, est_params = est_params)
    estimator = estimationManager.param_estimator
    print(estimator)



if __name__ == "__main__":
    main()
