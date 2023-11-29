import os.path
import DataHolder
import argparse
from DataHolder import DataHolder

ROOT_PATH = "../.././"


def load_and_scale_data(data_path):
    data_holder = DataHolder(data_path)
    data_holder.scale_data()
    data_holder.print_data_shapes()
    data_holder.print_data_stats()
    return data_holder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv")
    parser.add_argument("--dataspec")
    parser.add_argument("--clftype")
    parser.add_argument("--estalg")
    parser.add_argument("-v", "--verbose")
    args = parser.parse_args()
    print(args)

    data_path = os.path.join(ROOT_PATH, "data/preprocessing/processed_files")
    data_path = os.path.join(data_path, args.dataspec)
    data_holder = load_and_scale_data(data_path)


if __name__ == "__main__":
    main()
