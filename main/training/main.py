import DataHolder
from DataHolder import DataHolder


def load_and_scale_data():
    data_holder = DataHolder("processed_files/B")
    data_holder.scale_data()
    data_holder.print_data_stats()
    return data_holder


def main():

    data_holder = load_and_scale_data()


if __name__ == "__main__":
    main()
