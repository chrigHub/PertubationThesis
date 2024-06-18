import argparse
import joblib
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    args = parser.parse_args()

    file = args.file
    model = joblib.load(file)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
