import sys
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

ROOT_FOLER = os.path.abspath("../../../../")
sys.path.insert(0, ROOT_FOLER)
INPUT_FOLDER = os.path.join(ROOT_FOLER, "data/preprocessing/base/class")

from main.utils.data_manage_utils import load_processed_data_by_folder
from main.utils.time_utils import print_time

def main():
    X_train, y_train, X_test, y_test = load_processed_data_by_folder(INPUT_FOLDER)

    # Define your classifier and variables
    clf = RandomForestClassifier(class_weight="balanced", random_state=42, max_depth=12)
    results = []
    n_folds = 5

    remaining_columns = list(X_train.columns)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    # Establishing combined features for deletion
    buddies = [("ARR_MIN_OF_DAY_SIN", "ARR_MIN_OF_DAY_COS"), ("ARR_DAY_SIN", "ARR_DAY_COS")]

    print("Starting feature elimination: ")
    start_col_n = X_train.shape[1]
    done = False
    while not done:
        print(f"{X_train.shape[1]}/{start_col_n} columns left.")
        # Get accuracy using 5-fold cross-validation
        accuracy = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
        print(f"\tAccuracy of run: {accuracy:.2f}")

        # Fit classifier
        start, _ = print_time()
        clf.fit(X_train, y_train)
        end, _ = print_time()
        time = end - start
        print("\tFit time: ", time)

        # Calculating feature importances
        feature_importances = clf.feature_importances_

        # Safe results
        results.append((len(remaining_columns), accuracy, remaining_columns.copy(), time))

        # Determine least important feature
        least_important_feature_index = np.argmin(feature_importances)
        least_important_feature_importance = feature_importances[least_important_feature_index]

        # Remove least important feature from dataset and column names
        X_train = np.delete(X_train, least_important_feature_index, axis=1)
        removed_column = remaining_columns.pop(least_important_feature_index)
        print(f"\tRemoving column: '{removed_column}'")

        # Check if one of the buddy columns was deleted. Drop second if happened.
        if any([removed_column in pair for pair in buddies]):
            buddy_dict = {col1: col2 for col1, col2 in buddies}
            buddy_dict.update({col2: col1 for col1, col2 in buddies})
            buddy = buddy_dict.get(removed_column)
            buddy_index = remaining_columns.index(buddy)
            remaining_columns.pop(buddy_index)
            X_train = np.delete(X_train, buddy_index, axis=1)

        # Anchor for while
        if X_train.shape[1] <= 1:
            done = True

    print(f"Elimination is done with accuracy array of:\n{[res[1] for res in results]}")

    result_df = pd.DataFrame(results)
    result_df.to_pickle("result_df.pkl")

if __name__ == '__main__':
    main()