from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import os
import importlib
import sys

sys.path.insert(0, '../../..')
from main.utils import train_utils, data_manage_utils

importlib.reload(data_manage_utils)
importlib.reload(train_utils)

path = None
# path = "./training_results/RF/2022_11_04-2350"

if path is None:
    path = [x[0] for x in os.walk("training_results/MLP")][-1:][0]

X_train_scaled = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/X_train_scaled.pkl")
y_train = data_manage_utils.load_numpy_from_pickle("./processed_files/NEW/y_train.pkl")

grid = {"hidden_layer_sizes": [(n,) for n in range(300, 1000, 100)],
        "max_iter": [x for x in range(200, 1000, 100)],
        "epsilon": [1e-7, 1e-8, 1e-9]
        }

print("Print of grid: ")
print(grid)

# START VARIABLES
sm = SMOTE(random_state=42)
verbosity = 10
nr_jobs = 6
score_name = "balanced_accuracy"
kf = KFold(n_splits=3)
nr_runs = 300
clf = MLPClassifier(random_state=42, activation="relu", solver="adam", early_stopping=True)
# END VARIABLES


start, start_string = data_manage_utils.print_time()
print("Start time: " + start_string)

imba_pipeline = make_pipeline(sm, clf)

new_params = {'mlpclassifier__' + key: grid[key] for key in grid}
print(f"New params: {new_params}")

grid_search = GridSearchCV(imba_pipeline, param_grid=new_params, n_jobs=nr_jobs,
                           verbose=verbosity, cv=kf, scoring=score_name, return_train_score=True)
grid_search.fit(X_train_scaled, y_train)

end, end_string = data_manage_utils.print_time()
print("End time: " + end_string)
print("Time elapsed: " + str(end - start))

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best params: " + str(best_params))
print(f"Best score: {best_score}")

time, time_string = data_manage_utils.print_time(time_format="%Y_%m_%d-%H%M")
out_dir = "../training/training_results/MLP/" + time_string
data_manage_utils.save_search_params(out_dir=out_dir, param_dict=best_params, filename="params_final.txt")
