import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

ROOT_PATH = "../.././"
SEPERATOR = 20 * "=" + "{}" + 20 * "="

from main.utils import data_manage_utils


# Add function for retrieving rest of dicts
class ParamEstimationManager:
    __base_clf_opt_dicts__ = {
        "RF": {
            "clf": RandomForestClassifier(random_state=42, class_weight="balanced", bootstrap=True),
            "SH": {
                "min_resources": 25,
                "resource": "n_estimators",
                "max_resources": 1000,
            },
            "EG": {
            }
        },
        "KNN": {
            "clf": KNeighborsClassifier(algorithm="ball_tree"),
            "SH": {
                "min_resources": 3000,
                "resource": "n_samples",
                "max_resources": "auto",
            },
            "EG": {
            }
        },
        "RFR": {
            "clf": RandomForestClassifier(random_state=42),
            "SH": {
                "min_resources": 25,
                "resource": "n_samples",
                "max_resources": 1000,
            },
            "EG": {
            }
        },
        "ADAB": {
            "clf": AdaBoostClassifier(random_state=42),
            "SH": {
                "min_resources": 25,
                "resource": "n_estimators",
                "max_resources": 1000
            },
            "EG": {
            }
        },
        "XGB": {
            "clf": GradientBoostingClassifier(random_state=42),
            "SH": {
                "min_resources": 25,
                "resource": "n_estimators",
                "max_resources": 1000
            },
            "EG": {
            }
        }
    }

    __param_grid_dicts__ = {
        "RF": {
            "n_estimators": [int(x) for x in np.arange(start=1000, stop=1100, step=100)],
            "max_depth": [x for x in range(16, 21, 3)],
            #"max_features": [0.3, 0.5, 0.7, 1.0]
            #"max_samples": [0.3, 0.5, 0.7, 1.0],
            #"min_samples_split": [2, 0.1, 0.2, 0.3, 0.5],
            #"criterion": ["gini", "entropy"]
        },
        "KNN": {
            "n_neighbors": [int(x) for x in np.linspace(start=5, stop=60, num=20)],
            "weights": ["uniform", "distance"],
            "leaf_size": [int(x) for x in np.linspace(start=10, stop=100, num=20)],
            "p": [1, 2]
        },
        "RFR": {
            "n_estimators": [int(x) for x in np.linspace(start=20, stop=500, num=20)],
            "max_depth": [3, 4, 5],
            "max_features": ["sqrt", 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
            "min_samples_split": [2, 0.1, 0.2, 0.3, 0.5],
            "criterion": ["friedman_mse"]
        },
        "ADAB": {
            "n_estimators": [int(x) for x in np.arange(start=600, stop=1000, step=100)],
            "learning_rate": [float(x) for x in np.arange(start=0.1, stop=1.2, step=0.2)]
        },
        "XGB": {
            "n_estimators": [int(x) for x in np.arange(start=400, stop=700, step=100)],
            "learning_rate": [float(x) for x in np.arange(start=0.1, stop=1.2, step=0.5)],
            "max_depth": [x for x in range(3, 15, 2)]
        }
    }

    __est_alg_dict__ = {
        "SH": {
            "name": "Successive Halving",
            "func": HalvingGridSearchCV,
            "params": {
                "factor": 2
            }
        },
        "EG": {
            "name": "Exhaustive Grid",
            "func": GridSearchCV,
            "params": {
            }
        }
    }

    def __init__(self, clf_type: str, est_type: str, est_params: dict = None):
        if est_params is None:
            est_params = {}

        # Load fixed dictionaries
        self.param_grid = self.load_param_grid_dict(clf_type)
        self.base_clf, self.est_params = self.load_base_clf_and_est_params(clf_type=clf_type, est_type=est_type)
        self._result_doc_dict = {}

        # Remove resource from param_grid if resource is in param_grid is used
        if est_type == "SH":
            resource = self.est_params.get("resource")
            if resource in self.param_grid.keys():
                self.param_grid.pop(resource)

        # Add specific params to general estimator params
        specific_params = {
            "estimator": self.base_clf,
            "param_grid": self.param_grid
        }
        est_params.update(specific_params)
        self.est_params.update(est_params)

        # Create estimator instance
        self.param_estimator = self.load_param_estimator(est_type=est_type, param_dict=self.est_params)

        # Define names
        self.est_type_key = est_type
        self.est_alg_name = self.__est_alg_dict__.get(est_type).get("name")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        assert isinstance(X_train, np.ndarray), f"Expected type of key is 'np.ndarray'. Got {type(X_train)}"
        assert isinstance(y_train, np.ndarray), f"Expected type of key is 'np.ndarray'. Got {type(y_train)}"

        # Getting start time of estimator process
        start, start_str = data_manage_utils.print_time()
        print(f"Starting estimation at {start_str}\n")

        # Perform parameter estimation fit
        self.param_estimator.fit(X_train, y_train)

        # Getting end time of estimator process
        end, end_str = data_manage_utils.print_time()
        print(f"Ending estimation at {end_str} after {end - start} time")
        print(SEPERATOR)

        best_params = self.param_estimator.best_params_
        best_score = self.param_estimator.best_score_

        # Update base clf dict with best params
        params = self.base_clf.get_params()
        params.update(best_params)

        doc_dict = {
            "time_start": start_str,
            "time_end": end_str,
            "time_needed": str(end - start),
            "clf_name": self.base_clf.__class__.__name__,
            "clf_module": self.base_clf.__class__.__module__,
            "best_params": params,
            "score_metric": self.param_estimator.get_params().get("scoring"),
            "score_value": best_score
        }

        # Print statistics
        if self.est_type_key == "SH":
            n_resources = self.param_estimator.n_resources_
            n_candidates = self.param_estimator.n_candidates_
            print(f"Number of resources: {n_resources}")
            print(f"Number of candidates: {n_candidates}")
            doc_dict.update({
                "n_resources": n_resources,
                "n_candidates": n_candidates
            })
        print(f"Best params: {best_params}")
        print(f"Best score: {best_score}")

        self._result_doc_dict = doc_dict

        return self._result_doc_dict

    def load_param_grid_dict(self, key: str):
        assert isinstance(key, str), f"Expected type of key is 'str'. Got {type(key)}"
        assert key in self.__param_grid_dicts__.keys(), f"Key '{key}' is not allowed. List of allowed keys: {self.__param_grid_dicts__.keys()}"
        return self.__param_grid_dicts__.get(key)

    def load_base_clf_and_est_params(self, clf_type: str, est_type: str):
        assert isinstance(clf_type, str), f"Expected type of clf_type is 'str'. Got {type(clf_type)}"
        assert isinstance(est_type, str), f"Expected type of est_type is 'str'. Got {type(est_type)}"
        assert clf_type in self.__base_clf_opt_dicts__.keys(), f"Key '{clf_type}' is not allowed. List of allowed keys: {self.__base_clf_opt_dicts__.keys()}"

        base_clf_opt = self.__base_clf_opt_dicts__.get(clf_type)
        base_clf = base_clf_opt.get("clf")
        assert est_type in base_clf_opt.keys(), f"Key '{est_type}' is not allowed. List of allowed keys: {base_clf_opt.keys()}"
        est_params = base_clf_opt.get(est_type)
        return base_clf, est_params

    def load_param_estimator(self, est_type: str, param_dict: dict):
        assert isinstance(est_type, str), f"Expected type of est_type is 'str'. Got {type(est_type)}"
        assert isinstance(param_dict, dict), f"Expected type of param_dict is 'dict'. Got {type(param_dict)}"
        assert est_type in self.__est_alg_dict__.keys(), f"Key '{est_type}' is not allowed. List of allowed keys: {self.__est_alg_dict__.keys()}"

        est_dict = self.__est_alg_dict__.get(est_type)
        func = est_dict.get("func")
        set_params = est_dict.get("params")
        estimator = func(**param_dict, **set_params)
        return estimator

    def get_param_doc_dict(self):
        clf_str = re.sub(' {2,}', ' ', str(self.base_clf).replace("\n", ""))
        search_method_dict = {
            "method_type": str(type(self.param_estimator)),
            "method_params": data_manage_utils.params_to_string_dict(self.param_estimator.get_params())
        }
        param_dict = {
            "base_clf": clf_str,
            "param_grid": self.param_grid,
            "search_method": search_method_dict
        }
        return param_dict
