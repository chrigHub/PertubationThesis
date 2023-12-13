import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# Add function for retrieving rest of dicts
class ParamEstimationManager:
    __base_clf_opt_dicts__ = {
        "RF": {
            "clf": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "SH": {
                "min_resources": 25,
                "resource": "n_estimators",
                "max_resources": 1600,
            },
            "EG": {
            }
        },
        "KNN": {
            "clf": KNeighborsClassifier(algorithm="ball_tree"),
            "SH": {
                "min_resources": 100,
                "resource": "n_samples",
                "max_resources": "auto",
            },
            "EG": {
            }
        }
    }

    __param_grid_dicts__ = {
        "RF": {
            "n_estimators": [int(x) for x in np.linspace(start=25, stop=1600, num=20)],
            "max_depth": [1, 2, 3, None],
            "max_features": ["sqrt", 0.3, 0.6, 0.9, 1.0],
            "min_samples_split": [2, 0.1, 0.2, 0.3],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        },
        "KNN": {
            "n_neighbors": [int(x) for x in np.linspace(start=5, stop=60, num=20)],
            "weights": ["uniform", "distance"],
            "leaf_size": [int(x) for x in np.linspace(start=10, stop=100, num=20)],
            "p": [1, 2]
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

        # Create estimator instance
        self.param_estimator = self.load_param_estimator(est_type=est_type, param_dict=est_params)

        # Define names
        self.est_alg_name = self.__est_alg_dict__.get(est_type).get("name")

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
