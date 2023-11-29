import numpy as np

# TODO
# Add function for retrieving rest of dicts
class DictManager:
    def load_param_grid_dict(self,key : str):
        assert type(key) == str, f"Expected type of key is 'str'. Got {type(key)}"
        params_dict = {
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
        assert key in params_dict.keys(), f"Key '{key}' is not allowed. List of allowed keys: {params_dict.keys()}"
        return params_dict.get(key)