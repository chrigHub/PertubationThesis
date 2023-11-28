import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import random
import importlib
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import get_scorer, plot_confusion_matrix, make_scorer, accuracy_score, precision_score, \
    recall_score, f1_score
from utils import data_manage_utils


def do_smth(scorer_name, y_true, y_pred):
    print(scorer_name)
    print(get_scorer(scorer_name))
    return get_scorer(scorer_name)._score_func(y_true=y_true, y_pred=y_pred)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    assert type(y_true) == np.ndarray, f"y_true expected {np.ndarray}. Got {type(y_true)}"
    assert type(y_pred) == np.ndarray, f"y_pred expected {np.ndarray}. Got {type(y_pred)}"

    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
    prec_score = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    rec_score = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_score = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    dict = {
        "accuracy" : acc_score,
        "precision_macro" : prec_score,
        "recall_macro" : rec_score,
        "f1_macro" : f1_score
    }
    return dict


def make_raveled(y: np.ndarray):
    if y.shape != (len(y),):
        y = y.ravel()
    return y


def calc_mean_dict_scores(scores: dict):
    for key, val in scores.items():
        scores.update({key: val.mean()})
    return scores


def cross_validate(clf, X: np.ndarray, y: np.ndarray, n_folds: int = 5,
                   scoring=["balanced_accuracy", "f1_weighted", "precision_weighted", "recall_weighted"], sampler=None,
                   n_jobs: int = 1, verbosity: int = 0):
    start, start_str = data_manage_utils.print_time()
    print("Starting cross_validation at: " + start_str)
    make_raveled(y)
    cv = StratifiedKFold(n_splits=n_folds)
    if not sampler:
        print("Not using sampler for cross val")
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbosity)
    else:
        print("Using sampler {} for cross val".format(sampler))
        scores = {}
        count = 1
        for train_fold_idx, val_fold_idx in cv.split(X, y):
            print("At step {}/{} splits.".format(count, n_folds))
            X_train, y_train = X[train_fold_idx], y[train_fold_idx]
            X_val, y_val = X[val_fold_idx], y[val_fold_idx]
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            model_obj = clf.fit(X_train, y_train)
            for score in scoring:
                if (score == "balanced_accuracy"):
                    s = get_scorer(score)._score_func(y_val, model_obj.predict(X_val))
                else:
                    s = get_scorer(score)._score_func(y_val, model_obj.predict(X_val), average="weighted")
                if scores.get(score):
                    arr = scores.get(score)
                    arr.append(s)
                    scores.update({score: arr})
                else:
                    scores.update({score: [s]})
            count += 1
        for k, v in scores.items():
            scores.update({k: np.array(v)})
    end, end_str = data_manage_utils.print_time()
    print("Ending cross_validation fit at: " + end_str)
    print("Time elapsed: " + str(end - start))
    return scores


def evaluate_on_testset(clf, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                        figsize: tuple = (10, 5)):
    start, start_str = data_manage_utils.print_time()
    print("Starting model fit at: " + start_str)
    y_train = make_raveled(y_train)
    y_test = make_raveled(y_test)
    clf.fit(X_train, y_train)
    end, end_str = data_manage_utils.print_time()
    print("Ending model fit at: " + end_str)
    print("Time elapsed: " + str(end - start))
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].grid(visible=False)
    axes[1].grid(visible=False)
    plot_confusion_matrix(clf, X_test, y_test, ax=axes[0], cmap="cividis")
    plot_confusion_matrix(clf, X_test, y_test, normalize='true', ax=axes[1], cmap="cividis")
    y_pred = clf.predict(X_test)
    plt.show()
    return y_pred


def custom_random_search(model_func, X: np.ndarray, y: np.ndarray, random_grid: dict, nr_iter: int, n_folds: int = 5,
                         scoring=["balanced_accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
                         sampler=None, n_jobs: int = 1, verbosity: int = 0):
    print("Running random search with {} runs.".format(nr_iter * n_folds))
    start, start_str = data_manage_utils.print_time()
    print("Starting random search at: " + start_str)
    list_of_scores = []
    for i in range(nr_iter):
        print("CALCULATING GRID-RUN {}/{}.".format(i + 1, nr_iter))
        params = {}
        for key, val in random_grid.items():
            params.update({key: random.choice(val)})
        print(f"Current parameters {params}")
        clf = model_func(**params)
        scores = cross_validate(clf=clf, n_jobs=n_jobs, n_folds=n_folds, scoring=scoring, verbosity=verbosity, y=y, X=X,
                                sampler=sampler)
        mean_scores = calc_mean_dict_scores(scores)
        print(f"Mean score of run: Bal-Acc = {mean_scores.get('balanced_accuracy')}")
        list_of_scores.append((params, mean_scores))
    end, end_str = data_manage_utils.print_time()
    print("Ending random search at: " + end_str)
    print("Time elapsed: " + str(end - start))
    return list_of_scores
