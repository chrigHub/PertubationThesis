import logging

import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import get_scorer, plot_confusion_matrix, accuracy_score, precision_score
from main.utils import time_utils
from typing import Union


def make_raveled(y: np.ndarray):
    """
    Check if numpy nd-array is in need to be raveled and do so.
    :param y: Numpy nd-array.
    :return: Flattened numpy array.
    """
    if y.shape != (len(y),):
        y = y.ravel()
    return y


def calc_mean_result_stats(scores: dict):
    """
    Get the cv scores from the custom_cross_validate function and
    return the mean and std of the dictionaries contained.

    :param scores: cross validation results from custom_cross_validate.
    :return: result dict with mean and std values.
    """
    ret_dict = {}
    # Loading Train and Val Scores
    for key in scores.keys():
        score_dict = scores.get(key)
        sub_results = {}
        for k, v in score_dict.items():
            sub_results.update({k: {
                "mean": np.mean(v),
                "std": np.std(v)
            }})
        ret_dict.update({key: sub_results})
    return ret_dict


def custom_cross_validate(clf, X: np.ndarray, y: np.ndarray, n_folds: int = 5,
                          scoring: list = ["accuracy"], verbosity: int = 1, return_train_scores: bool = False):
    """
        Performs n-fold cross validation

        :param clf: Classifier object used for cross validation
        :param y: Target vector y for training and validation.
        :param X: Input matrix X for training and validation
        :param n_folds: Number of folds. Default: 5
        :param scoring: List of scoring identifiers. Default: ['accuracy']
        :param verbosity: Integer identifying verbosity of cross validation
            Values > 0 print status of validation. Default: 1
        :param return_train_scores: Boolean value to return train scores in addition to validation scores. Default. False

        :return: Returns dictionary of scoring results in following form ->
            {
                'val_scores': {
                    'score1': [s1, s2, s3, s4, s5],
                    'score2': [s1, s2, s3, s4, s5]
                },
                train_scores: {
                    'score1': [s1, s2, s3, s4, s5],
                    'score2': [s1, s2, s3, s4, s5]
                },
            }
    """

    if scoring is None:
        scoring = ["accuracy"]
    start, start_str = time_utils.print_time()
    if verbosity > 0:
        print("Starting cross_validation at: " + start_str)

    # Make y and X into numpy with correct forms
    if not isinstance(X, np.ndarray):
        warnings.warn("WARN: Input matrix is not of type 'np.ndarray'. Converting input matrix 'X' into np.ndarray.")
        X = X.to_numpy()
    if not isinstance(y, np.ndarray):
        warnings.warn("WARN: Input matrix is not of type 'np.ndarray'. Converting target vector 'y' into np.ndarray.")
        y = y.to_numpy()
    make_raveled(y)

    # Prepare parameters for cv
    cv = StratifiedKFold(n_splits=n_folds)
    train_scores = {}
    val_scores = {}
    count = 1

    # Cross validation loop
    fit_times = []
    v_pred_times = []
    t_pred_times = []
    for train_fold_idx, val_fold_idx in cv.split(X, y):
        if verbosity > 0:
            print("At step {}/{} splits.".format(count, n_folds))
        X_train, y_train = X[train_fold_idx], y[train_fold_idx]
        X_val, y_val = X[val_fold_idx], y[val_fold_idx]

        fit_start, _ = time_utils.print_time()
        fit_model = clf.fit(X_train, y_train)
        fit_end, _ = time_utils.print_time()
        _time = fit_end - fit_start
        fit_times.append(_time.total_seconds())

        v_pred_start, _ = time_utils.print_time()
        y_val_pred = fit_model.predict(X_val)
        v_pred_end, _ = time_utils.print_time()
        _time = v_pred_end - v_pred_start
        v_pred_times.append(_time.total_seconds())

        t_pred_start, _ = time_utils.print_time()
        y_train_pred = fit_model.predict(X_train)
        t_pred_end, _ = time_utils.print_time()
        _time = t_pred_end - t_pred_start
        t_pred_times.append(_time.total_seconds())

        # Create scoring results
        for score in scoring:
            v_s = get_scorer(score)._score_func(y_val, y_val_pred)
            if val_scores.get(score):
                arr = val_scores.get(score)
                arr.append(v_s)
                val_scores.update({score: arr})
            else:
                val_scores.update({score: [v_s]})

            if return_train_scores:
                t_s = get_scorer(score)._score_func(y_train, y_train_pred)
                if train_scores.get(score):
                    arr = train_scores.get(score)
                    arr.append(t_s)
                    train_scores.update({score: arr})
                else:
                    train_scores.update({score: [t_s]})

        count += 1

    train_scores.update({"fit_time": fit_times,
                         "pred_time": t_pred_times})
    val_scores.update({"fit_time": fit_times,
                         "pred_time": v_pred_times})

    end, end_str = time_utils.print_time()
    if verbosity > 0:
        print("Ending cross_validation fit at: " + end_str)
        print("Time elapsed: " + str(end - start))

    scores = {"val_scores": val_scores}
    if return_train_scores:
        scores.update({"train_scores": train_scores})
    return scores
