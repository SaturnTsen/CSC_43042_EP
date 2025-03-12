#! /usr/bin/env python3
from functools import cache  # noqa: F401
from timeit import timeit

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import List
from TD import KDTree as Yours
from scipy.spatial import KDTree as Theirs
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# @cache  => run this only once and cache the results, but then the 'drawn' data doesn't change
def iris():
    """Load the iris dataset's features and split"""
    logger.debug("Load iris data")
    X, y = load_iris(return_X_y=True, as_frame=False)
    return train_test_split(X, y, train_size=0.8)


def instantiate_kdtrees(X_train: np.ndarray):
    """Return objects of both implementations"""
    logger.debug("Compute kd-trees")
    return Yours(X_train), Theirs(X_train)


def predict(X_test: np.ndarray, implementation: Yours | Theirs):
    """Predict, i.e. give nearest neighbor's index in train test"""
    logger.debug(f"Predict nearest neighbor with implementation {implementation}")
    return [implementation.query(x_test) for x_test in X_test]


def compare_implementations(X_test: np.ndarray, implementations: List[Yours | Theirs]):
    """
    Compare implementations, i.e. verify that indices of nearest neighbors are similar
    Return list of indices in X_train corresponding to X_test's nearest neighbors' indices
    """
    logger.info("Compare implementations...")
    results = [predict(X_test, implementation) for implementation in implementations]
    try:
        assert all(
            [
                np.allclose(res_1, res_2)
                for res_1, res_2 in zip(results[0] * (len(results) - 1), *results[1:])
            ]
        ), "Predictions are not the same"
    except AssertionError as e:
        logger.info(f"{e}: looking whether it's just the way the two break ties...")
        # Just look at distances
        assert all(
            [
                np.allclose(res_1[0], res_2[0])
                for res_1, res_2 in zip(results[0] * (len(results) - 1), *results[1:])
            ]
        ), "No, it's definitely wrong"
    logger.info("Predictions are the same, congratulations")
    for implementation in implementations:
        logger.info(
            f"Implementation {implementation} takes "
            f"{timeit(lambda: predict(X_test, implementation), number=100):.4f}s "
            f"to run 100 times on {len(X_test)} query points."
        )
    # If you're curious about why scipy is faster,
    # see https://github.com/scipy/scipy/blob/v1.15.2/scipy/spatial/_ckdtree.pyx
    return [[res[1] for res in res_impl] for res_impl in results]


def error_rate(y_true: np.ndarray, y_pred: List[float] | np.ndarray):
    """Compute binary classification error rate"""
    return np.equal(y_true, y_pred).sum() / len(y_true)


def main():
    logger.info("Begin script...")
    X_train, X_test, y_train, y_test = iris()
    yours, theirs = instantiate_kdtrees(X_train)
    X_train_indices_for_test = compare_implementations(X_test, [yours, theirs])
    y_test_hats = y_train[X_train_indices_for_test]
    # Compare implementations by e.g. runtime in addition
    # N.B.: train_test_split is random, hence this will change when rerun
    error_rates = [error_rate(y_test, y_test_hat) for y_test_hat in y_test_hats]
    logger.info(
        f"Error rate on test set: yours {error_rates[0]:.2f} / SciPy's {error_rates[1]:.2f}"
    )


if __name__ == "__main__":
    main()
