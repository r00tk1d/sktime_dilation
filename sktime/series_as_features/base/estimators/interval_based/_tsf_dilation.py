# -*- coding: utf-8 -*-
"""Time Series Forest (TSF) Classifier."""

__author__ = [
    "Tony Bagnall",
    "kkoziara",
    "luiszugasti",
    "kanand77",
    "Markus Löning",
    "Oleksii Kachaiev",
]
__all__ = [
    "BaseTimeSeriesForestDilation",
    "_transform",
    "_get_intervals",
    "_fit_estimator",
]

import math
import random

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.multiclass import class_distribution
from sklearn.utils.validation import check_random_state

from sktime.base._base import _clone_estimator
from sktime.utils.slope_and_trend import _slope
from sktime.utils.validation import check_n_jobs


class BaseTimeSeriesForestDilation:
    """Base time series forest classifier."""

    def __init__(
        self,
        min_interval=3,
        n_estimators=200,
        n_jobs=1,
        random_state=None,

        num_of_random_dilations=1,
        n_intervals_prop=1,
        interval_length_prop=1
    ):
        super(BaseTimeSeriesForestDilation, self).__init__(
            base_estimator=self._base_estimator,
            n_estimators=n_estimators,
        )

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.n_jobs = n_jobs
        # The following set in method fit
        self.n_classes = 0
        self.series_length = 0
        self.n_intervals = 0
        self.estimators_ = []
        self.intervals_ = []
        self.classes_ = []

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

        self.num_of_random_dilations = num_of_random_dilations
        self.n_intervals_prop = n_intervals_prop
        self.interval_length_prop = interval_length_prop

    def _fit(self, X, y):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        Xt: np.ndarray or pd.DataFrame
            Panel training data.
        y : np.ndarray
            The class labels.

        Returns
        -------
        self : object
            An fitted instance of the classifier
        """
        X = X.squeeze(1)
        n_instances, self.series_length = X.shape

        n_jobs = check_n_jobs(self.n_jobs)

        rng = check_random_state(self.random_state)

        self.n_classes = np.unique(y).shape[0]

        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        self.n_intervals = int(math.sqrt(self.series_length)*self.n_intervals_prop)
        self.feature_count = 3 * self.n_intervals * self.n_estimators * self.num_of_random_dilations
        if self.n_intervals == 0:
            self.n_intervals = 1
        if self.series_length < self.min_interval:
            self.min_interval = self.series_length

        self.intervals_ = [
            _get_intervals(self.n_intervals, self.min_interval, self.series_length, rng, self.interval_length_prop, self.num_of_random_dilations)
            for _ in range(self.n_estimators)
        ]

        self.estimators_ = Parallel(n_jobs=n_jobs)(
            delayed(_fit_estimator)(
                _clone_estimator(self.base_estimator, rng), X, y, self.intervals_[i]
            )
            for i in range(self.n_estimators) # n_estimator: Number of estimators to build for the ensemble
        )

        self._is_fitted = True
        return self


def _transform(X, intervals):
    """Transform X for given intervals.

    Compute the mean, standard deviation and slope for given intervals of input data X.

    Parameters
    ----------
    Xt: np.ndarray or pd.DataFrame
        Panel data to transform.
    intervals : np.ndarray
        Intervals containing start and end values.

    Returns
    -------
    Xt: np.ndarray or pd.DataFrame
     Transformed X, containing the mean, std and slope for each interval
    """

    # MOD hier die dilation_size aus dem interval ausgelesen und angewendet
    n_instances, _ = X.shape
    n_intervals_with_dilation, _ = intervals.shape
    transformed_x = np.empty(shape=(3 * n_intervals_with_dilation, n_instances), dtype=np.float32)
    for j in range(n_intervals_with_dilation):
            
        d = intervals[j][2]
        
        X_dilated = X[:, 0::d]
        for i in range(1, d):
            second = X[:, i::d]
            X_dilated = np.concatenate((X_dilated, second), axis=1)
        
        X_slice = X_dilated[:, intervals[j][0] : intervals[j][1]] # MOD
        means = np.mean(X_slice, axis=1)
        std_dev = np.std(X_slice, axis=1)
        slope = _slope(X_slice, axis=1)
        transformed_x[3 * j] = means
        transformed_x[3 * j + 1] = std_dev
        transformed_x[3 * j + 2] = slope

        #X_dilated = X_dilated[0:0]

    return transformed_x.T


def _get_intervals(n_intervals, min_interval, series_length, rng, interval_length_prop, num_of_random_dilations):
    """Generate random intervals for given parameters."""
    # MOD hier dilation random gewählt (momentaner Stand verbessert nicht die performance da die anzahl der intervalle bisher nicht reduziert wird)
    intervals = np.zeros((n_intervals*num_of_random_dilations, 3), dtype=int) # MOD 2 -> 3
    for j in range(n_intervals):
        start = rng.randint(series_length - min_interval) # hier wird der interval start random bestimmt 
        length = int(rng.randint(series_length - intervals[j][0] - 1)*interval_length_prop) #  hier wird die length des intervals bestimmt
        if length < min_interval:
            length = min_interval
        for k in range(num_of_random_dilations):
            intervals[j*num_of_random_dilations+k][0] = start
            intervals[j*num_of_random_dilations+k][1] = start + length # -> interval j geht von interval[j][0] bis interval[j][1]
            d_size = 2 ** np.random.uniform(
                0, np.log2((series_length - 1) / (length - 1))
            )
            d_size = np.int32(d_size)

            intervals[j*num_of_random_dilations+k][2] = d_size
    return intervals


def _fit_estimator(estimator, X, y, intervals):
    """Fit an estimator on input data (X, y)."""
    transformed_x = _transform(X, intervals)
    return estimator.fit(transformed_x, y)
