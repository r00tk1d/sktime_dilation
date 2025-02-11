# -*- coding: utf-8 -*-
"""Interval based time series classifiers."""
__all__ = [
    "TimeSeriesForestClassifier",
    "RandomIntervalSpectralEnsemble",
    "SupervisedTimeSeriesForest",
    "CanonicalIntervalForest",
    "DrCIF",
    "TimeSeriesForestClassifierDilation",
    "TimeSeriesForestClassifierDilationReal",
]

from sktime.classification.interval_based._cif import CanonicalIntervalForest
from sktime.classification.interval_based._drcif import DrCIF
from sktime.classification.interval_based._rise import RandomIntervalSpectralEnsemble
from sktime.classification.interval_based._stsf import SupervisedTimeSeriesForest
from sktime.classification.interval_based._tsf import TimeSeriesForestClassifier
from sktime.classification.interval_based._tsf_dilation import TimeSeriesForestClassifierDilation
from sktime.classification.interval_based._tsf_dilation_real import TimeSeriesForestClassifierDilationReal