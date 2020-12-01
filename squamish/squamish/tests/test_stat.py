from unittest import TestCase

from sklearn.datasets import make_classification
import pytest

from squamish.stat import add_NFeature_to_X, Stats

from sklearn.linear_model import LassoLarsCV

from arfs_gen import genClassificationData
from squamish.utils import reduced_data
import numpy as np


def test():
    X, y = genClassificationData(n_features=10, n_strel=2, n_redundant=0)
    model = LassoLarsCV(cv=5)
    normal_score = model.fit(X, y).score(X, y)

    X_NF = add_NFeature_to_X(X, 1, np.random.RandomState())
    model = LassoLarsCV(cv=5)
    assert model.fit(X_NF, y).score(X_NF, y) > 0.5

    stats = Stats(model, X, y, n_resampling=50, fpr=1e-3, check_importances=False)
    bounds = stats.score_stat
    assert type(bounds) is tuple
    assert bounds[0] < normal_score < bounds[1]
