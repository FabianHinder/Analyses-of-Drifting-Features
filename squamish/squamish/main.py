import logging
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state

from squamish.algorithm import FeatureSorter
from squamish.utils import create_support_AR
from . import models
from .stat import Stats

logger = logging.getLogger(__name__)


class Main(BaseEstimator):
    def __init__(
        self,
        problem_type="classification",
        n_resampling=50,
        fpr=1e-6,
        random_state=None,
        n_jobs=-1,
        debug=True,
    ):
        """

        Parameters
        ----------
        problem_type : string
            "classification", "regression" or "ranking"
        n_resampling : int
            Number of samples used in statistics creation.
        fpr : float
            Parameter for t-statistic to control strictness of acceptance.
            Lower is more strict, higher allows more false positives.
        random_state : object or int
            Numpy random state object or int to set seed.
        n_jobs : int
            Number of parallel threads used.
            '-1' makes automatic choice depending on avail. CPUs.
        debug : bool
            Enable debug output.
        """
        self.n_jobs = n_jobs
        self.problem_type = problem_type
        self.n_resampling = n_resampling
        self.fpr = fpr
        self.random_state = check_random_state(random_state)
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

    def _get_support_mask(self):
        """
        Returns
        -------
        self.support_: vector
            Vector with boolean class of each input feature with it's computed relevance class.
            0 = irrelevant, 1 relevant
        """
        return self.support_

    def fit(self, X, y):
        X = scale(X)
        n, d = X.shape

        # All relevant set using Boruta
        m = models.MyBoruta(
            self.problem_type, random_state=self.random_state, n_jobs=self.n_jobs
        ).fit(X, y)
        # bor_score = m.cvscore(X, y)
        fset = m.fset(X, y)
        AR = np.where(fset)[0]

        # Fit a simple Random Forest to get a minimal feature subset
        m = models.RF(
            self.problem_type, random_state=self.random_state, n_jobs=self.n_jobs
        ).fit(X, y)
        self.score_ = m.score(X, y)
        logger.debug(f"RF score {self.score_}")
        logger.debug(f"importances {m.estimator.feature_importances_}")
        self.rfmodel = deepcopy(m)

        self.stat_ = Stats(
            m,
            X,
            y,
            n_resampling=self.n_resampling,
            fpr=self.fpr,
            random_state=self.random_state,
            check_importances=True,
            debug=self.debug
        )
        fset = self.rfmodel.fset(X, y, self.stat_)
        fset = np.where(fset)
        MR = fset[0]

        logger.debug(f"Features from Boruta: {AR}")
        logger.debug(f"Features from RF: {MR}")
        if len(AR) < 1:
            raise Exception("No features were selected in AR model. Is model properly fit? (score ok?)")

        # Sort features iteratively into strongly (S) and weakly (W) sets
        self.fsorter = FeatureSorter(
            self.problem_type,
            X,
            y,
            MR,
            AR,
            self.random_state,
            self.stat_,
            n_jobs=self.n_jobs,
            debug=self.debug
        )
        self.fsorter.check_each_feature()

        # Turn index sets into support vector
        # (2 strong relevant,1 weak relevant, 0 irrelevant)
        all_rel_support = create_support_AR(d, self.fsorter.S, self.fsorter.W)
        self._relevance_classes = all_rel_support
        logger.info(f"Relevance Classes: {self.relevance_classes_}")

        # Simple boolean vector where relevan features are regarded as one set (1 relevant, 0 irrelevant)
        self.support_ = self._relevance_classes > 0

    @property
    def relevance_classes_(self):
        """ Returnss vector of relevance classes. 0 = irrelevant, 1 = weakly relevant, 2 = strongly relevant"""
        if self._relevance_classes is None:
            raise NotFittedError("Call fit first.")
        return self._relevance_classes

    def score(self, X, y):
        """
            Score of internal random forest model.
        Parameters
        ----------
        X : matrix
            Data matrix
        y : vector
            target vector
        Returns
        -------
        score:
            score on data matrix
        """
        return self.rfmodel.score(X, y)

    def predict(self, X):
        return self.rfmodel.predict(X)
