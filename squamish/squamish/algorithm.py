import logging
from copy import copy

import numpy as np
from sklearn.preprocessing import scale

from squamish.models import RF
from squamish.utils import reduced_data

logger = logging.getLogger(__name__)


def combine_sets(A, B):
    C = np.union1d(A, B)  # Combine with weakly relevant features
    C = np.sort(C).astype(int)
    return C


def set_without_f(A, f):
    C = np.setdiff1d(A, f)  # Remove f from set
    return C.astype(int)


class FeatureSorter:
    PARAMS = {
        # "max_depth": 5,
        "boosting_type": "rf",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "importance_type": "gain",
        "subsample": None,
        "subsample_freq": None,
        "colsample_bytree": None,
        "verbose": -1,
    }
    # SPARSE_PARAMS = copy(PARAMS)
    # SPARSE_PARAMS["feature_fraction"] = 1
    DENSE_PARAMS = copy(PARAMS)
    DENSE_PARAMS["feature_fraction"] = 0.1

    def __init__(
        self,
        problem_type,
        X,
        y,
        MR,
        AR,
        random_state,
        statistics,
        n_jobs=-1,
        debug=False,
    ):
        self.problem_type = problem_type
        self.n_jobs = n_jobs
        if debug:
            logger.setLevel(logging.DEBUG)
        self.random_state = random_state
        self.X = X
        self.y = y
        self.MR = MR
        self.AR = AR
        self.S = []
        self.W = list(np.setdiff1d(AR, MR))
        self.MR_and_W = combine_sets(MR, self.W)
        self.X_onlyrelevant = reduced_data(X, self.MR_and_W)
        self.X_onlyrelevant = scale(self.X_onlyrelevant)
        logger.debug(f"predetermined weakly {self.W}")

        self.model = RF(
            self.problem_type, random_state=self.random_state, n_jobs=self.n_jobs
        )

        self.score_bounds = statistics.score_stat

        logger.debug(f"score bounds: {self.score_bounds}")

    def check_significance(self, f, score_without_f):
        # check score if f is removed
        # Test if value lies in acceptance range of null distribution
        # i.e. no signif. change compared to perm. feature
        # __We only check lower dist bound for worsening score when f is removed -> Strong relevant
        logger.debug(f"score without {f}: {score_without_f}")

        if not self.score_bounds[0] < score_without_f  < self.score_bounds[1]:
            logger.debug(f"removal_score:{score_without_f:.3}-> S")
            self.S.append(f)
            return True
        else:
            logger.debug(f"removal_score:{score_without_f:.3}-> W")
            self.W.append(f)
            return False

    def check_each_feature(self):

        if len(self.MR_and_W) == 1:
            # Only one feature, which should be str. relevant
            self.S = self.MR
            logger.debug("Only one feature")
            return

        for f in self.MR:
            logger.debug(f"------------------- Feature f:{f}")

            # Remove feature f from MR u W
            fset_without_f = set_without_f(self.MR_and_W, f)

            # What index has f in featureset
            index_of_f = np.where(self.MR_and_W == f)[0]

            # Fit model with f removed (permutated) and return score
            score_without_f = self.model.score_with_i_permuted(
                self.X_onlyrelevant, self.y, index_of_f, random_state=self.random_state
            )

            # Check score with previously created statistic
            self.check_significance(f, score_without_f)

        logger.debug(f"S: {self.S}")
        logger.debug(f"W: {self.W}")


def print_scores_on_sets(AR, MR, MR_and_W, X, model, y):
    score_on_MR = model.redscore(X, y, MR)
    score_on_AR = model.redscore(X, y, AR)
    score_on_MR_and_W = model.redscore(X, y, MR_and_W)
    normal_imps = model.importances()
    logger.debug(f"normal_imps:{normal_imps}")
    logger.debug(f"length MR and W {len(MR_and_W)}")
    scores = {"MR": score_on_MR, "AR": score_on_AR, "MR+W": score_on_MR_and_W}
    for k, sc in scores.items():
        logger.debug(f"{k} has score {sc}")
