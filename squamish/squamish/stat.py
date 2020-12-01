import numpy as np
from scipy import stats
from sklearn.utils import check_random_state
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def _create_probe_statistic(probe_values, fpr):
    # Create prediction interval statistics based on randomly permutated probe features (based on real features)
    n = len(probe_values)

    if n == 1:
        val = probe_values[0]
        low_t = val
        up_t = val
    else:
        probe_values = np.asarray(probe_values)
        mean = probe_values.mean()
        s = probe_values.std()
        low_t = mean + stats.t(df=n - 1).ppf(fpr) * s * np.sqrt(1 + (1 / n))
        up_t = mean - stats.t(df=n - 1).ppf(fpr) * s * np.sqrt(1 + (1 / n))
    return low_t, up_t


def add_NFeature_to_X(X, feature_i, random_state):
    X_copy = np.copy(X)
    # Permute selected feature
    permutated_feature = random_state.permutation(X_copy[:, feature_i])

    # Append permutation to dataset
    X_copy = np.hstack([X_copy, permutated_feature[:, None]])
    return X_copy


class Stats:
    def __init__(
        self,
        model,
        X,
        y,
        n_resampling=50,
        fpr=1e-4,
        random_state=None,
        check_importances=True,
        debug=False
    ):
        self.model = deepcopy(model)
        self.X = X
        self.y = y
        self.n_resampling = n_resampling
        self.fpr = fpr
        self.random_state = check_random_state(random_state)
        self.check_importances = check_importances
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

        samples = self.generate_samples()

        scores = [sc[0] for sc in samples]
        self.score_stat = _create_probe_statistic(scores, fpr)
        if check_importances:
            imps = np.array([sc[1] for sc in samples]).T
            # Iterate over columns of matrix, column corresponds to all samples for 1 feature
            self.imp_stat = [_create_probe_statistic(col, fpr) for col in imps]

            self.shadow_importance_samples = [sc[2] for sc in samples]
            self.shadow_stat = _create_probe_statistic(
                self.shadow_importance_samples, fpr
            )
            logger.debug(f"Shadow Bounds:{self.shadow_stat}")

    def generate_samples(self):

        # Random sample n_resampling shadow features by permuting real features
        random_choice = self.random_state.choice(
            a=self.X.shape[1], size=self.n_resampling
        )

        # Instantiate objects
        samples = []
        for di in random_choice:
            X_NF = add_NFeature_to_X(self.X, di, self.random_state)
            self.model.fit(X_NF, self.y)
            score = self.model.score(X_NF, self.y)
            if not self.check_importances:
                samples.append((score,))
            else:
                # Only get values of non-permutated features
                imps = self.model.importances()[:-1]
                shadowimp = self.model.importances()[-1]
                samples.append((score, imps, shadowimp))
        return samples
