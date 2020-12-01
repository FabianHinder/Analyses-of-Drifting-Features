import pytest
from arfs_gen import genClassificationData, genRegressionData
from sklearn.datasets import make_classification

from squamish.main import Main


@pytest.fixture
def data():
    X, y = make_classification(
        100,
        10,
        n_informative=2,
        n_redundant=1,
        n_clusters_per_class=2,
        flip_y=0,
        shuffle=False,
        random_state=123,
    )
    return X, y


@pytest.fixture(scope="module")
def model():
    return Main()


def test_fit(data, model):
    X, y = data
    assert len(X) == len(y)

    model.fit(X, y)
    assert model.relevance_classes_ is not None
    assert len(model._get_support_mask()) == X.shape[1]


def test_linear_data_class():
    X, y = genClassificationData(
        n_features=5, n_strel=1, n_redundant=2, n_samples=200, random_state=1234
    )
    model = Main(problem_type="classification")
    model.fit(X, y)
    assert len(model.relevance_classes_) == X.shape[1]

def test_linear_data_regression():
    X, y = genRegressionData(
        n_features=5, n_strel=1, n_redundant=2, n_samples=200, random_state=1234
    )
    model = Main(problem_type="regression")
    model.fit(X, y)
    assert len(model.relevance_classes_) == X.shape[1]