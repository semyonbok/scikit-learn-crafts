import pytest

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

from crafts.regressors import PredictionIntervalRegressor

RS = 17_17_17_17_17


@pytest.fixture
def get_split_dataset():
    peek = fetch_california_housing()
    X = peek.data.copy()
    y = peek.target.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RS,
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def get_regressor():
    estimator_ = HistGradientBoostingRegressor(random_state=RS)
    return PredictionIntervalRegressor(
        estimator=estimator_, random_state=RS, n_estimators=1000, n_jobs=-1
    )


def test_subtype():
    """Expect to fail same tests as BaggingRegressor"""
    br_report = check_estimator(
        BaggingRegressor(), on_fail=None, on_skip=None
    )
    pir_report = check_estimator(
        PredictionIntervalRegressor(), on_fail=None, on_skip=None
    )

    # picked dicts because unsure if check order is always respected
    br_dict = {check["check_name"]: check["status"] for check in br_report}
    pir_dict = {check["check_name"]: check["status"] for check in pir_report}

    assert not set(br_dict.keys()).symmetric_difference(pir_dict.keys())
    for status in br_dict.keys():
        assert br_dict[status] == pir_dict[status]


def test_coverage(get_regressor, get_split_dataset):
    """See if it approaches specified coverage on test set"""
    pir = get_regressor
    X_train, X_test, y_train, y_test = get_split_dataset
    pir.fit(X_train, y_train)
    qs = pir.predict_quantiles(X_test, [0.05, 0.95])
    y_low, y_high = qs.T
    test_coverage = pir.coverage_fraction(y_test, y_low, y_high)
    assert 0.885 <= test_coverage <= 0.915
