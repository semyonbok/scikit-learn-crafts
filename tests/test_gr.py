import pytest

import pandas as pd

from sklearn.base import clone, check_is_fitted
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from crafts.regressors import GroupRegressor

RS = 17_17_17_17_17
GROUPBY_COLS = ['sex', 'age_group']


@pytest.fixture
def get_split_dataset():
    peek = load_diabetes(return_X_y=False, as_frame=True, scaled=False)
    X = peek.data.copy()
    y = peek.target.copy()

    # add age group category
    range_ = range(10, 90, 10)
    X["age_group"] = pd.cut(
        X["age"], range_, right=False, labels=[f"{a}s" for a in range_[:-1]]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RS, stratify=X[GROUPBY_COLS]
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def get_regressor():
    be = ElasticNet(selection="cyclic")
    return GroupRegressor(groupby_cols=GROUPBY_COLS, base_estimator=be)


def test_clone(get_regressor):
    gr = get_regressor
    clone(gr)


def test_get_params_shallow(get_regressor):
    gr = get_regressor
    expected_keys = {"groupby_cols", "base_estimator", "n_jobs", "fallback"}
    params = gr.get_params(deep=False)
    keys = set(params.keys())
    assert keys.union(expected_keys) == expected_keys


def test_get_params_deep(get_regressor):
    gr = get_regressor
    gr_keys = {"groupby_cols", "base_estimator", "n_jobs", "fallback"}
    en_keys = ElasticNet().get_params().keys()
    be_keys = {f"base_estimator__{p}" for p in en_keys}
    expected_keys = gr_keys.union(be_keys)
    params = gr.get_params(deep=True)
    keys = set(params.keys())
    assert keys.union(expected_keys) == expected_keys


def test_set_groupby_cols(get_regressor):
    # TODO pass integer indices
    gr = get_regressor
    gr.set_params(groupby_cols=["Sex"])
    assert gr.groupby_cols == ("Sex",)


def test_set_groupby_cols_when_fitted(get_regressor):
    gr = get_regressor
    gr.estimators_ = {}
    with pytest.raises(ValueError):
        gr.set_params(groupby_cols=["Sex"])


def test_set_base_estimator(get_regressor):
    gr = get_regressor
    gr.set_params(base_estimator=DecisionTreeRegressor())
    assert isinstance(gr.base_estimator, DecisionTreeRegressor)


def test_set_base_estimator_when_fitted(get_regressor):
    gr = get_regressor
    gr.estimators_ = {}
    with pytest.warns(UserWarning) as record:
        gr.set_params(base_estimator=DecisionTreeRegressor())
    assert len(record) == 1  # might fail in case of a FutureWarning?
    assert "'base_estimator' was reset" in str(record[0].message)
    assert isinstance(gr.base_estimator, DecisionTreeRegressor)


@pytest.mark.parametrize("n_jobs", (None, 1, 2, -1))
def test_set_n_jobs(get_regressor, n_jobs):
    gr = get_regressor
    gr.set_params(n_jobs=n_jobs)
    assert gr.n_jobs == n_jobs


@pytest.mark.parametrize("fallback", ("global", "zero"))
def test_set_fallback(get_regressor, fallback):
    gr = get_regressor
    gr.set_params(fallback=fallback)
    assert gr.fallback == fallback


def test_set_fallback_invalid(get_regressor):
    gr = get_regressor
    with pytest.raises(ValueError):
        gr.set_params(fallback="fall what?")


@pytest.mark.skip(reason="Not yet implemented")
def test_set_be_params():
    pass


@pytest.mark.parametrize("n_jobs", (None, 1, 2, -1))
def test_fit(get_regressor, n_jobs, get_split_dataset):
    gr = get_regressor
    gr.set_params(n_jobs=n_jobs)
    X_train, _, y_train, _ = get_split_dataset
    gr.fit(X_train, y_train)
    assert check_is_fitted(gr, "estimators_") is None
    assert len(gr.estimators_) == 13


def test_fit_X_array(get_regressor, get_split_dataset):
    gr = get_regressor
    X_train, _, y_train, _ = get_split_dataset
    with pytest.raises(TypeError):
        gr.fit(X_train.values, y_train)


def test_fit_y_reset_index(get_regressor, get_split_dataset):
    gr = get_regressor
    X_train, _, y_train, _ = get_split_dataset
    y_train.reset_index(drop=True, inplace=True)
    with pytest.raises(IndexError):
        gr.fit(X_train, y_train)


@pytest.mark.skip(reason="Not yet implemented")
def test_predict():
    # TODO ensure index remains as in X_test
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_predict_missing_group():
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_predict_against_estimators():
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_cvp():
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_cvs():
    pass
