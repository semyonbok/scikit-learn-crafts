from typing import Any, Optional, List, Union, Tuple, Literal
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import ElasticNet
from sklearn.utils.validation import check_array, check_is_fitted
from joblib import Parallel, delayed


class GroupRegressor(BaseEstimator, RegressorMixin):
    """
    Fits separate regressors for each unique combo of categorical columns.

    Parameters
    ----------
    groupby_cols : List[str]
        Column names in X to group by.
    base_estimator : estimator, default=None
        Estimator to clone for each group. If None, uses ElasticNet.
    n_jobs : Optional[int], default=-1
        Number of CPU cores for parallel fitting.
    fallback : {'global', 'zero'}, default='global'
        Behavior for unseen groups: 'global' uses a model fitted on all data;
        'zero' returns zeros.
    """

    def __init__(
        self,
        groupby_cols: List[str],
        base_estimator: Optional[Any] = None,
        n_jobs: Optional[int] = -1,
        fallback: Literal["global", "zero"] = 'global'
    ):
        if fallback not in ('global', 'zero'):
            raise ValueError("fallback must be either 'global' or 'zero'")

        self.groupby_cols = list(groupby_cols)
        self.n_jobs = n_jobs
        self.fallback = fallback

        if base_estimator is None:
            self.base_estimator = ElasticNet()
        else:
            self.base_estimator = base_estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> "GroupRegressor":
        """
        Fit one estimator per unique group; also fit fallback if requested.
        """
        # Require DataFrame for grouping
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)
        elif not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series or numpy array")

        # Validate grouping columns
        missing = set(self.groupby_cols) - set(X.columns)
        if missing:
            raise KeyError(f"Missing grouping columns in X: {missing}")

        # Identify feature columns
        self._feature_cols = [
            c for c in X.columns if c not in self.groupby_cols]
        X_num = X[self._feature_cols]
        # Ensure numeric features
        X_num = check_array(X_num, ensure_2d=True)

        # Fit fallback model if needed
        if self.fallback == 'global':
            self._global_estimator = clone(self.base_estimator)
            self._global_estimator.fit(X_num, y)

        # Fit per-group models in parallel
        def _fit_group(key: Tuple, df: pd.DataFrame):
            key = key if isinstance(key, tuple) else (key,)  # if one group
            est = clone(self.base_estimator)
            Xg = df[self._feature_cols].values
            yg = y.loc[df.index].values
            est.fit(Xg, yg)
            return key, est

        groups = list(X.groupby(self.groupby_cols))
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_group)(key, grp) for key, grp in groups
        )

        self.estimators_ = {key: est for key, est in results}
        self.n_features_in_ = len(self._feature_cols)
        return self

    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict using group-specific estimators or fallback behavior.
        """
        check_is_fitted(self, 'estimators_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Validate grouping columns
        missing = set(self.groupby_cols) - set(X.columns)
        if missing:
            raise KeyError(f"Missing grouping columns in X: {missing}")

        y_pred = pd.Series(
            data=np.zeros(X.shape[0], dtype=float),
            index=X.index
        )

        # Map each row to its estimator
        for key, df in X.groupby(self.groupby_cols):
            idx = df.index
            norm_key = key if isinstance(key, tuple) else (key,)
            est = self.estimators_.get(norm_key)
            if est is None:
                if self.fallback == 'global':
                    est = self._global_estimator
                else:
                    # zero fallback
                    continue
            Xg = df[self._feature_cols].values
            y_pred[idx] = est.predict(Xg)
        return y_pred.values

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters and nested base estimator params.
        """
        params = {
            'groupby_cols': self.groupby_cols,
            'n_jobs': self.n_jobs,
            'fallback': self.fallback,
        }
        if deep:
            for name, val in self.base_estimator.get_params(deep=True).items():
                params[f'base_estimator__{name}'] = val
        return params

    def set_params(self, **params) -> "GroupRegressor":
        """
        Set parameters, splitting out base estimator params.
        """
        be_params = {}
        for key in list(params):
            if key.startswith('base_estimator__'):
                be_params[key.split('__', 1)[1]] = params.pop(key)
        for key, val in params.items():
            setattr(self, key, val)
        if be_params:
            self.base_estimator.set_params(**be_params)
        return self
