from typing import Optional, Union, Tuple, Literal, Sequence, Hashable
import warnings as wrn

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import ElasticNet
from sklearn.utils.validation import check_array, check_is_fitted

from joblib import Parallel, delayed


class GroupRegressor(BaseEstimator, RegressorMixin):
    """
    Fits separate regressors for each unique combination of categorical columns.

    Each unique combination of values in `groupby_cols` gets its own clone
    of `base_estimator`. At predict time, if a group in X was never seen during
    fit, one of two behaviors occurs depending on `fallback`:

    - 'global': use a single “global” estimator trained on all data
    - 'zero': return zero for those rows

    Parameters
    ----------
    groupby_cols : Sequence[Hashable]
        Column names in X to group by.
        These columns must exist in any DataFrame passed to `fit` or `predict`.
    base_estimator : BaseEstimator, optional (default=None)
        A scikit-learn-style regressor to clone for each group. If None,
        defaults to `sklearn.linear_model.ElasticNet()`.
    n_jobs : int or None, optional (default=-1)
        Number of parallel jobs to use when fitting group-specific estimators.
        -1 means “use all available cores.”
    fallback : {'global', 'zero'}, optional (default='global')
        Behavior for rows whose group was unseen during `fit`.
    """

    def __init__(
        self,
        groupby_cols: Sequence[Hashable],
        base_estimator: Optional[BaseEstimator] = None,
        n_jobs: Optional[int] = -1,
        fallback: Literal["global", "zero"] = "global",
    ):
        """
        Initialize GroupRegressor.

        Parameters
        ----------
        groupby_cols : Sequence[Hashable]
            Column names to group by. Must remain immutable after fitting.
        base_estimator : BaseEstimator, optional
            Estimator to clone for each group. If None, uses ElasticNet().
        n_jobs : int or None, optional
            Number of parallel jobs for fitting.
        fallback : {'global', 'zero'}, optional
            Behavior for unseen groups:
                'global' - fall back to a global model trained on all data
                'zero'   - output zeros for unseen-group rows
        """
        self._validate_fallback(fallback)

        self.groupby_cols = tuple(groupby_cols)  # to enable cloning and CV
        self.n_jobs = n_jobs
        self.fallback = fallback

        if base_estimator is None:
            self.base_estimator = ElasticNet()
        else:
            self.base_estimator = base_estimator

    def _validate_fallback(self, fallback):
        """Validate the fallback argument."""
        if fallback not in ("global", "zero"):
            raise ValueError("fallback must be either 'global' or 'zero'")

    def _validate_groupby_cols(self, groupby_cols, X):
        """Validate grouping columns"""
        missing = set(groupby_cols) - set(X.columns)
        if missing:
            raise KeyError(f"Missing grouping columns in X: {missing}")

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[Union[int, float]],
    ) -> "GroupRegressor":
        """
        Fit one estimator per unique group; also fit a global fallback if requested.

        This method will:
        - Check that X is a DataFrame and that y is aligned with X.index.
        - Verify that all columns in `groupby_cols` exist in X.
        - Separate X into numeric feature columns and grouping columns.
        - If `fallback='global'`, fit a “global” estimator on all rows of X.
        - Partition X by each unique combination of values in `groupby_cols`,
          clone `base_estimator` for each group, and fit it on that group’s subset.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Must contain the columns in `groupby_cols` and any
            numeric feature columns.
        y : array-like of shape (n_samples,)
            Target vector aligned with X.index.

        Returns
        -------
        self : GroupRegressor
            Fitted estimator with group-specific models stored in `estimators_`.

        Raises
        ------
        TypeError
            If X is not a pandas DataFrame.
        IndexError
            If y is a pandas Series whose index does not match X.index.
        KeyError
            If any column in `groupby_cols` is missing from X.
        """
        # Require DataFrame for grouping
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
        elif not X.index.equals(y.index):
            raise IndexError("y must have same index as X")

        self._validate_groupby_cols(self.groupby_cols, X)

        # Identify feature columns
        self._feature_cols = [c for c in X.columns if c not in self.groupby_cols]
        X_num = X[self._feature_cols]
        # Ensure numeric features
        X_num = check_array(X_num, ensure_2d=True)

        # Fit fallback model if needed
        if self.fallback == "global":
            self._global_estimator = clone(self.base_estimator)
            self._global_estimator.fit(X_num, y)

        # Fit per-group models in parallel
        def _fit_group(key: Tuple, df: pd.DataFrame):
            key = key if isinstance(key, tuple) else (key,)  # if one group
            est = clone(self.base_estimator)
            Xg = df[self._feature_cols]
            yg = y.loc[df.index]
            est.fit(Xg, yg)
            return key, est

        groups = X.groupby(list(self.groupby_cols), observed=True, dropna=False)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_group)(key, grp) for key, grp in groups
        )

        self.estimators_ = {key: est for key, est in results}
        self.n_features_in_ = len(self._feature_cols)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using group-specific estimators or fallback behavior.

        For each row in X:
        - Look up the group tuple (the combination of values in `groupby_cols`).
        - If a model for that group exists in `estimators_`, use it to predict.
        - Otherwise, if `fallback='global'`, use the global model trained on all data.
        - If `fallback='zero'`, leave the prediction as zero.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction. Must contain the same `groupby_cols`
            used during fit, plus the numeric feature columns.

        Returns
        -------
        pd.Series
            Predicted values indexed the same as X.index.

        Raises
        ------
        TypeError
            If X is not a pandas DataFrame.
        KeyError
            If any column in `groupby_cols` is missing from X.
        NotFittedError
            If `fit` has not been called before `predict`.
        """
        check_is_fitted(self, "estimators_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self._validate_groupby_cols(self.groupby_cols, X)

        y_pred = pd.Series(data=np.zeros(X.shape[0], dtype=float), index=X.index)

        # Map each row to its estimator
        unseen_keys = []
        for key, df in X.groupby(list(self.groupby_cols), observed=True, dropna=False):
            idx = df.index
            key = key if isinstance(key, tuple) else (key,)
            est = self.estimators_.get(key)
            if est is None:
                unseen_keys.append(key)
                if self.fallback == "global":
                    est = self._global_estimator
                else:
                    continue  # zero fallback
            Xg = df[self._feature_cols]
            y_pred[idx] = est.predict(Xg)

        if unseen_keys:
            wrn.warn(f"Groups unseen in training encountered: {unseen_keys}")

        return y_pred

    def get_params(self, deep: bool = True) -> dict:
        """
        Return parameters for this estimator, including nested base estimator params.

        This method ensures compatibility with scikit-learn's `clone()` and
        hyperparameter tuning utilities by returning both top-level parameters
        (`groupby_cols`, `n_jobs`, `fallback`, `base_estimator`) and, if `deep=True`,
        all parameters of `base_estimator` prefixed with 'base_estimator__'.

        Parameters
        ----------
        deep : bool, default=True
            If True, return nested parameters of `base_estimator` as well.

        Returns
        -------
        params : dict
            Parameter names mapped to their values. Nested parameters from
            `base_estimator` appear with keys like 'base_estimator__param_name'.
        """
        params = {
            "groupby_cols": self.groupby_cols,
            "n_jobs": self.n_jobs,
            "fallback": self.fallback,
            "base_estimator": self.base_estimator,
        }
        if deep:
            for name, val in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{name}"] = val
        return params

    def set_params(self, **params) -> "GroupRegressor":
        """
        Set parameters for this estimator, splitting out nested estimator params.

        Top-level parameters that can be set:
            - n_jobs : int
            - fallback : {'global', 'zero'}
            - base_estimator : BaseEstimator

        Nested parameters for `base_estimator` can be set by using keys
        prefixed with 'base_estimator__', e.g. 'base_estimator__alpha'.

        Attempting to change `groupby_cols` after fit will raise an error. Changing
        `base_estimator` after fit will issue a warning; users must call `fit()` again
        to retrain group-specific models.

        Parameters
        ----------
        **params : dict
            Parameter names mapped to their new values.

        Returns
        -------
        self : GroupRegressor
            Estimator instance with updated parameters.

        Raises
        ------
        ValueError
            If attempting to set `groupby_cols` after fitting, or if any provided key
            does not match a known parameter name.
        """
        if "groupby_cols" in params.keys():
            params["groupby_cols"] = tuple(params["groupby_cols"])

        if "fallback" in params.keys():
            self._validate_fallback(params["fallback"])

        be_params = {}
        for key in list(params):
            if key.startswith("base_estimator__"):
                be_params[key.split("__", 1)[1]] = params.pop(key)

        is_fitted = hasattr(self, "estimators_")
        valid_params = ("groupby_cols", "n_jobs", "fallback", "base_estimator")
        for key, val in params.items():
            if (key == "groupby_cols") and is_fitted:
                raise ValueError(
                    f"Setting {key!r} is forbidden when GroupRegressor is"
                    " fitted. Redefine GroupRegressor instead."
                )
            elif key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {valid_params!r}."
                )
            else:
                setattr(self, key, val)

        if ("base_estimator" in params.keys()) and is_fitted:
            wrn.warn(
                "'base_estimator' was reset. To take effect, "
                "GroupRegressor must be refit."
            )

        if be_params:
            self.base_estimator.set_params(**be_params)

        return self
