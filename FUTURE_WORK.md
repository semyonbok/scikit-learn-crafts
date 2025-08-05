## Regressors
### GroupRegressor
* features:
    - [ ] parallelize predict
    - [ ] constant (or constant per group) rather than zeros
    - [X] use protocol to typehint and check `base_regressor` 
    - [ ] allow to group by a numerical feature(s) but bin them first
    - [ ] try to make it work with numpy (perhaps by allowing to pass separate sequence as `groupby_cols` argument)
    - [ ] instead of training models on group splits, train them on entire set but augment the category with higher weights
* tests:
    - [X] create fixture: toy dataset
    - [ ] raising warnings:
        - [X] base estimator reset
        - [ ] unseen group in predictions
    - [ ] missing keys in groups
    - [ ] test if it works in a pipeline
    - [ ] make sure it works in cross-validation
    - [ ] try creating an ensemble with `GroupRegressor`
    - [X] try passing `check_estimator` tests (or at least a subset)
    - [ ] check that predictions for entire test set are consistent with group-specific predictions
### PredictionIntervalRegressor
* features:
* tests:

## Classifiers
### GroupClassifier

## Transformers
### MetaFeatureEncoder(s)
### Splitter
### ResidualBinner
### JSON Preprocessor
### MinFreqOneHot
### TargetEncoder (variant)

## Utilities/Displays
### Enhanced PrecisionRecallDisplay