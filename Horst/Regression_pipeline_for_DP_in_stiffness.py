import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -8.246111780807835
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=1.0, loss="ls", max_depth=10, max_features=0.55, min_samples_leaf=13, min_samples_split=14, n_estimators=100, subsample=0.45)),
    SelectPercentile(score_func=f_regression, percentile=5),
    StackingEstimator(estimator=LinearSVR(C=0.01, dual=True, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.1)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.001, loss="lad", max_depth=10, max_features=0.35000000000000003, min_samples_leaf=9, min_samples_split=14, n_estimators=100, subsample=0.8)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=19, min_samples_split=17)),
    GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=6, max_features=0.8, min_samples_leaf=3, min_samples_split=20, n_estimators=100, subsample=0.7500000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
