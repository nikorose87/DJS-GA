import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -5.803243275441528
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.042),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.1, tol=0.0001)),
    Nystroem(gamma=0.30000000000000004, kernel="poly", n_components=3),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.45, tol=0.001)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=9, min_samples_leaf=7, min_samples_split=4)),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="huber", max_depth=7, max_features=0.2, min_samples_leaf=8, min_samples_split=19, n_estimators=100, subsample=0.6000000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
