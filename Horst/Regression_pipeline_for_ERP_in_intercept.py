import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -3.831305433075368
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=0.5, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.001)),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.001, loss="square", n_estimators=100)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.7000000000000001, tol=0.01)),
    XGBRegressor(learning_rate=0.5, max_depth=7, min_child_weight=5, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.7000000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
