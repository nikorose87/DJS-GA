import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -24.57982532992075
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    RBFSampler(gamma=0.6000000000000001),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=15, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.25)),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=48, p=2, weights="distance")),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.1, loss="exponential", n_estimators=100)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=13, min_samples_split=11)),
    GradientBoostingRegressor(alpha=0.85, learning_rate=1.0, loss="huber", max_depth=9, max_features=1.0, min_samples_leaf=5, min_samples_split=17, n_estimators=100, subsample=1.0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
