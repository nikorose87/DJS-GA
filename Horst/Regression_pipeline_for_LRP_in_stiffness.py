import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, SGDRegressor
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

# Average CV score on the training set was: -54.948450196778886
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=11, min_samples_split=5, n_estimators=100)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.01, fit_intercept=True, l1_ratio=0.75, learning_rate="constant", loss="epsilon_insensitive", penalty="elasticnet", power_t=1.0)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.9500000000000001, tol=0.1)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=1, min_samples_split=14)),
    GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="lad", max_depth=2, max_features=0.6000000000000001, min_samples_leaf=5, min_samples_split=13, n_estimators=100, subsample=0.4)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
