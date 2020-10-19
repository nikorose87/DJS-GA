import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -287.59299016563995
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.9, learning_rate=1.0, loss="ls", max_depth=4, max_features=0.7000000000000001, min_samples_leaf=11, min_samples_split=20, n_estimators=100, subsample=0.8)),
    SelectPercentile(score_func=f_regression, percentile=4),
    FeatureAgglomeration(affinity="l1", linkage="complete"),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.5, loss="square", n_estimators=100)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.01, fit_intercept=False, l1_ratio=0.75, learning_rate="constant", loss="squared_loss", penalty="elasticnet", power_t=1.0)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=1, min_samples_split=20, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
