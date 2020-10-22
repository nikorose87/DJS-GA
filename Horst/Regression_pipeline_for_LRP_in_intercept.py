import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -15.540873121310648
exported_pipeline = make_pipeline(
    Nystroem(gamma=0.55, kernel="poly", n_components=10),
    SelectPercentile(score_func=f_regression, percentile=66),
    GradientBoostingRegressor(alpha=0.99, learning_rate=0.5, loss="huber", max_depth=9, max_features=0.35000000000000003, min_samples_leaf=19, min_samples_split=15, n_estimators=100, subsample=0.8500000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
