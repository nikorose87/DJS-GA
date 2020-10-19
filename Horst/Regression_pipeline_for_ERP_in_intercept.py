import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import OneHotEncoder, StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -32.70130300130747
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    PCA(iterated_power=1, svd_solver="randomized"),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.25, min_samples_leaf=7, min_samples_split=14, n_estimators=100)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    RobustScaler(),
    DecisionTreeRegressor(max_depth=9, min_samples_leaf=9, min_samples_split=15)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
