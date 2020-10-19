import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive


def DP_intercept(training_features, testing_features, training_target, testing_target):
    # Average CV score on the training set was: -80.64562030506049
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=28, p=2, weights="distance")),
        RBFSampler(gamma=0.75),
        StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=20, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.5)),
        StackingEstimator(estimator=DecisionTreeRegressor(max_depth=9, min_samples_leaf=16, min_samples_split=6)),
        DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, min_samples_split=20)
    )
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 42)
    
    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_target)
    
    return results
    
