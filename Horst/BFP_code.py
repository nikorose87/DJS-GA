#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 10:01:50 2021
Clean code to process data paper Human Body Fat Percentage Prediction Using XGBoost Algorithm
@author: enprietop y ccuervo
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import statsmodels.api as sm

#Loading the preprocessed dataset
root = os.getcwd()
os.chdir("/home/enprietop/Dropbox/Guardados/")
marker_data = pd.read_csv('Horst_markers_raw.csv',sep=',', header=[0,1], index_col=[0],
                          low_memory=False)
subject_info = pd.read_csv('Gait_subject_info.csv',sep=';', decimal=',', index_col=[0])
sublabels = ['Gender', 'Age', 'Body Mass', 'Height']
subject_info.columns = sublabels
os.chdir(root)

#Fixing a mistake in labelling the level 1
marker_col_0 = list(marker_data.columns.get_level_values(0).unique())
new_marker_col_0 = ['{}_{}'.format(i,j) for i in marker_col_0 for j in ['X', 'Y', 'Z']]
marker_col_1 = list(marker_data.columns.get_level_values(1))
marker_col_1 = [i[:13] for i in marker_col_1]
marker_col_1 = set(marker_col_1)
new_cols = pd.MultiIndex.from_product([new_marker_col_0,list(marker_col_1)])
marker_data.columns = new_cols

#Letting trials being the samples

marker_data_new = marker_data.T
marker_data_new = marker_data_new.unstack(level=0)

#Reconstucting the index in order to establish the label
subjects = [i[:3] for i in marker_data_new.index]
trials = [i[5:8] for i in marker_data_new.index]
new_idx = pd.MultiIndex.from_arrays([subjects, trials])
marker_data_new.index = new_idx


for i in sublabels:
    marker_data_new[i] = 0

idx = pd.IndexSlice
for ind in subject_info.index:
    for num, j in enumerate(sublabels):
        marker_data_new.loc[idx[ind,:], j] = subject_info.loc[ind, subject_info.columns[num]]

marker_data_new['BMI'] = marker_data_new['Body Mass'] / np.power(marker_data_new['Height'],2)

#Formula to be applied BF% = 76.0 - 1097.8 * (1/BMI) - 20.6 * sex
       # + 0.053 * age + 154 * sex * (1/BMI)
       # + 0.034 * sex * age
      
#Replacing males as ones a females as 0
marker_data_new['Gender'] = marker_data_new['Gender'] -1
marker_data_new['Gender'] = marker_data_new['Gender'].replace(-1, 1)

BFP_formula = lambda age, bmi, sex: 76.0 - 1097.8 * (1/bmi) - 20.6 * \
    sex + 0.053 * age + 154 * sex * (1/bmi) + 0.034 * sex * age

marker_data_new['BFP'] = BFP_formula(marker_data_new.Age,
                                     marker_data_new.BMI,
                                     marker_data_new.Gender)

#Describe in the paper some data about max (32.12) and mins (9.40) in BFP

# =============================================================================
# Performing the ML model
# =============================================================================

marker_data_new = marker_data_new.dropna(axis=1)
#Mention that features are reduced from 16200 to 14300

#Droppiing the sublabels
marker_data_new = marker_data_new.drop(sublabels, axis=1)

train_features, test_features, train_target, test_target = train_test_split(marker_data_new.iloc[:,:-1],
                                                                            marker_data_new['BFP'],
                                                    train_size=0.8, test_size=0.2, random_state=42)



exported_pipeline = make_pipeline(
    PCA(iterated_power=2, svd_solver="randomized"),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=5, 
                                             min_child_weight=11, n_estimators=100, 
                                             n_jobs=-1, objective="reg:squarederror", 
                                             subsample=1.0, verbosity=0)),
    LassoLarsCV(normalize=False)
)

set_param_recursive(exported_pipeline.steps, 'random_state', 0)

exported_pipeline.fit(train_features, train_target)

prediction = exported_pipeline.predict(test_features)
print("MSE = "+str(mean_squared_error(test_target, prediction)))
print("R2 = "+str(r2_score(test_target, prediction)))
errors = list()
for i in range(len(prediction)):
    err = (test_target[i] - prediction[i])**2
    errors.append(err)
plt.scatter(prediction, errors)
plt.xlabel('Predicted Value')
plt.ylabel('Squared Error')
plt.title('Predicted values vs SE per sample')
plt.show()

test_target = test_target[:,np.newaxis]
prediction = prediction[:,np.newaxis]
errors = np.asarray(errors).reshape((227,1))
results_df = pd.DataFrame(data=np.hstack((test_target,prediction,errors)),columns=['Test target BF%', 'Predicted BF%', 'SE'])

f, ax = plt.subplots(1, figsize = (7,5))
sm.graphics.mean_diff_plot(test_target, prediction, ax = ax)
plt.title('Mean difference plot')
plt.show()











