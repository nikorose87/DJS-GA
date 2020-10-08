#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:31:21 2020
Script tp discover if the given df to detect any features for the stiffness 
or any other target is feasible
@author: nikorose
"""

import pandas as pd 
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

# =============================================================================
# Loading the data
# =============================================================================

input_horst = pd.read_csv('Horst/Horst_marker_distances_anthropometry.csv',index_col=0)
output_horst = pd.read_csv('Horst/Horst_reg_lines_raw.csv', index_col=[0,1])

idx = pd.IndexSlice

ERP = output_horst.loc[idx[:,'ERP'],:]
LRP = output_horst.loc[idx[:,'LRP'],:]
DP = output_horst.loc[idx[:,'DP'],:]

input_horst_amp = pd.DataFrame(np.empty((output_horst.shape[0], 
                               input_horst.shape[1])), index=output_horst.index,
                               columns = input_horst.columns)

for row_base in input_horst.index:
    for num, row_amp in enumerate(input_horst_amp.index.get_level_values(0)):    
        if row_base in row_amp:
            input_horst_amp.iloc[num] = input_horst.loc[row_base]

for col in ['ERP', 'LRP', 'DP']: #,
    for var in ['intercept', 'stiffness']:
    
        X_train, X_test, y_train, y_test = train_test_split(input_horst_amp.loc[idx[:,col], :], 
                                                            output_horst.loc[idx[:, col],:][var], 
                                                            test_size=.2, \
                                                            random_state=42)
          
        pipeline_regressor = TPOTRegressor(generations=100, population_size=100, cv=3,
                                            random_state=42, verbosity=2, n_jobs=-1,
                                            early_stop=15, scoring='neg_mean_squared_error')
        
        pipeline_regressor.fit(X_train, y_train)
        
        print('The best score for the metamodel was {}'.format(pipeline_regressor.score(X_test, y_test)))
        
        predicted_regression = pd.Series(pipeline_regressor.predict(X_test), name='Predicted', \
                              index=y_test.index)
            
        comp_table = pd.concat([y_test, predicted_regression], axis=1)
        comp_table.to_csv('Horst/comp_table_{}_in_{}'.format(col, var))
        pipeline_regressor.export('Horst/Regression_pipeline_for_{}_in_{}.py'.format(col, var))
