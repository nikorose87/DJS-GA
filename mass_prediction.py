#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:28:42 2020

@author: nikorose
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from GPyOpt.methods import BayesianOptimization
from tpot import TPOTRegressor

bayes_opt =False
topopt_ = True



# =============================================================================
# Data preprocessing
# =============================================================================

processed_data = pd.read_csv('Horst/Horst_markers_mean.csv',sep=',',
                             header=[0,1,2], index_col=[0,1], low_memory=False)

processed_data.columns.names = ['Coordinate', 'Subject', 'Marker']
processed_data.index.names = ['Feature', 'GC']

#Stacking the first level (Coordinates)
processed_data_mod = processed_data.stack(level=0)
#Stacking markers
processed_data_mod = processed_data_mod.stack(level=1)
# Transposing to keep subjects as samples and the features in columns (57,16200)
processed_data_mod = processed_data_mod.T
#Dropping nan values
processed_data_mod = processed_data_mod.dropna(axis=1) #100 columns were erased, why those nan values?

# Subject information 
anthro_info = pd.read_csv('Horst/Gait_subject_info.csv', sep=";",
                          decimal=',')
anthro_info.columns = ['ID','gender','age','mass','height']   

#Predicted var 
var_to_pred = anthro_info['mass']

# =============================================================================
# Analyzing the number of components first 
# ============================================================================

var = []
for comp in range(1,50):
  pc = PCA(n_components = comp)
  Horst_pca = pc.fit_transform(processed_data_mod)
  var.append(sum(pc.explained_variance_ratio_))

#Seeing the variance elbow
fig, ax = plt.subplots(1,1)
ax.plot(range(len(var)), var)
plt.ylabel('Variance')
plt.xlabel('n° of components')
plt.show()
#According to this results we would like to include 6 components at least

if bayes_opt:
    
    bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
            {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
            {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
            {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 1000)},
            {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 500)}]

    best_params= {}
    for n_comp in range(6,50):
        pc = PCA(n_components = comp)
        processed_data_pca =  pc.fit_transform(processed_data_mod)
        
        #Let us divide the dataset into 2 splits
        x_train, x_test, y_train, y_test = train_test_split(processed_data_pca, 
                                                            var_to_pred, 
                                                            test_size=0.2, random_state=42)
        # Optimization objective 
        def cv_score(parameters):
            parameters = parameters[0]
            score = cross_val_score(
                        XGBRegressor(learning_rate=parameters[0],
                                      gamma=int(parameters[1]),
                                      max_depth=int(parameters[2]),
                                      n_estimators=int(parameters[3]),
                                      min_child_weight = parameters[4]), 
                                      processed_data_pca, var_to_pred, 
                                      n_jobs=-1, scoring='neg_mean_squared_error').mean()
            score = np.array(score)
            return score
        
        
        # =============================================================================
        # Beginning the Bayesian optimization to determine the best hyperparams
        # =============================================================================
    
        #Setting the optimization parameters
        optimizer = BayesianOptimization(f=cv_score, 
                                          domain=bds,
                                          model_type='GP',
                                          acquisition_type ='EI',
                                          acquisition_jitter = 0.05,
                                          exact_feval=True, 
                                          maximize=True)
        
        # Only 20 iterations because we have 5 initial random points
        optimizer.run_optimization(max_iter=50)
        optimizer.save_report('Horst/Opt_files/mass_pred_{}_pca.txt'.format(n_comp))
        optimizer.save_evaluations('Horst/Opt_files/mass_pred_{}_eval.txt'.format(n_comp))
        optimizer.plot_convergence('Horst/Opt_files/mass_pred_{}_pca.png'.format(n_comp))
    
        #Saving x opt and result
        best_params.update({'{} comp'.format(n_comp): (optimizer.x_opt, optimizer.Y_best)})
        
        
        
    # =============================================================================
    # Proving the R2 with the best values
    # =============================================================================
    pc = PCA(n_components = 11)
    processed_data_pca =  pc.fit_transform(processed_data_mod)
    xgb = XGBRegressor(learning_rate=0.722,
                       gamma=3.7759,
                       max_depth=50, 
                       n_estimators=500,
                       min_child_weight=100)
    #Let us divide the dataset into 2 splits
    x_train, x_test, y_train, y_test = train_test_split(processed_data_mod, 
                                                        var_to_pred, 
                                                        test_size=0.2, random_state=42)
    xgb.fit(x_train, y_train)
    pred_test = xgb.predict(x_test)
    score = xgb.score(x_test, y_test)


    
if topopt_:
    for n_comp in range(6,50):
        pc = PCA(n_components = comp)
        processed_data_pca =  pc.fit_transform(processed_data_mod)
        
        #Let us divide the dataset into 2 splits
        x_train, x_test, y_train, y_test = train_test_split(processed_data_pca, 
                                                            var_to_pred, 
                                                            test_size=0.2, random_state=42)

        pipeline_regressor = TPOTRegressor(generations=100, population_size=100, cv=3,
                                            random_state=42, verbosity=2, 
                                            early_stop=10, n_jobs=-1, 
                                            scoring='neg_mean_squared_error')
        
        pipeline_regressor.fit(x_train, y_train)
        
        print('The best score for the metamodel was {}'.format(pipeline_regressor.score(x_test, y_test)))
        
        predicted_regression = pd.Series(pipeline_regressor.predict(x_test), name='Predicted', \
                              index=y_test.index)
            
        comp_table = pd.concat([y_test, predicted_regression], axis=1)
        comp_table.to_csv('Horst/Opt_files/comp_table_{}'.format(n_comp))
        pipeline_regressor.export('Horst/Opt_files/Regression_pipeline_withTPOT_{}_n_comp.py'.format(n_comp))

