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
import csv
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


build_df = False
make_pred_TPOT =False
make_RF =True
reducing = False
writing = False
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

# =============================================================================
# functions
# =============================================================================
def RandomForest_pred(X_train, X_test, y_train, y_test, writing=False, n_iter=20):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 60)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 200, num = 30)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 4, 5, 10, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    #n_componenst
    n_comp = range(2,35)
    gamma = range(1,25)
    kernel = ['rbf', 'poly', 'sigmoid']
    
    # Creating the random grid
    random_grid = {'regressor__n_estimators': n_estimators,
    'regressor__max_features': max_features,
    'regressor__max_depth': max_depth,
    'regressor__min_samples_split': min_samples_split,
    'regressor__min_samples_leaf': min_samples_leaf,
    'regressor__bootstrap': bootstrap,
    'reducer__n_components': n_comp,
    'reducer__gamma': gamma,
    'reducer__kernel': kernel}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('reducer', KernelPCA()),
                         ('regressor', RandomForestRegressor(random_state = 42))])
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=pipeline, param_distributions=random_grid,
                                    n_iter = n_iter, scoring='neg_mean_squared_error',
                                    cv = 3, verbose=1, random_state=42, n_jobs=-1,
                                    return_train_score=True)
    rf_random.fit(X_train, y_train)
    predicted_regression = pd.Series(rf_random.predict(X_test), name='Predicted', \
                          index=y_test.index)
    #concatenating best params obtained with the accuracy
    rmse= mean_squared_error(y_test, predicted_regression)
    comp_table = pd.concat([y_test, predicted_regression], axis=1)
    print('The score using RME in {} for {} is {}'.format(col, var,
                                              rmse))
    results = rf_random.best_params_
    model = rf_random.best_estimator_

    if writing:
        comp_table.to_csv('Horst/comp_table_RF_{}_in_{}'.format(col, var))
        # Exporting to a csv file
        a_file = open('Horst/Best_params_RF_for_{}_in_{}.csv'.format(col, var), "w")
        writer = csv.writer(a_file)
        for key, value in results.items():
            writer.writerow([key, value])
        
        a_file.close()
    return model, results, rmse
for row_base in input_horst.index:
    for num, row_amp in enumerate(input_horst_amp.index.get_level_values(0)):    
        if row_base in row_amp:
            input_horst_amp.iloc[num] = input_horst.loc[row_base]


RFs = {}
for col in ['ERP', 'LRP', 'DP']: #,
    for var in ['intercept', 'stiffness']:
    
        X_train, X_test, y_train, y_test = train_test_split(input_horst_amp.loc[idx[:,col], :], 
                                                            output_horst.loc[idx[:, col],:][var], 
                                                            test_size=.2, \
                                                            random_state=42)
        if reducing:
            for n_comp in range(3,20):
                pc = PCA(n_components=n_comp)
                X_train_red = pc.fit_transform(X_train)
                X_test_red = pc.transform(X_test)
                print("The variance with {} components is {}".format(n_comp, 
                                         sum(pc.explained_variance_ratio_)),
                                         "for {} in {}".format(col, var))
                model, result, rmse = RandomForest_pred(X_train_red, X_test_red, 
                                                        y_train, y_test, n_iter=3)
                if n_comp == 3:
                    RFs['model_{}_{}'.format(col,var)] = [model]
                    RFs['results_{}_{}'.format(col,var)] = [result]
                    RFs['rmse_{}_{}'.format(col,var)] = [rmse]
                else:
                    RFs['model_{}_{}'.format(col,var)].append(model)
                    RFs['results_{}_{}'.format(col,var)].append(result)
                    RFs['rmse_{}_{}'.format(col,var)].append(rmse)
        else:
            model, result, rmse = RandomForest_pred(X_train, X_test, y_train, 
                                                    y_test, n_iter=100)
            RFs['model_{}_{}'.format(col,var)] = [model]
            RFs['results_{}_{}'.format(col,var)] = [result]
            RFs['rmse_{}_{}'.format(col,var)] = [rmse]
            
        if make_pred_TPOT:
          
            pipeline_regressor = TPOTRegressor(generations=100, population_size=100, cv=3,
                                                random_state=42, verbosity=2, n_jobs=-1,
                                                early_stop=15, scoring='neg_mean_absolute_error')
            
            pipeline_regressor.fit(X_train, y_train)
            
            print('The best score for the metamodel was {}'.format(pipeline_regressor.score(X_test, y_test)))
            
            predicted_regression = pd.Series(pipeline_regressor.predict(X_test), name='Predicted', \
                                  index=y_test.index)
                
            comp_table = pd.concat([y_test, predicted_regression], axis=1)
            comp_table.to_csv('Horst/comp_table_{}_in_{}'.format(col, var))
            pipeline_regressor.export('Horst/Regression_pipeline_for_{}_in_{}.py'.format(col, var))
            

            


    
# =============================================================================
# Consolidating the predicted results as one df
# =============================================================================
if build_df:
    pred_output = pd.DataFrame(np.empty((output_horst.shape[0], 2)), 
                               index=output_horst.index, columns=['intercept', 'stiffness'])
    
    for col in ['ERP', 'LRP', 'DP']: #,
        for var in ['intercept', 'stiffness']:
            df_ = pd.read_csv('Horst/comp_table_{}_in_{}'.format(col, var), index_col=[0,1])
            df_.columns = ['intercept', 'stiffness']
            pred_output.loc[idx[:, col], var] = df_.loc[:,var]
    
    pred_output = pred_output.dropna()
    # pred_output.to_csv('Horst/predicted_data_raw.csv')


# =============================================================================
# Plotting the RMSE vs PC
# =============================================================================
if reducing:
    fig1, axs = plt.subplots(3,2, figsize=(10,10))
    ax = np.ravel(axs)
    count = 0
    for col in ['ERP', 'LRP', 'DP']: #,
        for var in ['intercept', 'stiffness']:
            key = 'rmse_{}_{}'.format(col,var)
            series = pd.Series(RFs[key], index=range(3,20), 
                      name=key)
            axes = sns.lineplot(data=series, ax= ax[count])
            axes.set(ylabel = key)
            count +=1
    fig1.suptitle('Neg MSE in QS prediction for 20 components')
    fig1.savefig('Figures/RFscorewithPCA.eps')
