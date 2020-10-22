#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:33:07 2020
Script to analyse only the Horst raw data set
@author: nikorose
"""

from DJSFunctions import extract_preprocess_data, ankle_DJS
from plot_dynamics import plot_dynamic, plot_ankle_DJS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

Horst_ = extract_preprocess_data('/home/eprietop/enprietop@unal.edu.co/'+
                                 'Tesis de Doctorado/Gait Analysis Data/Downloaded/Horst et al./Nature_paper/Horst_Nature_paper.csv', 
                                    dir_loc=None,
                                    header=[0])

Horst_df = Horst_.all_dfs

idx= pd.IndexSlice

Horst_QS = Horst_df.loc[idx[['ankle_angle_r','ankle_angle_r_moment',
                             'ground_force_1_v_2'], :]]

#According to Winter https://ouhsc.edu/bserdac/dthompso/web/gait/epow/pow1.htm
#in order to calculate the power plot we should do the following:
#Obtain the derivative of the angle plot to obtain angular velocity
#make the product between the moment and the ang vel, then P=Mw

def multi_idx(name, df_):
    index_mi = pd.MultiIndex.from_product([[name],df_.index])
    df_.index = index_mi
    return df_
    

ang_vel = Horst_QS.loc['ankle_angle_r',:].apply(lambda x: np.gradient(x), axis=0)
ang_vel = multi_idx('Angular vel [deg / GC]', ang_vel)
Horst_QS = pd.concat([Horst_QS, ang_vel], axis=0)

#Because moments are negatives because orientation axis, we're gonna change the sign
Horst_QS.loc['ankle_angle_r_moment'] = Horst_QS.loc['ankle_angle_r_moment',:].apply(lambda x: -x).values

def power(col):
    return col['ankle_angle_r_moment']*col['Angular vel [deg / GC]']

power = Horst_QS.apply(power, axis=0)
power = multi_idx('Power [W]', power)

Horst_QS = pd.concat([Horst_QS, power], axis=0)

#Removing angular vel
Horst_QS = Horst_QS.drop(['Angular vel [deg / GC]'], axis =0)

#Replacing index names to be similar with Ferrarin
idx_names = ['Ankle Dorsi/Plantarflexion Deg [°]', 'Ankle Dorsi/Plantarflexion [Nm/kg]',
             'Vertical Force [N]',  'Ankle [W/kg]'][::-1]

idx_old = list(Horst_QS.index.get_level_values(0).unique())


for num, name in enumerate(idx_names[::-1]):
    Horst_QS.index = Horst_QS.index.set_levels(\
                    Horst_QS.index.levels[0].str.replace(idx_old[num], name), level=0)
        
# In order to be proportional with the levels we are going to create another level of means only
idx_col = pd.MultiIndex.from_product([Horst_QS.columns, ['mean']])
Horst_QS.columns = idx_col

# =============================================================================
# Until here the dataframe is ready to be loaded in the object
# =============================================================================

Horst_ = ankle_DJS(Horst_QS, exp_name = 'Horst exp analysis')

all_dfs = Horst_.extract_df_DJS_data(idx=[0,2,1,3])
df_turn = Horst_.get_turning_points(turning_points= 6, 
                            smoothing_radius = 4, cluster_radius= 15)
# Horst_.deg_to_rad()
Horst_.energy_calculation()
#Sensitive results may vary when integrating degrees, the best is to do in radians
Horst_.deg_to_rad()
total_work = Horst_.total_work()    

# =============================================================================
# Plotting dynamic parameters
# =============================================================================
# plot_dyn = plot_dynamic(SD=False, save=True, plt_style='bmh')

# # Plotting power information

# fig2 = plot_dyn.gait_plot(Horst_.all_dfs_ankle, 
#                     cols = np.r_[:15],
#                     rows = None,
#                     title='Ankle dynamics of Sub 1 in Horst data')

# =============================================================================
# Obtaining the mechanical work through power instances in regular walking 
# =============================================================================
idx= pd.IndexSlice
work_df = Horst_.power_energy.loc[idx[: , 'mean'], :]
zero_ro = Horst_.energy_fun.min_max_power(Horst_.power_ankle)

# =============================================================================
# Loading the predicted data 
# =============================================================================

pred_reg = pd.read_csv('Horst/predicted_data_raw.csv', index_col=[0,1])

Horst_pred = Horst_.all_dfs_ankle.loc[:,idx[pred_reg.index.get_level_values(0)]]
# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
DJS = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=False)

fig4 = DJS.plot_DJS(Horst_pred, 
                    cols=[7], rows= np.r_[0,2],
                    title="Ankle DJS subject 1", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)

# =============================================================================
# # Saving data to df
# =============================================================================
# DJS.reg_info_df.to_csv('Horst/Horst_reg_lines_raw.csv')
# df_turn.to_csv('Horst/Horst_tp_raw.csv')

pred_data = DJS.add_reg_lines(pred_reg, label='TPOT')


# =============================================================================
# Performing the RF prediction to compare with the Meta algorithm
# =============================================================================

def RF_pred(label_to_pred):
    idx = pd.IndexSlice
    input_horst = pd.read_csv('Horst/Horst_marker_distances_anthropometry.csv',index_col=0)
    output_horst = pd.read_csv('Horst/Horst_reg_lines_raw.csv', index_col=[0,1])

    #Multiplying the features for each
    input_horst_amp = pd.DataFrame(np.empty((output_horst.shape[0], 
                               input_horst.shape[1])), index=output_horst.index,
                               columns = input_horst.columns)
    input_horst_amp[:] = np.nan
    for row_base in input_horst.index:
        for num, row_amp in enumerate(input_horst_amp.index.get_level_values(0)):    
            if row_base in row_amp:
                input_horst_amp.iloc[num] = input_horst.loc[row_base]
    #Creating the df for the output
    pred_output = pd.DataFrame(np.empty((output_horst.shape[0], 2)), 
                               index=output_horst.index, columns=['intercept', 'stiffness'])
    for col in ['ERP', 'LRP', 'DP']: 
        for var in ['intercept', 'stiffness']:
            X_train, X_test, y_train, y_test = train_test_split(input_horst_amp.loc[idx[:,col], :], 
                                                                output_horst.loc[idx[:, col],:][var], 
                                                                test_size=.2, \
                                                                random_state=42)
            df_params = pd.read_csv('Horst/Best_params_RF_for_{}_in_{}.csv'.format(col, var), 
                                    index_col=[0])
            df_params.columns = ['Values']
            #to float 
            for ind in df_params.index:
                try:
                    if pd.isnull(df_params.loc[ind]).values:
                        df_params.loc[ind] = None
                    df_params.loc[ind] = df_params.loc[ind].astype(int)
                except (ValueError, TypeError) as e:
                    continue
            #to dict
            df_dict = df_params.to_dict()
            rf_ = RandomForestRegressor(random_state=42, **df_dict['Values'])
            rf_.fit(X_train, y_train)
            #Predicting the data
            y_pred = rf_.predict(X_test)
            print('The score using RME in {} for {} is {}'.format(col, var,
                                                                  mean_squared_error(y_test, 
                                                                  y_pred)))
            #Taking the same variables to pred
            input_pred = input_horst_amp.loc[idx[label_to_pred, col], :]
            pred_output.loc[idx[label_to_pred, col], var] = rf_.predict(input_pred)
    #Dropping nan
    pred_output = pred_output.dropna()
    #Removing non predicted categories
    pred_output = pred_output.loc[idx[label_to_pred, :], :]
    return pred_output


# pred_RF = RF_pred(pred_reg.index.get_level_values(0).unique())
# pred_data_RF = DJS.add_reg_lines(pred_RF, label='RF')


            

            
            
            
