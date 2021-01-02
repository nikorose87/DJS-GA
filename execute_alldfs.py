#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:37:14 2020
Ferrain and horst Together
@author: nikorose
"""
from DJSFunctions import plot_ankle_DJS, ankle_DJS
import os
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from utilities_QS import ttest





# =============================================================================
# Ferrarin execution 
# =============================================================================

#Excluding not regular intentions
exclude_list = ["{} {}".format(i,j) for i in ['Toe', 'Heel', 'Descending', 
                                            'Ascending'] for j in ['A','Y']]

# exclude_list.extend(['Free A', 'Very Slow A', 'Slow A', 'Medium A', 'Fast A'])

Ferrarin_ = ankle_DJS('mmc3.xls', 
                      dir_loc = 'Ferrarin',
                      exp_name = 'Ferrarin analysis',
                      exclude_names = exclude_list)

all_dfs_ferra = Ferrarin_.extract_DJS_data()
#Changing labels
all_dfs_ferra = Ferrarin_.change_labels([ r'$0.363 < v* < 0.500$ Y',
                                          r"$v/h<0.6$ Y",
                                          r'$0.6 < v/h < 0.8$ Y',
                                          r'$0.8 < v/h < 1$ Y',
                                          r'$v/h > 1.0$ Y', 
                                           r'$0.363 < v* < 0.500$ A',
                                           r'$v/h < 0.6$ A', #
                                           r'$0.6 < v/h < 0.8$ A',
                                           r'$0.8 < v/h < 1$ A',
                                           r'$v/h > 1.0$ A'])

df_turn_ferra = Ferrarin_.get_turning_points(turning_points= 6, 
                            param_1 = 4, cluster_radius= 15)
# Ferrarin_.deg_to_rad()
Ferrarin_.energy_calculation()
#Sensitive results may vary when integrating degrees, the best is to do in radians
Ferrarin_.deg_to_rad()
total_work_ferra = Ferrarin_.total_work()

# =============================================================================
# Obtaining the mechanical work through power instances in regular walking Ferra
# =============================================================================
idx= pd.IndexSlice
work_df_ferra = Ferrarin_.power_energy.loc[idx[: , 'mean'], :]
zero_ro_ferra = Ferrarin_.energy_fun.min_max_power(Ferrarin_.power_ankle)

# =============================================================================
# Schwartz execution 
# =============================================================================

Schwartz_ = ankle_DJS('Schwartz.xls', 
                      dir_loc = 'Schwartz',
                      exp_name = 'Schwartz analysis',
                      features= ['Ankle Dorsi/Plantarflexion', 
                                  'Vertical',
                                  'Ankle Dorsi/Plantarflexion',
                                  'Ankle'])

all_dfs_schwartz = Schwartz_.extract_DJS_data()
all_dfs_schwartz = Schwartz_.change_labels([r'$v* < 0.227$ CH',r'$0.227 < v* < 0.363$ CH',r'$0.363 < v* < 0.500$ CH',
                                            r'$0.500 < v* < 0.636$ CH','$v* > 0.636$ CH'])
df_turn_schwartz = Schwartz_.get_turning_points(turning_points= 6, 
                           param_1 = 2, cluster_radius= 8)
# Schwartz_.deg_to_rad()
Schwartz_.energy_calculation()
#Sensitive results may vary when integrating degrees, the best is to do in radians
Schwartz_.deg_to_rad()
total_work_schwartz = Schwartz_.total_work()

# =============================================================================
# Obtaining the mechanical work through power instances in regular walking Ferra
# =============================================================================
work_df_schwartz = Schwartz_.power_energy.loc[idx[: , 'mean'], :]
zero_ro_schwartz = Schwartz_.energy_fun.min_max_power(Schwartz_.power_ankle)


# =============================================================================
# concatenating DFs
# =============================================================================

concat_gait = pd.concat([Ferrarin_.all_dfs_ankle, Schwartz_.all_dfs_ankle], axis=1)
concat_gait = concat_gait.dropna(axis=0)
concat_gait = concat_gait.reindex(Schwartz_.index_ankle.get_level_values(0).unique(), 
                                  level=0, axis=0)

# =============================================================================
# Obtaining new values for the concatenated df
# =============================================================================
concat_ = ankle_DJS(concat_gait, exp_name = 'Concat Ferrarin and Schwartz analysis')

all_dfs = concat_.extract_df_DJS_data(idx=[0,2,1,3], units=False)

df_turn = concat_.get_turning_points(turning_points= 6, 
                            param_1 = 2, cluster_radius= 8) 

#Tuning 
df_turn.iloc[2,:] = np.array([3,10,21,32,42]) # -> sr_2,cr_6 Y medium
df_turn.iloc[8,:] = np.array([2,11,19,31,42]) # -> sr_2,cr_7 A medium 
df_turn.iloc[-2,:] = np.array([3,12,23,31,42]) # -> sr_6,cr_7 CH fast
df_turn.iloc[-1,:] = np.array([4,15,23,31,43]) # -> sr_6,cr_7 CH V fast
df_turn.iloc[4,:] = np.array([3,11,19,31,43]) # -> sr_2,cr_7 Y V fast
df_turn.iloc[9,:] = np.array([3,14,22,30,41]) # -> sr_2,cr_7 A Fast
#Sensitive results may vary when integrating degrees, the best is to do in radians
concat_.deg_to_rad()
total_work = concat_.total_work()   
# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
params = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 'color_reg':['black']*20, 
          'color_symbols': ['slategray']*20, 'arr_size': 10, 'left_margin': 0.1}

DJS_all_ch = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[2,5],
                      alpha=2, fig_size=[3,7], params=params, ext='png')
DJS_all_ch.colors = Color
#Previuos config for Y, A and CH
#np.r_[10,1,6,11,2,7,12,0,5,13,3,8,14,4,9]
# config for more rows np.r_[5,1,6,2,7,0,8,3,9,4]
fig4 = DJS_all_ch.plot_DJS(concat_.all_dfs_ankle, 
                    cols=np.r_[1,2,0,3,4,10:15], rows= np.r_[0,1],
                    title="Individual ankle DJS Children", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)

reg_info_concat_ch = DJS_all_ch.reg_info_df.round(3)
# =============================================================================
# Best params
# =============================================================================

# df_turn_ = hyperparams(all_dfs.loc[:,idx['$0.500 < v* < 0.636$ CH',:]], 
#                         smooth_radius=(2,8), c_radius=(6,12), R2=True) #


# =============================================================================
# Obtaining the ttest of children (Schwartz) against youth (Ferrarin)
# =============================================================================

cols_ch = df_turn.index
cols_ch = cols_ch[np.r_[1,2,0,3,4,10:15]]
etiquete = ['VS','S','C','F','VF']
#Dropping GRF and powers, we are interested in QS only
df_QS_ch = concat_.all_dfs_ankle
df_QS_ch = df_QS_ch.drop(['Vertical  Force [%BH]', 'Ankle  [W/kg]'], axis=0, level=0)

#Samples of each experiment
n_schwartz = [77, 82, 82, 76, 51] #Very Slow Slow Free Fast Very Fast
n_ferra_y = [34, 76, 111, 71, 100, 83, 51, 75, 67] #XS S Natural M L T H A D
n_ferra_a = [140, 110, 124, 68, 52, 124, 85, 73, 72] #XS S Natural M L T H A D

tt_ch = pd.concat([ttest(df_QS_ch[cols_ch[i]],
                         df_QS_ch[cols_ch[i+5]], 
                         samples=[n_schwartz[i], n_ferra_y[i]], 
                         name='Ttest_{}'.format(etiquete[i]), method='scipy') for i in range(5)], axis=1)

tt_angles_ch = tt_ch.loc['Ankle Dorsi/Plantarflexion  Deg [°]'].mean(axis=0)
tt_moments_ch = tt_ch.loc['Ankle Dorsi/Plantarflexion  [Nm/kg]'].mean(axis=0)


# =============================================================================
# Showing the DJS results for youths and adults
# =============================================================================

DJS_all_ad = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[2,5],
                      alpha=2, fig_size=[3,7], params=params, ext='png')
DJS_all_ad.colors = Color
#Previuos config for Y, A and CH
#np.r_[10,1,6,11,2,7,12,0,5,13,3,8,14,4,9]
# config for more rows np.r_[5,1,6,2,7,0,8,3,9,4]
fig5 = DJS_all_ad.plot_DJS(concat_.all_dfs_ankle, 
                    cols=np.r_[1,2,0,3,4,6,7,5,8,9], rows= np.r_[0,1],
                    title="Individual ankle DJS Y v A", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)

reg_info_concat_ad = DJS_all_ad.reg_info_df.round(3)


# =============================================================================
# Obtaining the ttest of adults vs youth (Ferrarin)
# =============================================================================

cols_ad = df_turn.index
cols_ad = cols_ad[np.r_[1,2,0,3,4,6,7,5,8,9]]
etiquete = ['VS','S','C','F','VF']
#Dropping GRF and powers, we are interested in QS only
df_QS_ad = concat_.all_dfs_ankle
df_QS_ad = df_QS_ad.drop(['Vertical  Force [%BH]', 'Ankle  [W/kg]'], axis=0, level=0)

tt_ad = pd.concat([ttest(df_QS_ad[cols_ad[i]],
                         df_QS_ad[cols_ad[i+5]], 
                         samples= [n_ferra_y[i], n_ferra_a[i]], 
                         name='Ttest_{}'.format(etiquete[i]), 
                         method='scipy') for i in range(5)], axis=1)

tt_angles_ad = tt_ad.loc['Ankle Dorsi/Plantarflexion  Deg [°]'].mean(axis=0)
tt_moments_ad = tt_ad.loc['Ankle Dorsi/Plantarflexion  [Nm/kg]'].mean(axis=0)

#Conclusion
#Significant differences were found in terms of angles, whereas for ankle moment no significant 
#differences were found with p value < 0.05

#The statistic conclusion according with
#https://stats.stackexchange.com/questions/339243/r-welch-two-sample-t-test-t-test-interpretation-help
