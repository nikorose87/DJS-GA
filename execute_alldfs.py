#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:37:14 2020
Ferrain and horst Together
@author: nikorose
"""
from DJSFunctions import *
import os
from pathlib import PurePath
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

def hyperparams(df_, smooth_radius=(4,8), c_radius=(10,14), features=
                ['Ankle Dorsi/Plantarflexion ', 'Vertical Force',
                 'Ankle Dorsi/Plantarflexion',  'Ankle'], R2=False):
    """
    Parameterization of the curvature settings to see which of them are suitable
    for all the gait instances

    Parameters
    ----------
    df_ : dataframe to analyze the best hyperparameters for Smoothing radious.
    smooth_radius : TYPE, optional
        DESCRIPTION. The default is (4,8).
    c_radius : TYPE, optional
        DESCRIPTION. The default is (10,14).
    features : TYPE, optional
        DESCRIPTION. The default is ['Ankle Dorsi/Plantarflexion ', 'Vertical Force',                 
                                     'Ankle Dorsi/Plantarflexion',  'Ankle'].

    Returns
    -------
    df_turn_instance : Dict with dataframes indicating which regression could be done

    """
    df_turn_instance = {}
    for i in range(*smooth_radius):
        for j in range(*c_radius):
            try:
                print('For {}, {} values'.format(i,j))
                _instance = ankle_DJS(df_, features= features)
                
                all_dfs_instance = _instance.extract_df_DJS_data(idx=[0,2,1,3])
                df_turn_instance.update({'sr_{}_cr_{}'.format(i,j): _instance.get_turning_points(turning_points= 6, 
                                            smoothing_radius = i, cluster_radius= j)})
                if R2:
                    DJS_all = plot_ankle_DJS(SD=True, save=True, plt_style='bmh')
                    fig4 = DJS_all.plot_DJS(df_, 
                                        cols=None, rows= np.r_[0,1],
                                        title="Individual ankle DJS C vs Y v A", 
                                        legend=True, reg=df_turn_instance['sr_{}_cr_{}'.format(i,j)],
                                        integration= True, rad = True)
                    print('The R2 for all datasets in {},{} is {}'.format(i,j,
                                            DJS_all.reg_info_df['R2'].mean()))
            except ValueError:
                print('parameters {},{} failed'.format(i,j))
                continue
    return df_turn_instance

# =============================================================================
# Ferrarin execution 
# =============================================================================

#Excluding not regular intentions
exclude_list = ["{} {}".format(i,j) for i in ['Toe', 'Heel', 'Descending', 
                                            'Ascending'] for j in ['A','Y']]
Ferrarin_ = ankle_DJS('mmc3.xls', 
                      dir_loc = 'Ferrarin',
                      exp_name = 'Ferrarin analysis',
                      exclude_names = exclude_list)

all_dfs_ferra = Ferrarin_.extract_DJS_data()
#Changing labels
all_dfs_ferra = Ferrarin_.change_labels([r"$v/h$",
                                          r"$v/h<0.6$",
                                          r'$0.6 < v/h < 0.8$',
                                          r'$0.8 < v/h < 1$',
                                          r'$v/h > 1.0$', 
                                          r'$v/h$ A',
                                          r'$v/h < 0.6$ A', #
                                          r'$0.6 < v/h < 0.8$ A',
                                          r'$0.8 < v/h < 1$ A',
                                          r'$v/h > 1.0$ A'])

df_turn_ferra = Ferrarin_.get_turning_points(turning_points= 6, 
                            smoothing_radius = 4, cluster_radius= 17)
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
all_dfs_schwartz = Schwartz_.change_labels([r'$v* < 0.227$',r'$0.227 < v* < 0.363$',r'$0.363 < v* < 0.500$',
                                            r'$0.500 < v* < 0.636$','$v* > 0.636$'])
df_turn_schwartz = Schwartz_.get_turning_points(turning_points= 6, 
                           smoothing_radius = 2, cluster_radius= 8)
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
concat_ = ankle_DJS(concat_gait, exp_name = 'Concata Ferrarin and Schwartz analysis')


all_dfs = concat_.extract_df_DJS_data(idx=[0,2,1,3], units=False)

df_turn = concat_.get_turning_points(turning_points= 6, 
                            smoothing_radius = 2, cluster_radius= 8) 

#Tuning 
df_turn.loc['$0.6 < v/h < 0.8$'] = np.array([3,10,21,32,42]) # -> sr_2,cr_6
df_turn.loc['$0.8 < v/h < 1$ A'] = np.array([4,12,20,32,43]) # -> sr_3,cr_7
df_turn.loc['$0.500 < v* < 0.636$'] = np.array([0,10,18,33,43]) # -> sr_6,cr_7
df_turn.loc['$v* > 0.636$'] = np.array([7,15,23,31,43]) # -> sr_6,cr_7
df_turn.loc['$v/h > 1.0$'] = np.array([3,11,19,31,43]) # -> sr_2,cr_7
df_turn.loc['$v/h > 1.0$ A'] = np.array([3,14,22,30,41]) # -> sr_2,cr_7
#Sensitive results may vary when integrating degrees, the best is to do in radians
concat_.deg_to_rad()
total_work = concat_.total_work()   
# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
params = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 'color_reg':['black']*20, 
          'color_symbols': ['slategray']*20, 'arr_size': 10, 'left_margin': 0.1}

DJS_all = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[5,3],
                      alpha=6, fig_size=[8,6], params=params, ext='png')
DJS_all.colors = Color
fig4 = DJS_all.plot_DJS(concat_.all_dfs_ankle, 
                    cols=np.r_[10,1,6,11,2,7,12,0,5,13,3,8,14,4,9], rows= np.r_[0,1],
                    title="Individual ankle DJS C vs Y v A", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)

# =============================================================================
# Best params
# =============================================================================

# df_turn_ = hyperparams(all_dfs.loc[:,idx['$0.8 < v/h < 1$ A',:]], 
#                         smooth_radius=(2,8), c_radius=(6,12), R2=True) #
