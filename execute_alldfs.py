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
concat_df_turn = pd.concat([df_turn_ferra.apply(lambda x: x/2).astype(np.int64), 
                            df_turn_schwartz], axis=0)

concat_gait = concat_gait.dropna(axis=0)
concat_gait = concat_gait.reindex(Schwartz_.index_ankle.get_level_values(0).unique(), 
                                  level=0, axis=0)
# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
params = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 'color_reg':['black']*20, 
          'color_symbols': ['slategray']*20, 'arr_size': 10, 'left_margin': 0.1}

DJS_all = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[5,3],
                      alpha=6, fig_size=[8,6], params=params, ext='png')
DJS_all.colors = Color
fig4 = DJS_all.plot_DJS(concat_gait, 
                    cols=np.r_[10,1,6,11,2,7,12,0,5,13,3,8,14,4,9], rows= np.r_[0,2],
                    title="Individual ankle DJS C vs Y v A", 
                    legend=True, reg=concat_df_turn,
                    integration= True, rad = True)
