#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 8th 2020
Script for executing the DJS functions
@author: nikorose
"""
from DJSFunctions import *
import os
from pathlib import PurePath
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors



Schwartz_ = ankle_DJS('Schwartz.xls', 
                      dir_loc = 'Schwartz',
                      exp_name = 'Schwartz analysis' )

all_dfs = Schwartz_.extract_DJS_data()
df_turn = Schwartz_.get_turning_points(turning_points= 6, 
                           smoothing_radius = 2, cluster_radius= 8)
# Schwartz_.deg_to_rad()
Schwartz_.energy_calculation()
#Sensitive results may vary when integrating degrees, the best is to do in radians
Schwartz_.deg_to_rad()
total_work = Schwartz_.total_work()
# =============================================================================
# Plotting dynamic parameters
# =============================================================================
plot_dyn = plot_dynamic(SD=True, save=True, plt_style='bmh', alpha=1.5,
                        fig_size=[8,6])


fig2 = plot_dyn.gait_plot(Schwartz_.all_dfs_ankle, 
                    cols = None,
                    rows = None,
                    title='Ankle dynamics features for youth')



# =============================================================================
# Obtaining the mechanical work through power instances in regular walking 
# =============================================================================
idx= pd.IndexSlice
work_df = Schwartz_.power_energy.loc[idx[: , 'mean'], :]
zero_ro = Schwartz_.energy_fun.min_max_power(Schwartz_.power_ankle)

# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
# for i, name in enumerate(df_turn.index):
#     #Changing the tuning for the turning points in the last two
#     if i > 2:
#         df_turn = Schwartz_.get_turning_points(turning_points= 6, 
#                            smoothing_radius = 5, cluster_radius= 7)
#     DJS = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=True)
#     DJS.colors =  plt.rcParams['axes.prop_cycle'].by_key()['color'][i:]
#     fig4 = DJS.plot_DJS(Schwartz_.all_dfs_ankle, 
#                         cols=np.r_[i], rows= np.r_[0,2],
#                         title="Ankle DJS in children at {} speed".format(name), 
#                         legend=True, reg=df_turn,
#                         integration= True, rad = True)
Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*2
DJS_schwartz = plot_ankle_DJS(SD=True, save=True, plt_style='seaborn-whitegrid', sep=[1,3],
                          alpha=1.5, fig_size=[2.25, 7.0])
DJS_schwartz.colors = Color
fig3 = DJS_schwartz.plot_DJS(Schwartz_.all_dfs_ankle, 
                    cols=np.r_[:3], rows= np.r_[0,2],
                    title="Ankle DJS for children at different gait speed 1", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)

DJS_schwartz = plot_ankle_DJS(SD=True, save=True, plt_style='seaborn-whitegrid', sep=[1,2],
                          alpha=1.5, fig_size=[2.25, 4.67])
DJS_schwartz.colors = Color
fig4 = DJS_schwartz.plot_DJS(Schwartz_.all_dfs_ankle, 
                    cols=np.r_[3:5], rows= np.r_[0,2],
                    title="Ankle DJS for children at different gait speed 2", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)