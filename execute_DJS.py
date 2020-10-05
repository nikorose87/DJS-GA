#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:25:22 2020
Script for executing the DJS functions
@author: nikorose
"""
from DJSFunctions import *
import os
from pathlib import PurePath
import pandas as pd
import numpy as np



Schwartz_ = extract_preprocess_data('Schwartz.xls', 
                                    dir_loc='Schwartz')
Schwartz_.complete_data()

#Excluding not regular intentions
# exclude_list = ["{}_{}".format(i,j) for i in ['Toe', 'Heel', 'Descending', 
#                                             'Ascending'] for j in ['a','y']]
#Excluding some regular intentions
exclude_list = ['{}{}'.format(i,j) for i in ['S','M','L','Xs'] \
                      for j in ['a','y']]

Ferrarin_ = ankle_DJS('mmc3.xls', 
                      dir_loc = 'Ferrarin',
                      exp_name = 'Ferrarin analysis' )

all_dfs = Ferrarin_.extract_DJS_data()
df_turn = Ferrarin_.get_turning_points(turning_points= 6, 
                           smoothing_radius = 4, cluster_radius= 15)
# Ferrarin_.deg_to_rad()
Ferrarin_.energy_calculation()
#Sensitive results may vary when integrating degrees, the best is to do in radians
Ferrarin_.deg_to_rad()
total_work = Ferrarin_.total_work()
# =============================================================================
# Plotting dynamic parameters
# =============================================================================
plot_dyn = plot_dynamic(SD=True, save=True, plt_style='bmh')

# # Plotting power information
plot_dyn.gait_plot(Ferrarin_.power, 
                    cols =np.r_[:5, 9:14], 
                    title='Power dynamics')

fig1 = plot_dyn.gait_plot(Ferrarin_.angles, 
                    cols = np.r_[:5, 9:14], 
                    rows = None,
                    title='Angle dynamics')

fig2 = plot_dyn.gait_plot(Ferrarin_.all_dfs_ankle, 
                    cols = np.r_[:5],
                    rows = None,
                    title='Ankle dynamics features for youth')

fig3 = plot_dyn.gait_plot(Ferrarin_.all_dfs_ankle, 
                    cols = np.r_[5:10], 
                    rows = [2],
                    title='Ankle dynamics features for adults')

# =============================================================================
# Plotting some specific features separately
# =============================================================================
# fig4, ax = plt.subplots()
# ax.remove()
# fig4.axes.append(fig2)
# fig2.get_axes()[0].figure = fig4
# fig4.axes.append(fig2.get_axes()[0])
# fig4.add_axes(fig2.get_axes()[0])

# dummy = fig4.add_subplot(111)
# fig2.get_axes()[0].set_position(dummy.get_position())
# dummy.remove()

# =============================================================================
# Obtaining the mechanical work through power instances in regular walking 
# =============================================================================
idx= pd.IndexSlice
work_df = Ferrarin_.power_energy.loc[idx[: , 'mean'], :]
zero_ro = Ferrarin_.energy_fun.min_max_power(Ferrarin_.power_ankle)

# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
DJS = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False)
fig4 = DJS.plot_DJS(Ferrarin_.all_dfs_ankle, 
                    cols=np.r_[:5], rows= np.r_[0,2],
                    title="Ankle Dynamic Joint Stiffness at irregular gait intentions", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)


