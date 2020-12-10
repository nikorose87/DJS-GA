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
import matplotlib.colors as mcolors




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
                           smoothing_radius = 4, cluster_radius= 17)
# Ferrarin_.deg_to_rad()
Ferrarin_.energy_calculation()
#Sensitive results may vary when integrating degrees, the best is to do in radians
Ferrarin_.deg_to_rad()
total_work = Ferrarin_.total_work()
# =============================================================================
# Plotting dynamic parameters
# =============================================================================
# plot_dyn = plot_dynamic(SD=True, save=True, plt_style='bmh')

# # Plotting power information
# plot_dyn.gait_plot(Ferrarin_.power, 
#                     cols =np.r_[:5, 9:14], 
#                     title='Power dynamics')

# fig1 = plot_dyn.gait_plot(Ferrarin_.angles, 
#                     cols = np.r_[:5, 9:14], 
#                     rows = None,
#                     title='Angle dynamics')

# fig2 = plot_dyn.gait_plot(Ferrarin_.all_dfs_ankle, 
#                     cols = np.r_[:5],
#                     rows = None,
#                     title='Ankle dynamics features for youth')

# fig3 = plot_dyn.gait_plot(Ferrarin_.all_dfs_ankle, 
#                     cols = np.r_[9:14], 
#                     rows = None,
#                     title='Ankle dynamics features for adults')

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
Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*2

# DJS_all = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=[5,2],
#                       alpha=7.0, fig_size=[10,6])
# DJS_all.colors = Color
# fig4 = DJS_all.plot_DJS(Ferrarin_.all_dfs_ankle, 
#                     cols=np.r_[1, 10, 2, 11, 0, 9, 3, 12, 4, 13], rows= np.r_[0,2],
#                     title="Individual ankle DJS for regular speed comparing Youth and Adults", 
#                     legend=True, reg=df_turn,
#                     integration= True, rad = True)

    
# =============================================================================
# Plotting QS one by one comparing youths and adults 
# =============================================================================
# cols_to_joint ={'Very Slow': (1, 10), 'Slow':(2, 11), 'Free': (0, 9), 
#                 'Medium': (3, 12), 'Fast': (4, 13), 'Toes':(5,14), 'Heels':(6,15),
#                 'Ascending': (7,16), 'Descending': (8,17)}

# for key in cols_to_joint.keys():
#     # Changing the tuning for the turning points in the last two
#     # if i > 2:
#     #     df_turn = Ferrarin_.get_turning_points(turning_points= 6, 
#     #                         smoothing_radius = 5, cluster_radius= 7)
#     DJS_comp = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
#                               alpha=1.5, fig_size=[2.5, 2.2])
#     fig5 = DJS_comp.plot_DJS(Ferrarin_.all_dfs_ankle, 
#                         cols=list(cols_to_joint[key]), rows= np.r_[0,2],
#                         title="Ankle DJS comparison at {} speed".format(key), 
#                         legend=True, reg=df_turn,
#                         integration= True, rad = True)


# =============================================================================
# trying to do the best fit as possible for Ferrarin
# =============================================================================
DJS_all = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=False,
                      alpha=7.0)
DJS_all.colors = Color
fig4 = DJS_all.plot_DJS(Ferrarin_.all_dfs_ankle, 
                    cols=np.r_[0,1,2,3,4], rows= np.r_[0,2],
                    title="Individual ankle DJS for regular speed comparing Youth and Adults", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)