#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:25:22 2020
Script for executing the DJS functions
@author: nikorose
"""
from DJSFunctions import plot_ankle_DJS, ankle_DJS
import os
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from utilities_QS import ttest




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
                           param_1 = 4, cluster_radius= 17)

#Tuning according to best hyperparams
df_turn.iloc[3,:] = np.array([3,10,21,32,42])*2 # -> sr_2,cr_6 Y medium
df_turn.iloc[12,:] = np.array([2,11,19,31,42])*2 # -> sr_2,cr_7 A medium 
df_turn.iloc[4,:] = np.array([3,11,19,31,43])*2 # -> sr_2,cr_7 Y V fast
df_turn.iloc[13,:] = np.array([3,14,22,30,41])*2 # -> sr_2,cr_7 A Fast
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

Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
params = {'sharex':False, 'sharey':True, 'left_margin': 0.2, 'arr_size':12,
          'hide_labels':(False, True), 'yticks': np.arange(-0.25, 2.25, 0.25), 
          'xticks':None}

# =============================================================================
# Plotting QS one by one comparing adults and youths
# =============================================================================

cols_to_joint ={r'$0.363 < v* < 0.500$': (0,9, True),
                r'$v/h < 0.6$': (1,10, False),
                r'$0.6 < v/h < 0.8$': (2,11, True),
                r'$0.8 < v/h < 1$': (3,12, True),
                r'$v/h > 1.0$': (4,13, True),
                'Toes': (5,14, False),
                'Heels': (6,15, True),
                'Ascending': (7,16, True),
                'Descending': (8,17, True)}

for num, key in enumerate(cols_to_joint.keys()):
    params.update({'hide_labels': (False, cols_to_joint[key][-1])})
    DJS_all = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                              alpha=1.5, fig_size=[3.0,2.5], params=params)
    fig6 = DJS_all.plot_DJS(Ferrarin_.all_dfs_ankle, 
                        cols=list(cols_to_joint[key][:2]), rows= np.r_[0,2],
                        title="Ankle DJS age group comparison {}".format(num), 
                        legend=True, reg=df_turn,
                        integration= True, rad = True, header= key)
    if num == 0:
        reg_info = pd.DataFrame(DJS_all.reg_info_df)
        work_ = pd.DataFrame(DJS_all.areas)
    else:
        reg_info = pd.concat([reg_info, DJS_all.reg_info_df])
        work_ = pd.concat([work_, DJS_all.areas])
reg_info = reg_info.round(3)
work_ = work_.round(3)


# =============================================================================
# Obtaining the ttest of youth against adults (Ferrarin) at the same speed
# =============================================================================
n_ferra_y = [34, 76, 111, 71, 100, 83, 51, 75, 67] #XS S Natural M L T H A D
n_ferra_a = [140, 110, 124, 68, 52, 124, 85, 73, 72] #XS S Natural M L T H A D

cols_ = df_turn.index
cols_ = cols_[np.r_[1,2,0,3:9,10,11,9,12:18]]
etiquete = ['VS','S','C','F','VF','T','H', 'A', 'D']
#Dropping GRF and powers, we are interested in QS only
df_QS = Ferrarin_.all_dfs_ankle
df_QS = df_QS.drop(['Vertical  Force [%BH]', 'Ankle  [W/kg]'], axis=0, level=0)

tt_age = pd.concat([ttest(df_QS[cols_[i]],
                         df_QS[cols_[i+9]], 
                         samples=[n_ferra_y[i], n_ferra_a[i]], 
                         name='Ttest_{}'.format(etiquete[i]), method='scipy') for i in range(9)], axis=1)

tt_angles_age = tt_age.loc['Ankle Dorsi/Plantarflexion  Deg [°]'].mean(axis=0)
tt_moments_age = tt_age.loc['Ankle Dorsi/Plantarflexion  [Nm/kg]'].mean(axis=0)

#To sort the p values
tt_pvalues_age = pd.concat([tt_angles_age.loc[idx[:,'p_value']],
                            tt_moments_age.loc[idx[:,'p_value']]], axis=1)
tt_pvalues_age.columns = ['ang P values', 'mom P values']


# Conclusion: There is no significant differences between age groups at their own speed range or mode

# =============================================================================
# Obtaining the ttest comparing speeds and see where the statistical difference starts from 
# =============================================================================

tt_speed = pd.concat([ttest(df_QS[cols_[i]],
                         df_QS[cols_[j]], 
                         samples=[n_ferra_a[num_i], n_ferra_a[num_j]], 
                         name='{}_Ttest_{}'.format(etiquete[i-9], etiquete[j-9]), #substract 9 for Y
                         method='scipy') for num_i, i in enumerate(range(9,14)) for num_j, j in enumerate(range(9,14)) if i != j], axis=1) #replace with range(5) for Y

tt_angles_sp = tt_speed.loc['Ankle Dorsi/Plantarflexion  Deg [°]'].mean(axis=0)
tt_moments_sp = tt_speed.loc['Ankle Dorsi/Plantarflexion  [Nm/kg]'].mean(axis=0)

# conclusion: There is a statistical diff between the averages of the two groups, if and only if 
# we set an alpha = 0.1, in VS and VF with 0.07 in youth
# With respect to adults there is a statistical difference between VS ans VF (0.0505) and VF and S (0.097)
# If we set an alpha = 0.1 VF is statistically different w.r.t other gait speeds in angles.
# 

#To sort the p values
tt_comp_speed = pd.concat([tt_angles_sp.loc[idx[:,'p_value']],
                           tt_moments_sp.loc[idx[:,'p_value']]], axis=1)
tt_comp_speed.columns = ['ang P value','mom P value']

 