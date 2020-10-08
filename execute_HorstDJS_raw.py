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

Horst_ = extract_preprocess_data('/home/nikorose/enprietop@unal.edu.co/'+
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
plot_dyn = plot_dynamic(SD=False, save=True, plt_style='bmh')

# Plotting power information

fig2 = plot_dyn.gait_plot(Horst_.all_dfs_ankle, 
                    cols = np.r_[:15],
                    rows = None,
                    title='Ankle dynamics of Sub 1 in Horst data')

# =============================================================================
# Obtaining the mechanical work through power instances in regular walking 
# =============================================================================
idx= pd.IndexSlice
work_df = Horst_.power_energy.loc[idx[: , 'mean'], :]
zero_ro = Horst_.energy_fun.min_max_power(Horst_.power_ankle)

# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
DJS = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=False)

fig4 = DJS.plot_DJS(Horst_.all_dfs_ankle, 
                    cols=None, rows= np.r_[0,2],
                    title="Ankle DJS subject 1", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)

# DJS.reg_info_df.to_csv('Horst/Horst_reg_lines_raw.csv')
# df_turn.to_csv('Horst/Horst_tp_raw.csv')
