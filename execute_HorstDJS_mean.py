#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:04:42 2020
Script to analyse only the Horst mean data set
@author: nikorose
"""

from DJSFunctions import extract_preprocess_data, ankle_DJS
from plot_dynamics import plot_dynamic, plot_ankle_DJS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Horst_ = extract_preprocess_data('Horst_mean.csv', 
                                    dir_loc='Horst')
# Subject information 
anthro_info = pd.read_csv('Horst/Gait_subject_info.csv', sep=";",
                          decimal=',')
anthro_info.columns = ['ID','gender','age','mass','height']     
       
Horst_df = Horst_.all_dfs

idx= pd.IndexSlice

Horst_QS = Horst_df.loc[idx[['ankle_angle_r','ankle_angle_r_moment',
                             'ground_force_1_v_2'], :]]

#According to Winter https://ouhsc.edu/bserdac/dthompso/web/gait/epow/pow1.htm
#in order to calculate the power plot we should do the following:
#Obtain the derivative of the angle plot to obtain angular velocity
#make the product between the moment and the ang vel, then P=Mw

def multi_idx(name, df_, idx=True):
    
    if idx:
        index_mi = pd.MultiIndex.from_product([[name],df_.index])
        df_.index = index_mi
    else:
        index_mi = pd.MultiIndex.from_product([[name],df_.columns])
        df_.columns = index_mi
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


#Normalization method
Stansfield = False
Hof = True
#Normalizing according to Stansfield et al 2006
for num, sub in enumerate(anthro_info['ID']):
    if Stansfield:        
        #Moment
        vals_mom = Horst_QS.loc['ankle_angle_r_moment',sub] / (anthro_info['mass'][num] *9.81 * anthro_info['height'][num])
        Horst_QS.loc['ankle_angle_r_moment', sub] = vals_mom.values
        #Power
        vals_pow = Horst_QS.loc['Power [W]',sub] / (anthro_info['mass'][num] *9.81**(3/2) * anthro_info['height'][num]**0.5)
        Horst_QS.loc['Power [W]', sub] = vals_pow.values       

    elif Hof:
        #Moment
        vals_mom = Horst_QS.loc['ankle_angle_r_moment',sub] / (anthro_info['mass'][num])
        Horst_QS.loc['ankle_angle_r_moment', sub] = vals_mom.values
        #Power
        vals_pow = Horst_QS.loc['Power [W]',sub] / (anthro_info['mass'][num])
        Horst_QS.loc['Power [W]', sub] = vals_pow.values       

    vals_GRF = Horst_QS.loc['ground_force_1_v_2',sub] / (anthro_info['mass'][num] *9.81)
    Horst_QS.loc['ground_force_1_v_2', sub] = vals_GRF.values


#Removing angular vel
Horst_QS = Horst_QS.drop(['Angular vel [deg / GC]'], axis =0)

#Replacing index names to be similar with Ferrarin
idx_names = ['Ankle Dorsi/Plantarflexion', 'Ankle Dorsi/Plantarflexion',
             'Vertical Force',  'Ankle'][::-1]

idx_old = list(Horst_QS.index.get_level_values(0).unique())


for num, name in enumerate(idx_names[::-1]):
    Horst_QS.index = Horst_QS.index.set_levels(\
                    Horst_QS.index.levels[0].str.replace(idx_old[num], name), level=0)
        
# =============================================================================
# Creating df for gender
# =============================================================================
cols_women = []
cols_men = []
for i in anthro_info.index:
    if anthro_info.iloc[i,1]:
        cols_women.append(anthro_info['ID'][i])
    else:
        cols_men.append(anthro_info['ID'][i])   

#Obtaining the mean on each        
Horst_QS_men = Horst_QS.loc[:,cols_men].mean(axis=1, level=1)
Horst_QS_women = Horst_QS.loc[:,cols_women].mean(axis=1, level=1)

#Setiing multiidx name
Horst_QS_men = multi_idx('Men', Horst_QS_men, idx=False)
Horst_QS_women = multi_idx('Women', Horst_QS_women, idx= False)

Horst_QS_gen = pd.concat([Horst_QS_men, Horst_QS_women], axis=1)

# =============================================================================
# Setting variables and plotting in individuals
# =============================================================================


Horst_ = ankle_DJS(Horst_QS, exp_name = 'Horst individuals analysis')

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
plot_dyn = plot_dynamic(SD=True, save=True, plt_style='bmh')

# # Plotting power information

fig1 = plot_dyn.gait_plot(Horst_.all_dfs_ankle, 
                    cols = np.r_[:8],
                    rows = None,
                    title='Ankle dynamics of Horst Experiment subjects')

# =============================================================================
# Obtaining the mechanical work through power instances in regular walking 
# =============================================================================
idx= pd.IndexSlice
work_df = Horst_.power_energy.loc[idx[: , 'mean'], :]
zero_ro = Horst_.energy_fun.min_max_power(Horst_.power_ankle)

# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
DJS = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=True)
fig2 = DJS.plot_DJS(Horst_.all_dfs_ankle, 
                    cols=np.r_[:8], rows= np.r_[0,2],
                    title="Ankle DJS of Horst subjects", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)

# =============================================================================
# Setting variables and plotting in gender
# =============================================================================


Horst_gen = ankle_DJS(Horst_QS_gen, exp_name = 'Horst gender analysis')

all_dfs_gen = Horst_gen.extract_df_DJS_data(idx=[0,2,1,3])
df_turn_gen = Horst_gen.get_turning_points(turning_points= 6, 
                            smoothing_radius = 4, cluster_radius= 15)
# Horst_.deg_to_rad()
Horst_gen.energy_calculation()
#Sensitive results may vary when integrating degrees, the best is to do in radians
Horst_gen.deg_to_rad()
total_work_gen = Horst_gen.total_work()    

# =============================================================================
# Plotting dynamic parameters
# =============================================================================
plot_dyn_gen = plot_dynamic(SD=True, save=True, plt_style='bmh')

# # Plotting power information

fig3 = plot_dyn_gen.gait_plot(Horst_gen.all_dfs_ankle, 
                    cols = None,
                    rows = None,
                    title='Ankle dynamics for gender in Horst data')

# =============================================================================
# Obtaining the mechanical work through power instances in regular walking 
# =============================================================================
idx= pd.IndexSlice
work_df_gen = Horst_gen.power_energy.loc[idx[: , 'mean'], :]
zero_ro_gen = Horst_gen.energy_fun.min_max_power(Horst_gen.power_ankle)

# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
DJS_gen = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=True)
fig4 = DJS_gen.plot_DJS(Horst_gen.all_dfs_ankle, 
                    cols=None, rows= np.r_[0,2],
                    title="Ankle DJS for gender in Horst data", 
                    legend=True, reg=df_turn,
                    integration= True, rad = True)
