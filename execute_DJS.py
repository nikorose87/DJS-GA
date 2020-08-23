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
                                    dir_loc='../DJS-Scripts/Schwartz')
Schwartz_.complete_data()

#Excluding not regular intentions
# exclude_list = ["{}_{}".format(i,j) for i in ['Toe', 'Heel', 'Descending', 
#                                             'Ascending'] for j in ['a','y']]
#Excluding some regular intentions
exclude_list = ['{}{}'.format(i,j) for i in ['S','M','L','Xs'] \
                      for j in ['a','y']]

Ferrarin_ = ankle_DJS('mmc3.xls', 
                      dir_loc = '../DJS-Scripts/Ferrarin',
                      exp_name = 'Ferrarin analysis',
                      exclude_names=exclude_list)

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
                    cols=np.r_[0,5], rows= np.r_[0,2],
                    title="Ankle Dynamic Joint Stiffness at irregular gait intentions", 
                    legend=True, reg=df_turn,
                    integration= True, rad = False)


# =============================================================================
# Amputee information reading from the STO
# =============================================================================
root_dir = PurePath(os.getcwd())
amp_dir = os.path.join(root_dir, 
        "TRANSTIBIAL/Transtibial  Izquierda/Gerson Tafud/Opensim files")
S1_dir = os.path.join(root_dir, 
                      'C3D/IK_and_ID_S01_0002_Gait')
# Finding where the dataset begins in the sto or mot file
def endheader_line(file):
    with open(file,'r') as f:
        lines = f.read().split("\n")
    
    word = 'endheader' # dummy word. you take it from input
    
    # iterate over lines, and print out line numbers which contain
    # the word of interest.
    for i,line in enumerate(lines):
        if word in line: # or word in line.split() to search for full words
            #print("Word \"{}\" found in line {}".format(word, i+1))
            break_line = i+1
    return break_line


def dyn_tables(IK, ID, _dir):
    break_line = endheader_line(os.path.join(_dir, IK))
    IK_file = pd.read_table(os.path.join(_dir, IK),
                          skiprows=break_line, delim_whitespace=True, 
                          engine='python', decimal='.', index_col=0)
    
    
    break_line = endheader_line(os.path.join(_dir, ID))
    ID_file = pd.read_table(os.path.join(_dir, ID),
                          skiprows=break_line, delim_whitespace=True, 
                          engine='python', decimal='.', index_col=0)
    
    QS_df = pd.concat([IK_file['ankle_angle_r'],ID_file['ankle_angle_r_moment']], axis=1)
    return IK_file, ID_file, QS_df

IK_GT, ID_GT, QS_GT = dyn_tables('0143~ab~Walking_01_IK.mot', 
                                     'inverse_dynamics.sto', amp_dir)

IK_S1, ID_S1, QS_S1 = dyn_tables('IK_subject001.sto', 
                                     'inverse_dynamics.sto', S1_dir)
