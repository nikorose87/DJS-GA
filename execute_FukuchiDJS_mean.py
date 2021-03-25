#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:47:55 2020
Analysing the DJS of Fukuchi's paper 
@author: nikorose
"""


from DJSFunctions import extract_preprocess_data, ankle_DJS
from plot_dynamics import plot_ankle_DJS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from utilities_QS import multi_idx, create_df, best_hyper, change_labels

#stats
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import seaborn as sns
from scipy.stats.mstats import kruskal
import scikit_posthocs as sp

# =============================================================================
# Settings
# =============================================================================
instances = False
Age_group = False
Age_group_mod = False
Age_group_gen = False
Gender = False
individuals = True
statistics = True

sns.set_context('paper', font_scale=1.5)
sns.set_style("whitegrid")


# =============================================================================
# Helper functions
# =============================================================================

def power(col):
    return col['RAnkleMomentZ']*col['Angular vel [deg / GC]']

# =============================================================================
# Charging the folders 
# =============================================================================
root_dir = os.getcwd()
info_dir = '/home/nikorose/enprietop@unal.edu.co/Tesis de Doctorado/Gait Analysis Data/' \
            + 'Downloaded/Fukuchi/TreadmillAndOverground/Dataset'

Fukuchi_df = pd.read_csv('Fukuchi/Fukuchi_mean.csv', header=[0,1], index_col=[0,1])

#Take care when filtering specific columns
idx = pd.IndexSlice

Fukuchi_df = Fukuchi_df.loc[idx[['RAnkleAngleZ', 'RAnkleMomentZ', 'RGRFY'],:],:]

# =============================================================================
# Adding power column by calculating through moment and ang vel dot product
# =============================================================================


#According to Winter https://ouhsc.edu/bserdac/dthompso/web/gait/epow/pow1.htm
#in order to calculate the power plot we should do the following:
#Obtain the derivative of the angle plot to obtain angular velocity
#make the product between the moment and the ang vel, then P=Mw


ang_vel = Fukuchi_df.loc['RAnkleAngleZ',:].apply(lambda x: np.gradient(x), axis=0)
ang_vel = multi_idx('Angular vel [deg / GC]', ang_vel)
Fukuchi_df = pd.concat([Fukuchi_df, ang_vel], axis=0)


power_df = Fukuchi_df.apply(power, axis=0)
power_df = multi_idx('Ankle Power [W]', power_df)
Fukuchi_df = pd.concat([Fukuchi_df, power_df], axis=0)


#Removing angular vel
Fukuchi_df = Fukuchi_df.drop(['Angular vel [deg / GC]'], axis =0)
#Replacing index names to be similar with Ferrarin
idx_names = ['Ankle Dorsi/Plantarflexion ', 'Ankle Dorsi/Plantarflexion',
             'Vertical Force',  'Ankle'][::-1]
idx_old = list(Fukuchi_df.index.get_level_values(0).unique())

for num, name in enumerate(idx_names[::-1]):
    Fukuchi_df.index = Fukuchi_df.index.set_levels(\
                    Fukuchi_df.index.levels[0].str.replace(idx_old[num], name), level=0)
        
# =============================================================================
# Performing the Shapiro Wilk test per category in order to see normal distributions
# =============================================================================
#Removing nan due to shapiro is sensitive to it.
Fukuchi_df_nan = Fukuchi_df.dropna(axis=1, how='all')

#In order to fill the few nan spaces
Fukuchi_df_nan = Fukuchi_df_nan.interpolate(axis=1)

shapiro = {}

for col in Fukuchi_df.columns.get_level_values(1).unique():
    shapiro_t = pd.DataFrame(Fukuchi_df_nan.loc[:, idx[:,col]].apply(stats.shapiro, axis=1), 
                             columns= ['res'])
    shapiro_t['stats'] = shapiro_t.apply(lambda x: x.res[0], axis=1)
    shapiro_t['p value'] = shapiro_t.apply(lambda x: x.res[1], axis=1)
    shapiro_t = shapiro_t.drop(['res'], axis=1)
    shapiro.update({'{}'.format(col):shapiro_t})

resume_shapiro = pd.concat([item.mean(level=0) for item in shapiro.values()], axis=1)


        

# =============================================================================
# Plotting dynamic parameters
# =============================================================================
# plot_dyn_gen = plot_dynamic(SD=True, save=True, plt_style='bmh')

# # # Plotting power information

# fig1 = plot_dyn_gen.gait_plot(Fukuchi_df, 
#                     cols = np.r_[0:1],
#                     rows = np.r_[0:3],
#                     title='Ankle Dynamics Right Foot Fukuchi data at instances')  
# =============================================================================
# Anthropometric information

meta_info = pd.read_excel('Fukuchi/WBDSinfo.xlsx')
meta_info = meta_info.drop(meta_info.iloc[:, [0,-1]].columns, axis=1)

anthro_info = meta_info.iloc[:,np.r_[2:6]]
anthro_info = anthro_info[['Mass','Age','Gender', 'Height']]
anthro_info.columns = ['mass','age','gender', 'height']

# =============================================================================
# we are going to analyze only angles and kinetic information 
# =============================================================================

processed_dir = info_dir + '/WBDSascii'

# Index where the dynamic information is
index_angtxt = [j for j, i in enumerate(meta_info['FileName']) if 'ang.txt' in i]
index_knttxt = [j for j, i in enumerate(meta_info['FileName']) if 'knt.txt' in i]

#Index labels
labels_ang = meta_info['FileName'][index_angtxt]
labels_knt = meta_info['FileName'][index_knttxt]

meta_info_red = meta_info.filter(index_angtxt, axis=0)
meta_info_red['Subject'] = Fukuchi_df.columns.get_level_values(0)
meta_info_red['Mode'] = Fukuchi_df.columns.get_level_values(1)


# =============================================================================
# Obtaining the gait speed on each
# ============================================================================

#replacing with Nan
meta_info_valid = meta_info.replace('--', np.nan)
#dropping non nan values
meta_info_valid =  meta_info_valid.dropna(axis=0, subset= ['GaitSpeed(m/s)'])
#Dropping non useful columns
meta_info_valid = meta_info_valid.dropna(axis=1)
# Extracting only the subject
meta_info_valid['Subject'] = meta_info_valid['FileName'].apply(lambda x: x[3:6])
#Index ending in .c3d
index_c3d = [j for j, i in zip(meta_info_valid.index, meta_info_valid['FileName']) if '.c3d' in i]
#Erasing all non c3d indexes
meta_info_valid = meta_info_valid.loc[index_c3d]
# Extracting only the trial
meta_info_valid['Trial'] = meta_info_valid['FileName'].apply(lambda x: x[10:-4])
#Mode column
meta_info_valid['Mode'] = meta_info_valid['Trial'].apply(lambda x: x[0])
meta_info_valid['Mode'] = meta_info_valid['Mode'].replace(['T', 'O'], ['Treadmill', 'Overground'])
#Type column
meta_info_valid['Type'] = meta_info_valid['Trial'].apply(lambda x: x[-1])
#Obtaining fraude number
meta_info_valid['Fraude'] = meta_info_valid['GaitSpeed(m/s)']/(np.sqrt(9.81*meta_info_valid['LegLength']))
#Obtaining Stansfiels number
meta_info_valid['Stansfield'] = 100*meta_info_valid['GaitSpeed(m/s)']/meta_info_valid['Height']

#Fraude vs Stansfield proportion Females: 0.5617, Male:0.5856, overall: 0.5747

# =============================================================================
# Building filters by speed 
# =============================================================================

meta_very_slow = meta_info_valid.query("Mode == 'Overground' & Fraude < 0.227")
meta_slow = meta_info_valid.query("Mode == 'Overground' & Fraude >= 0.227 & Fraude < 0.363")
meta_free = meta_info_valid.query("Mode == 'Overground' & Fraude >= 0.363 & Fraude < 0.5")
meta_fast = meta_info_valid.query("Mode == 'Overground' & Fraude >= 0.5 & Fraude < 0.636")
meta_very_fast = meta_info_valid.query("Mode == 'Overground' & Fraude > 0.636")

#Unique trials
unique_type = meta_info_valid['Type'].unique()
mean_T = {i:[np.round(meta_info_valid.query("Type == '{}'".format(i))['Fraude'].mean(),2)] for i in unique_type}
std_T = {i:np.round(meta_info_valid.query("Type == '{}'".format(i))['Fraude'].std(),2) for i in unique_type}
for key in mean_T.keys():
    mean_T[key].append(std_T[key]) #Including std in mean


new_dict_labels= ['T01','T02','T03','T04','T05','T06','T07','T08','OS','OC','OF']
vel_labels = [r'$v*={}({})$'.format(i,j) for i,j in mean_T.values()]
for num, old_key in enumerate(mean_T.keys()):
    mean_T[new_dict_labels[num]] = mean_T.pop(old_key)

#As is the same and generate some errors
vel_labels[-3] = '$v*=0.30(0.04)$'
#Reordering Fukuchi_df_nan
Fukuchi_df_nan = Fukuchi_df_nan.reindex(new_dict_labels, level=1, axis=1)
#Params for all comparatives plots
color_labels = ['blue','red','green','violet','orange','grey','goldenrod']
color_regs = ['dark'+i for i in color_labels]

Colors_tab = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
Color_DJS = [mcolors.CSS4_COLORS[item] for item in color_labels]*3
Color_reg = [mcolors.CSS4_COLORS[item] for item in color_regs]*3
params = {'sharex':False, 'sharey':False, 'left_margin': 0.2, 'arr_size':12,
          'yticks': np.arange(-0.25, 1.80, 0.25), 'xticks':None, 
          'color_reg':Color_DJS, 'color_symbols': Color_reg, #'color_DJS': Colors_tab,
          'alpha_prod': 0.3, 'alpha_absorb': 0.0, 'DJS_linewidth': 1.5,
          'sd_linewidth': 0.08,'reg_linewidth': 1.0}

times=3
smooth_ = [2,3,4]
cluster_ = range(15*times, 20*times, times)
if instances:
    
    # =============================================================================
    #     Let us keep only the piece of df and change labels
    # =============================================================================
    Fukuchi_df_modes = Fukuchi_df_nan.loc[:,idx[:,['T03','OS','T05','OC','T07','OF']]]
    # =============================================================================
    # Performing overall average over modes
    # =============================================================================
    
    Fukuchi_mean_modes = Fukuchi_df_modes.groupby(level=1, axis=1).mean()
    Fukuchi_sd_modes = Fukuchi_df_modes.groupby(level=1, axis=1).std()
    
    Fukuchi_modes = create_df(Fukuchi_mean_modes, Fukuchi_sd_modes)
    
    
    # =============================================================================
    # Setting variables and plotting in individuals
    # =============================================================================
    
    
    Fukuchi_instance = ankle_DJS(Fukuchi_modes, 
                        features= ['Ankle Dorsi/Plantarflexion ', 
                                   'Ankle Dorsi/Plantarflexion',
                                   'Vertical Force',
                                   'Ankle'], 
                        exp_name = 'Fukuchi Instances variation analysis')
    
    all_dfs_instance = Fukuchi_instance.extract_df_DJS_data(idx=[0,2,1,3])
    all_dfs_instance = Fukuchi_instance.interpolate_ankledf(times=times, replace=True)
    all_dfs_instance = Fukuchi_instance.change_labels(["Free O", "Fast O", 'Slow O', 'Slow T', 
                                    'Free T', 'Fast T'])

    
    df_turn_instance = best_hyper(all_dfs_instance, save='Fukuchi/best_params_instance.csv',
                             smooth_radius=smooth_,
                             cluster_radius=cluster_, verbose=False,
                             rows=[0,1])
    
    Fukuchi_instance.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_instance.deg_to_rad()
    total_work_instance = Fukuchi_instance.total_work()  

    # =============================================================================
    # Obtaining the mechanical work through power instances in regular walking 
    # =============================================================================
   
    cols_to_joint ={r'$v* = 0.3 \pm 0.04$': (2,3, False), 
                    r'$v* = 0.43 \pm 0.05$': (0,4, True),
                    r'$v* = 0.55 \pm 0.06$': (1,5, True)}
    
    for num, key in enumerate(cols_to_joint.keys()):
        params.update({'hide_labels': (False, cols_to_joint[key][-1])})
        DJS_instances = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=1.5, fig_size=[3.0,2.5], params=params)
        fig5 = DJS_instances.plot_DJS(all_dfs_instance, 
                            cols=list(cols_to_joint[key][:2]), rows= np.r_[0,2],
                            title="Ankle DJS OvsT group comparison at {}".format(key), 
                            legend=True, reg=df_turn_instance.loc[idx[:,'mean'],:],
                            integration= True, rad = True, header= key)
        if num == 0:
            reg_info_ins = pd.DataFrame(DJS_instances.reg_info_df)
            work_ins = pd.DataFrame(DJS_instances.areas)
        else:
            reg_info_ins = pd.concat([reg_info_ins, DJS_instances.reg_info_df])
            work_ins = pd.concat([work_ins, DJS_instances.areas])
    
    reg_info_ins = reg_info_ins.round(3)
    work_ins = work_ins.round(3)
    
    params.update({'hide_labels': (False, False)})
# =============================================================================
# Plotting dynamic parameters
# =============================================================================
# plot_dyn_gen = plot_dynamic(SD=True, save=True, plt_style='bmh')

# # # Plotting power information

# fig1 = plot_dyn_gen.gait_plot(Fukuchi_instance.all_dfs_ankle, 
#                     cols = None,
#                     rows = None,
#                     title='Ankle Dynamics Right Foot Fukuchi data at instances')  

velocities = {k: r'$v* = {} \pm {}$'.format(round(j[0],2),round(j[1],2)) for k, j in mean_T.items()}
velocities_text=['Very Slow','Slow', 'Free', 'Fast', 'Very Fast']

# =============================================================================
# Comparing DJS of adults and old people
# =============================================================================

# Which subjects are adults
adult_info = meta_info_red[meta_info_red['AgeGroup'] == 'Young']
adult_group = {i:'Adult' for i in adult_info['Subject']}

# Which subjects are old
old_info = meta_info_red[meta_info_red['AgeGroup'] == 'Older']
old_group = {i:'Elderly' for i in old_info['Subject']}

age_dict = dict(adult_group, **old_group)

# Creating the df for Age population 
Fukuchi_mean_age = Fukuchi_df.rename(columns= age_dict, level=0).mean(level=[0,1], axis=1)
#Merging column labels
Fukuchi_mean_age.columns = Fukuchi_mean_age.columns.map('{0[0]} {0[1]}'.format)
Fukuchi_sd_age = Fukuchi_df.rename(columns= age_dict, level=0).std(level=[0,1], axis=1)
#Merging column labels
Fukuchi_sd_age.columns = Fukuchi_sd_age.columns.map('{0[0]} {0[1]}'.format)

Fukuchi_age =  create_df(Fukuchi_mean_age, Fukuchi_sd_age)
Fukuchi_age = Fukuchi_age.interpolate(axis=1)

if Age_group:   
    
   
    # =============================================================================
    # Setting variables and plotting in individuals
    # =============================================================================
    
    
    Fukuchi_ages = ankle_DJS(Fukuchi_age, 
                        features= ['Ankle Dorsi/Plantarflexion ', 
                                  'Vertical Force',
                                   'Ankle Dorsi/Plantarflexion',
                                   'Ankle'], 
                        exp_name = 'Fukuchi Instances variation analysis')
    
    all_dfs_ages = Fukuchi_ages.extract_df_DJS_data(idx=[0,2,1,3])
    all_dfs_ages = Fukuchi_ages.interpolate_ankledf(times=times, replace=True)
    
    df_turn_ages = best_hyper(all_dfs_ages, save='Fukuchi/best_params_ages.csv',
                             smooth_radius=smooth_,
                             cluster_radius=cluster_, verbose=False)
    
    
    
    Fukuchi_ages.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_ages.deg_to_rad()
    total_work_ages = Fukuchi_ages.total_work() 
    
    # =============================================================================
    # Plotting QS one by one comparing adults and elderly people
    # =============================================================================
    cols_to_joint ={item: (num, num+11) for num, item in enumerate(meta_info_red['Mode'].unique())}
    
    for num, key in enumerate(cols_to_joint.keys()):
        if num == 2 or num == 3 or num == 7:
            params.update({'hide_labels':(False, False)})
        else:
            params.update({'hide_labels':(False, True)})
        DJS_comp = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=1.5, fig_size=[3.0,2.5], params=params)
        fig5 = DJS_comp.plot_DJS(Fukuchi_ages.all_dfs_ankle, 
                            cols=list(cols_to_joint[key]), rows= np.r_[0,2],
                            title="Ankle DJS age group comparison at {}".format(key), 
                            legend=True, reg=df_turn_ages, header=velocities[key],
                            integration= True, rad = True)
    
    params.update({'hide_labels': (False, False)})
    
if Age_group_mod:   
    # =============================================================================
    # Modifying Fukuchi ages
    # =============================================================================
    Fukuchi_age_mod = Fukuchi_age.drop(['Adult OC','Adult OS', 'Adult OF',
                                        'Elderly OC','Elderly OS', 'Elderly OF'], level=0, axis=1)
    
    order_labels = ['{} {}'.format(A, T) for A in ['Adult', 'Elderly'] for T in [1,3,6]]
    order_labels.extend(['{} {}'.format(A, T) for A in ['Adult', 'Elderly'] for T in [5,8]])
    
    #Groups average 01 and 02, 03 and 04, 06 and 07
    Fukuchi_vel = {'{} {}'.format(j,k): Fukuchi_age_mod.loc[:,idx[['{} T0{}'.format(j,k), 
                    '{} T0{}'.format(j,k+1)],:]].mean(axis=1, 
                level=1) for j,k in [(A,T) for A in ['Adult', 'Elderly'] for T in [1,3,6]]}
    
    Fukuchi_vel.update({'{} {}'.format(j,k): Fukuchi_age_mod.loc[:,idx['{} T0{}'.format(j,k),:]].mean(axis=1, 
                        level=1) for j,k in [(A,T) for A in ['Adult', 'Elderly'] for T in [5,8]]})
    
    labels_vel = pd.MultiIndex.from_product([['{} {}'.format(V,A) for  A in ['A', 'E'] for V in [r'$0.151 < v* < 0.265$', r'$0.263 < v* < 0.407$',
                                              r'$0.378 < v* < 0.478$', r'$0.434 < v* < 0.623$',
                                              r'$0.544 < v* < 0.0.689$']],['-1sd','mean','+1sd']])
    Fukuchi_vel_df = pd.concat([Fukuchi_vel[key] for key in sorted(order_labels)], axis=1)
    Fukuchi_vel_df.columns = labels_vel
    # =============================================================================
    # Setting variables and plotting in individuals
    # =============================================================================
    
    
    Fukuchi_ages_mod = ankle_DJS(Fukuchi_vel_df, 
                        features= ['Ankle Dorsi/Plantarflexion ', 
                                  'Vertical Force',
                                   'Ankle Dorsi/Plantarflexion',
                                   'Ankle'], 
                        exp_name = 'Fukuchi age mode variation analysis')
    
    
    
    all_dfs_ages_mod = Fukuchi_ages_mod.extract_df_DJS_data(idx=[0,2,1,3])
    all_dfs_ages_mod = Fukuchi_ages_mod.interpolate_ankledf(times=times, replace=True)
    df_turn_ages_mod = best_hyper(all_dfs_ages_mod, save='Fukuchi/best_params_ages_mod.csv',
                             smooth_radius=smooth_,
                             cluster_radius=cluster_, verbose=False)
    
    
    Fukuchi_ages_mod.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_ages_mod.deg_to_rad()
    total_work_ages = Fukuchi_ages_mod.total_work() 

    # =============================================================================
    # Plotting QS one by one comparing adults and elderly people
    # =============================================================================
    cols_to_joint = [(num, num+5) for num in range(5)]
    for num, key in enumerate(cols_to_joint):
        if num == 0:
            params.update({'hide_labels':(False, False)})
        else:
            params.update({'hide_labels':(False, True)})
        DJS_mod = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=1.5, fig_size=[3.0,2.5], params=params)
        fig5 = DJS_mod.plot_DJS(Fukuchi_ages_mod.all_dfs_ankle, 
                            cols=list(key), rows= np.r_[0,2],
                            title="Ankle DJS AvsE group comparison at {}".format(velocities_text[num]), 
                            legend=True, reg=df_turn_ages_mod, header=velocities_text[num],
                            integration= True, rad = True)
        if num == 0:
            reg_info_mode = pd.DataFrame(DJS_mod.reg_info_df)
            work_mode = pd.DataFrame(DJS_mod.areas)
        else:
            reg_info_mode = pd.concat([reg_info_mode, DJS_mod.reg_info_df])
            work_mode = pd.concat([work_mode, DJS_mod.areas])
            
    reg_info_mode = reg_info_mode.round(3)
    work_mode = work_mode.round(3)
    params.update({'hide_labels': (False, False)})
    
if Age_group_gen:   
    # =============================================================================
    # Modifying Fukuchi ages
    # =============================================================================
    Fukuchi_age_gen = Fukuchi_age.loc[:,idx[['Adult OC','Adult OS', 'Adult OF',
                                        'Elderly OC','Elderly OS', 'Elderly OF'],:]]
    # =============================================================================
    # Setting variables and plotting in individuals
    # =============================================================================
    
    
    Fukuchi_ages_gen = ankle_DJS(Fukuchi_age_gen, 
                        features= ['Ankle Dorsi/Plantarflexion ', 
                                  'Vertical Force',
                                   'Ankle Dorsi/Plantarflexion',
                                   'Ankle'], 
                        exp_name = 'Fukuchi Overground and gender variation analysis')
    
    all_dfs_ages_gen = Fukuchi_ages_gen.extract_df_DJS_data(idx=[0,2,1,3])
    all_dfs_ages_gen = Fukuchi_ages_gen.interpolate_ankledf(times=times, 
                                                            replace=True)
    df_turn_ages_gen = best_hyper(all_dfs_ages_gen, save='Fukuchi/best_params_ages_gen.csv',
                             smooth_radius=[False],
                             cluster_radius=range(32,35), verbose=False)
    
    
    Fukuchi_ages_gen.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_ages_gen.deg_to_rad()
    total_work_gen = Fukuchi_ages_gen.total_work() 
    
       
    # =============================================================================
    # Plotting QS one by one comparing adults and elderly people
    # =============================================================================
    cols_to_joint ={r'$v* = 0.3 \pm 0.04$': (2,5, False), r'$v* = 0.43 \pm 0.05$': (0,3, True),
                    r'$v* = 0.55 \pm 0.06$': (1,4, True)}
    
    for num, key in enumerate(cols_to_joint.keys()):
        params.update({'hide_labels': (False, cols_to_joint[key][-1])})
        DJS_age_gen = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=1.5, fig_size=[3.0,2.5], params=params)
        fig5 = DJS_age_gen.plot_DJS(Fukuchi_ages_gen.all_dfs_ankle, 
                            cols=list(cols_to_joint[key][:2]), rows= np.r_[0,2],
                            title="Ankle DJS age group comparison at {}".format(key), 
                            legend=True, reg=df_turn_ages_gen,
                            integration= True, rad = True, header= key)
        if num == 0:
            reg_info_gen = pd.DataFrame(DJS_age_gen.reg_info_df)
            work_gen = pd.DataFrame(DJS_age_gen.areas)
        else:
            reg_info_gen = pd.concat([reg_info_gen, DJS_age_gen.reg_info_df])
            work_gen = pd.concat([work_gen, DJS_age_gen.areas])
    reg_info_gen = reg_info_gen.round(3)
    work_gen = work_gen.round(3)
    params.update({'hide_labels': (False, False)})
    
if Gender:   
    # =============================================================================
    # Comparing DJS of adults and old people
    # =============================================================================
    
    # Which subjects are males
    M_info = meta_info_red[meta_info_red['Gender'] == 'M']
    M_group = {i:'Male' for i in M_info['Subject']}
    
    # Which subjects are females
    F_info = meta_info_red[meta_info_red['Gender'] == 'F']
    F_group = {i:'Female' for i in F_info['Subject']}
    
    age_dict = dict(M_group, **F_group)
    
    # Creating the df for Age population 
    Fukuchi_mean_gender = Fukuchi_df.rename(columns= age_dict, level=0).mean(level=[0,1], axis=1)
    #Merging column labels
    Fukuchi_mean_gender.columns = Fukuchi_mean_gender.columns.map('{0[0]} {0[1]}'.format)
    Fukuchi_sd_gender = Fukuchi_df.rename(columns= age_dict, level=0).std(level=[0,1], axis=1)
    #Merging column labels
    Fukuchi_sd_gender.columns = Fukuchi_sd_gender.columns.map('{0[0]} {0[1]}'.format)
    
    Fukuchi_gender =  create_df(Fukuchi_mean_gender, Fukuchi_sd_gender)
    
    # The best are 4 and 13
    # df_turn_gender = hyperparams(Fukuchi_gender)
    
    # =============================================================================
    # Setting variables and plotting in individuals
    # =============================================================================
    
    
    Fukuchi_gen = ankle_DJS(Fukuchi_gender, 
                        features= ['Ankle Dorsi/Plantarflexion ', 
                                  'Vertical Force',
                                    'Ankle Dorsi/Plantarflexion',
                                    'Ankle'], 
                        exp_name = 'Fukuchi Gender Comparison analysis')
    
    all_dfs_gender = Fukuchi_gen.extract_df_DJS_data(idx=[0,2,1,3])
    all_dfs_gender = Fukuchi_gen.interpolate_ankledf(times, True)
    df_turn_gender = best_hyper(all_dfs_gender, save='Fukuchi/best_params_gender.csv',
                             smooth_radius=smooth_,
                             cluster_radius=cluster_, verbose=False)
    
    
    Fukuchi_gen.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_gen.deg_to_rad()
    total_work_gen = Fukuchi_gen.total_work() 
    
    # =============================================================================
    # Plotting QS one by one comparing adults and elderly people
    # =============================================================================
    cols_to_joint ={item: (num, num+11) for num, item in enumerate(meta_info_red['Mode'].unique())}
    
    for num, key in enumerate(cols_to_joint.keys()):
        if num == 2 or num == 3 or num == 7:
            params.update({'hide_labels':(False, False)})
        else:
            params.update({'hide_labels':(False, True)})
        DJS_gender = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=1.5, fig_size=[3.0,2.5], params=params)
        fig5 = DJS_gender.plot_DJS(Fukuchi_gen.all_dfs_ankle, 
                            cols=list(cols_to_joint[key]), rows= np.r_[0,2],
                            title="Ankle DJS gender group comparison at {}".format(key), 
                            legend=True, reg=df_turn_gender.loc[idx[:,'mean'],:], header=None,
                            integration= True, rad = True)
        if num == 0:
            reg_info_gen = pd.DataFrame(DJS_gender.reg_info_df)
            work_gen = pd.DataFrame(DJS_gender.areas)
        else:
            reg_info_gen = pd.concat([reg_info_gen, DJS_gender.reg_info_df])
            work_gen = pd.concat([work_gen, DJS_gender.areas])
    reg_info_gen = reg_info_gen.round(3)
    work_gen = work_gen.round(3)
    params.update({'hide_labels': (False, False)})
    
if individuals:

    params_ind = {'sharex':True, 'sharey':True, 'color_DJS':['slategray']*50, 
                 'color_reg':['black']*50, 'color_symbols': ['slategray']*50, 
                 'arr_size': 6, 'left_margin': 0.15, 'DJS_linewidth': 0.2, 
                 'reg_linewidth': 1.0, 'grid': False, 'alpha_prod': 0.4,
                 'alpha_absorb': 0.1, 'yticks': np.arange(-0.25, 2.25, 0.25)}
    
    #Do not forget to interpolate and see if the error does not appear
    times=2
    test_ind = False
    if test_ind:
        for num, lev1 in enumerate(Fukuchi_df_nan.columns.get_level_values(1).unique()):
            Fukuchi_ind = ankle_DJS(Fukuchi_df_nan.loc[:,idx[:,lev1]], 
                                features= ['Ankle Dorsi/Plantarflexion ', 
                                          'Vertical Force',
                                            'Ankle Dorsi/Plantarflexion',
                                            'Ankle'], 
                                exp_name = 'Fukuchi individuals Comparison analysis')
            
            all_dfs_ind = Fukuchi_ind.extract_df_DJS_data(idx=[0,2,1,3])
            all_dfs_ind = Fukuchi_ind.interpolate_ankledf(times=times, replace=True)
    
            Fukuchi_ind.energy_calculation()
            #Sensitive results may vary when integrating degrees, the best is to do in radians
            Fukuchi_ind.deg_to_rad()
            total_work_ind = Fukuchi_ind.total_work() 
            
            # If calculating the best params is desired
            optimize_params = False
            if optimize_params:
                best_df_turn = best_hyper(all_dfs_ind, save='Fukuchi/best_params_{}.csv'.format(lev1),
                                         smooth_radius=[2,3,4],
                                         cluster_radius=range(25,30), verbose=False)
            else:
                best_df_turn = pd.read_csv('Fukuchi/best_params_{}.csv'.format(lev1), index_col=[0,1])
            DJS_ind = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=[6,7],
                                      alpha=5.0, fig_size=[7*2,6*2], params=params_ind)
            
            fig5 = DJS_ind.plot_DJS(Fukuchi_ind.all_dfs_ankle, 
                                cols=None, rows= np.r_[0,2],
                                title="Ankle DJS subject comparison at {}".format(lev1), 
                                legend=True, reg=best_df_turn, header=lev1,
                                integration= True, rad = True)
    
            if num == 0:
                reg_info_ind = pd.DataFrame(DJS_ind.reg_info_df)
                work_ind = pd.DataFrame(DJS_ind.areas)
                df_turn_ind_all = best_df_turn
                total_work_ind_all = total_work_ind
            else:
                df_turn_ind_all = pd.concat([df_turn_ind_all, best_df_turn], axis = 0)
                total_work_ind_all = pd.concat([total_work_ind_all, total_work_ind], axis = 0)
                reg_info_ind = pd.concat([reg_info_ind, DJS_ind.reg_info_df])
                work_ind = pd.concat([work_ind, DJS_ind.areas])
        
        # Storing results
        df_turn_ind_all.to_csv('Fukuchi/best_df_turn_all.csv')
        total_work_ind_all.to_csv('Fukuchi/total_work_ind.csv')
        reg_info_ind.to_csv('Fukuchi/regression_ind.csv')
        work_ind.to_csv('Fukuchi/work_ind.csv')
    else:
        df_turn_ind_all = pd.read_csv('Fukuchi/best_df_turn_all.csv', index_col=[0,1])
        total_work_ind_all = pd.read_csv('Fukuchi/total_work_ind.csv', index_col=[0,1])
        reg_info_ind = pd.read_csv('Fukuchi/regression_ind.csv', index_col=[0,1,2])
        work_ind = pd.read_csv('Fukuchi/work_ind.csv', index_col=[0,1])
    
    
    reg_info_ind = reg_info_ind.round(3)
    #Adjusting bad R2 results with MSE
    work_ind = work_ind.round(3)  

    #reordering levels
    reg_info_ind = reg_info_ind.sort_index(level=0, axis=0)
    work_ind = work_ind.sort_index(level=0, axis=0)
    df_turn_ind_all = df_turn_ind_all.sort_index(level=0, axis=0)
    df_turn_ind_all = df_turn_ind_all.reindex(new_dict_labels, level=1, axis=0)
    df_turn_ind_allGC = df_turn_ind_all.apply(lambda x: x/times)
    
    #How many samples have got a bad R2 but a good MSE (below 0.0001)
    reg_info_badR2 = reg_info_ind.query("R2 <= 0.2 and MSE <= 0.0001") # 55 samples with 
    reg_info_wobad = reg_info_ind.drop(reg_info_badR2.index, axis=0)
    stiff_labels = ['CP', 'ERP', 'LRP','DP']
    R2mean_per_cat = pd.DataFrame({cat: reg_info_wobad.loc[idx[:,:,cat],
                                                       :].mean() for cat in stiff_labels})
    R2std_per_cat = pd.DataFrame({cat: reg_info_wobad.loc[idx[:,:,cat],
                                                       :].std() for cat in stiff_labels})
    #Final results CP: 0.81(0.22), ERP 0.96(0.06), LRP: 0.95(0.11), DP 0.95(0.06)
    # =============================================================================
    #     plotting per subject at different instances 
    # =============================================================================
    per_subject = [False, 'all'] #If plot, if Overground plotting, otherwise is Treadmill
    if per_subject[0]:
        if per_subject[1] == 'over':
            Fukuchi_df_T = Fukuchi_df_nan.drop(['T{:02d}'.format(ind) for ind in range(1,9)], level=1, axis=1)
            df_turn_ind_T =df_turn_ind_all.drop(['T{:02d}'.format(ind) for ind in range(1,9)], level=1, axis=0)
            mod = 'overground'
            fig_s = [2,4]
            sep_ = [1,3]
            alpha_ = 2.5
        elif per_subject[1] == 'tread':
            Fukuchi_df_T = Fukuchi_df_nan.drop(['OC', 'OS', 'OF'], level=1, axis=1)
            df_turn_ind_T =df_turn_ind_all.drop(['OC', 'OS', 'OF'], level=1, axis=0)
            mod = 'treadmill'
            fig_s = [4,6]
            sep_ = [2,4]
            alpha_ = 2.5
        elif per_subject[1] == 'all':
            Fukuchi_df_T = Fukuchi_df_nan
            df_turn_ind_T =df_turn_ind_all
            mod = 'all'
            fig_s = [5,6]
            sep_ = [3,4]
            alpha_ = 3.0
        for num, lev0 in enumerate(Fukuchi_df_T.columns.get_level_values(0).unique()[:5]):
            Fukuchi_ind2 = ankle_DJS(Fukuchi_df_T.loc[:,idx[lev0,:]], 
                                features= ['Ankle Dorsi/Plantarflexion ', 
                                          'Vertical Force',
                                            'Ankle Dorsi/Plantarflexion',
                                            'Ankle'], 
                                exp_name = 'Fukuchi individuals Comparison analysis') 
            #Sensitive results may vary when integrating degrees, the best is to do in radians
            Fukuchi_ind2.extract_df_DJS_data(idx=[0,2,1,3])
            if mod == 'overground':
                Fukuchi_ind2.change_labels([r'$v* = 0.3(0.04)$', r'$v* = 0.43(0.05)$',
                                            r'$v* = 0.55(0.06)$'], level=1)
            elif mod == 'treadmill':
                Fukuchi_ind2.change_labels(vel_labels[:Fukuchi_ind2.all_dfs_ankle.shape[1]], level=1)
            elif mod == 'all':
                Fukuchi_ind2.change_labels(vel_labels, level=1)
            Fukuchi_ind2.interpolate_ankledf(times, True)
            Fukuchi_ind2.deg_to_rad()
            
            DJS_ind2 = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=sep_,
                                      alpha=alpha_, fig_size=fig_s, params=params_ind)
            df_sub = Fukuchi_ind2.all_dfs_ankle.droplevel(0, axis=1)
            df_sub.columns =  pd.MultiIndex.from_product([list(df_sub.columns),['mean']])
            fig10 = DJS_ind2.plot_DJS(df_sub, 
                                cols=None, rows= np.r_[0,2],
                                title="Ankle DJS subject comparison in subject {} on {}".format(lev0, mod), 
                                legend=True, reg=df_turn_ind_T.loc[idx[lev0,:]], header=None,
                                integration= True, rad = True)
    # =============================================================================
    #     Adapting the general df to ANOVA study
    # =============================================================================
    decimal = 2 #How many decimals you want to show
    if statistics:
        #Knowing which values were out of the confidence
        summary_df_turn = rp.summary_cont(reg_info_ind, decimals=decimal)
        #converting all to absolute values as we would like to compare stiffness magnitudes
        # reg_info_ind['stiffness'] = reg_info_ind['stiffness'].abs() 
        #Do not do this so far, there are opposite stiffneses
        #For high stiffnesses we are setting 25 as it was the maximum value found
        reg_info_ind['stiffness'][reg_info_ind['stiffness'] >= 25.0] = 20.0
        meta_info_anova = meta_info_red
        
        #Creating a categorical value to define speeds from Very Slow to Very Fast
        speed_cat = {'OC': 'Free', 'OS':'Slow', 'OF':'Fast', 'T01': 'Very Slow',
                     'T02': 'Very Slow', 'T03': 'Slow', 'T04': 'Slow', 'T05': 'Free',
                     'T06': 'Fast', 'T07': 'Fast', 'T08': 'Very Fast'}
        
        meta_info_anova.index = Fukuchi_df.columns
        meta_info_anova['speed'] = meta_info_anova.index.get_level_values(1).map(speed_cat)
        
        #Creating categorical value for Overground and Treadmill
        meta_info_anova['mode'] = [x[0] for x in meta_info_anova.index.get_level_values(1)]
        meta_info_anova['mode'] = meta_info_anova['mode'].replace(['T', 'O'], ['Treadmill', 
                                                                               'Overground'])
        # bad samples
        bad_samples = [('S22', 'T07'),('S23', 'T06'), ('S27', 'T07'),('S36', 'T07'),
                       ('S06', 'T08'),('S23', 'T06'), ('S15', 'OC'), #Bad regressions
                       ('S36', 'T01'),('S15', 'T03') #Bad regressions
                       ]
        
        
    # =============================================================================
    #         #Metrics 
    # =============================================================================
        metrics = pd.concat([reg_info_ind['R2'].unstack(),reg_info_ind['MSE'].unstack()], axis=1)
        metrics.columns = pd.MultiIndex.from_product([['R2', 'MSE'], stiff_labels])
        metrics = metrics.drop(bad_samples, axis=0)
        #Verifying those with good MSE and bad R2
        who_badR2 = {item: metrics[metrics.MSE[item] <= 1e-3][metrics.R2[item] <= 0.6] \
                     for item in stiff_labels}
        metrics_wobad = {item: metrics.loc[:,idx[:,item]].drop(who_badR2[item].index, axis=0) \
                     for item in stiff_labels}
        
        metrics_wobad_mean = pd.concat({item: metrics_wobad[item].mean() for item in stiff_labels}, axis=0)
        metrics_wobad_std = pd.concat({item: metrics_wobad[item].std() for item in stiff_labels}, axis=0)
        metrics_wobad_res = pd.concat([metrics_wobad_mean, metrics_wobad_std], axis=1)
        metrics_wobad_res = metrics_wobad_res.droplevel(2, axis=0)
        metrics_wobad_res.columns = ['Mean', 'SD']
        # metrics_wobad_res.drop(['CP'], level=0).mean(axis=0,level=1) to know R2 for the rest of stiffness
        meta_info_anova = meta_info_anova.reindex(Fukuchi_df_nan.columns, axis=0)
        meta_info_anova = meta_info_anova.drop(meta_info_anova.columns[np.r_[0,8:15,16,17]], axis=1)
    
        #Adding All results
        results_df = pd.concat([df_turn_ind_allGC, work_ind, 
                                reg_info_ind['stiffness'].unstack()], axis=1)
        #Appending to anova df
        meta_info_anova = pd.concat([meta_info_anova, results_df], axis=1)
        
        #Removing samples that visually are not coherent
        meta_info_anova = meta_info_anova.drop(bad_samples, axis=0)
        Fukuchi_df_nan = Fukuchi_df_nan.drop(bad_samples, axis=1)
        Fukuchi_df_export = Fukuchi_df_nan.drop(['Vertical Force','Ankle Power [W]'],level=0)
        Fukuchi_df_export.to_csv("Fukuchi/dynamic_data_Fukuchi.csv")
        # =============================================================================
        #         How many are negatives and in which cases
        # =============================================================================
        CP_negative = meta_info_anova[meta_info_anova.CP <= 0]
        LRP_negative = meta_info_anova[meta_info_anova.LRP <= 0]
        # =============================================================================
        #         About work, look for narrow loops and directions
        # =============================================================================
        work_ind_wobad = work_ind.drop(bad_samples, axis=0)
        cw = meta_info_anova.query("direction == 'cw'")
        ccw = meta_info_anova.query("direction == 'ccw'")
        narrow = meta_info_anova[meta_info_anova['work prod'] <= 0.02]
        #Let us continue here
        # =============================================================================
        #         #Redifining labels
        # =============================================================================
        dep_vars = meta_info_anova.columns[np.r_[11:18,19:23]]
        
        labels = ['Init {} '.format(i)+r'$[\%GC]$' for i in ['ERP', 'LRP', 'DP', 'S', 'TS']]
        labels.extend(['Work Absorbed '+r'$\frac{J}{kg}$', 'Net Work '+r'$\frac{J}{kg}$'])
        labels.extend(['Stiffness {}'.format(stiff)+r'$\frac{Nm}{kg \times rad}$' for stiff in stiff_labels])
        
        #Change the column order
        labels_complete = list(meta_info_anova.columns[:-4])
        labels_complete.extend(stiff_labels)
        meta_info_anova = meta_info_anova.reindex(columns=labels_complete)
        # =============================================================================
        #     Applying statistics in gender
        # =============================================================================
        
        # Performing variance analysis
        Fem = meta_info_anova[meta_info_anova['Gender'] == 'F']
        Male = meta_info_anova[meta_info_anova['Gender']== 'M']
        
        #Main information
        summary_Fem = multi_idx('Female', rp.summary_cont(Fem.groupby(Fem['speed']), decimals=decimal).round(2).T, 
                                idx=False)
        summary_Male = multi_idx('Male', rp.summary_cont(Male.groupby(Male['speed']), decimals=decimal).round(2).T, 
                                 idx=False)
        #Let's perform the Bartetts's test whose Null Hypothesis is that the 
        #variances are equal. We will use a significance level of 5.0%
        var_gender = {item: stats.bartlett(Fem[item], 
                     Male[item]).pvalue for item in dep_vars}
        #Variances are equal, excepting point 1, 5, ERP, CP and DP, It means that the F-statistics 
        # is not reliable when ANOVA is applied
        
        # =============================================================================
        #     ttest student analysis
        # =============================================================================
        
        # Assumptions:
        #     1. Independent samples
        #     2. Large enough sample size or observations come from a normally-distributed 
        #     population
        #     3. Variances are equal
        
        ttest_gender = {item: stats.ttest_ind(Fem[item], Male[item], 
                        equal_var=var_gender[item] > 0.05).pvalue for item in dep_vars}
        
        #There is then a statistical significant difference between gender
        #in the two analyzed groups. The DP, ERP, and CP, point 1 is significantly different
        #We need to see which is higher
        
        # summary content
        summary_dep_variables_gen = rp.summary_cont(meta_info_anova.loc[:,dep_vars], decimals=decimal)
        summary_gender = rp.summary_cont(meta_info_anova.groupby(meta_info_anova['Gender']), decimals=decimal).T
        
          
        # =============================================================================
        #     Plot statistics
        # =============================================================================
        
        #Seeing statistical differences between gender and the significan
        fig6, axes = plt.subplots(2,2, figsize = (8,8))
        deps_gen = dep_vars[np.r_[1,7,8,9]]
        labels_gen = np.array(labels)[np.r_[1,7,8,9]]
        for num, ax in enumerate(np.ravel(axes)):
            sns.boxplot(x='Gender', y=deps_gen[num], data=meta_info_anova, ax=ax)
            ax.set_ylabel(labels_gen[num])
    
        # fig6.suptitle('Variables with statistical differences in gender', fontsize = 18)
        fig6.savefig('Fukuchi/stats_diff_gender.png')
    
          
        # =============================================================================
        #     Applying statistics in Ages
        # =============================================================================
        
        # Performing variance analysis
        adults = meta_info_anova[meta_info_anova['AgeGroup']=='Young']
        old = meta_info_anova[meta_info_anova['AgeGroup']=='Older']
        #Let's perform the Bartetts's test whose Null Hypothesis is that the 
        #variances are equal. We will use a significance level of 5.0%
        var_ages = {item: stats.bartlett(adults[item], 
                     old[item]).pvalue for item in dep_vars}
        #Variances are unequal in point 0, work prod and DP
        
        # =============================================================================
        #     ttest student analysis
        # =============================================================================
        
        # Assumptions:
        #     1. Independent samples
        #     2. Large enough sample size or observations come from a normally-distributed 
        #     population
        #     3. Variances are equal
        
        ttest_ages = {item: stats.ttest_ind(adults[item], old[item], 
                        equal_var=var_ages[item] > 0.05).pvalue for item in dep_vars}
        
        #There is then a statistical significant difference between gender
        #in the two analyzed groups. The work prod, work abs and DP phase showed statistical differences
        #We need to see which is higher
        
        # summary content
        summary_dep_variables_age = rp.summary_cont(meta_info_anova.loc[:,dep_vars], decimals=decimal)
        summary_ages = rp.summary_cont(meta_info_anova.groupby(meta_info_anova['AgeGroup']), decimals=decimal)
        summary_adults = multi_idx('Adults', 
                         rp.summary_cont(adults.groupby(adults['speed']), decimals=decimal).T, idx=False)
        summary_old = multi_idx('Elderly', 
                                rp.summary_cont(old.groupby(old['speed']), decimals=decimal).T, idx=False)
        #OLS method
        results_ages = {item: ols("Q('{}') ~ C(AgeGroup)".format(item), data=meta_info_anova).fit() \
                          for item in dep_vars}
        #If p-value is greater than the alpha (0.05) value then there is no association between the 
        #dependent variable and AgeGroup
        table_ANOVA_age = pd.Series({item: sm.stats.anova_lm(results_ages[item], 
                                               typ=2).iloc[0,-1] for item in dep_vars})
        #Conclusion: there is an association between Agegroups and work abs, prod and stiffness DP.
        
        # =============================================================================
        # Tukey's Analysis
        # Null hypothesis: There is no significant difference betweem groups
        # =============================================================================
        mc_ages = {item: MultiComparison(meta_info_anova[item], 
                        meta_info_anova['AgeGroup']) for item in dep_vars}
        mc_results_ages = pd.DataFrame({item: [mc_ages[item].tukeyhsd().reject[0],
                                       mc_ages[item].tukeyhsd().pvalues[0],
                                       mc_ages[item].tukeyhsd().meandiffs[0]]
                                       for item in dep_vars}, index=['reject','p value','mean diff']).T
        
        # conclusion: We can reject the null hypothesis in which there is no significant 
        # differences in age groups only for the dependent variables work prod, abs and DP,
        # the remaining ones do not have statistical differences.
    
        # =============================================================================
        #     Plot statistics for age groups
        # =============================================================================
        
        #Seeing statistical differences between gender and the significant
        fig7, axes = plt.subplots(2,2, figsize = (8,8))
        deps_age = dep_vars[np.r_[5:8,9]]
        labels_age = np.array(labels)[np.r_[5:8,9]]
        for num, ax in enumerate(np.ravel(axes)):
            sns.boxplot(x='AgeGroup', y=deps_gen[num], data=meta_info_anova, ax=ax)
            ax.set_ylabel(labels_age[num])
    
        # fig7.suptitle('Variables with statistical differences in Age groups', fontsize = 18)
        fig7.savefig('Fukuchi/stats_diff_ages.png')
    
        # =============================================================================
        #     Applying statistics overground vs treadmill
        # =============================================================================
        
        # Performing variance analysis
        overground = meta_info_anova[meta_info_anova['mode'] == 'Overground']
        treadmill = meta_info_anova[meta_info_anova['mode'] == 'Treadmill']
        #We need to remove the very fast and slow in order to homogenized the speed
        #dropping T01, T02, T04, T06 and T08
        treadmill_adj = treadmill.drop(['T01','T02','T04','T06','T08'], level=1,axis=0)
    
        # concat the df for further operations 
        mode_df = pd.concat([overground, treadmill_adj], axis=0)
        #Let's perform the Bartetts's test whose Null Hypothesis is that the 
        #variances are equal. We will use a significance level of 5.0%
        var_mode = {item: stats.bartlett(overground[item], 
                     treadmill_adj[item]).pvalue for item in dep_vars}
        #Variances are point 2 and work prod, rest of them are unequal
        
        # =============================================================================
        #     ttest student analysis
        # =============================================================================
        
        # Assumptions:
        #     1. Independent samples
        #     2. Large enough sample size or observations come from a normally-distributed 
        #     population
        #     3. Variances are equal, if not apply weltch test
        
        # Does the samples come from a normally distributed population
        ttest_mode = {item: stats.ttest_ind(overground[item], treadmill_adj[item], 
                        equal_var=var_mode[item] > 0.05).pvalue for item in dep_vars}
        
        #There is then a statistical significant difference between gender
        #in the two analyzed groups. Point 0, 2, work prod (0.054) do not show statistical differences
        #The rest of them were statistically significant.
        
        # summary content
        summary_dep_variables_mode = rp.summary_cont(mode_df.loc[:,dep_vars], decimals=decimal)
        summary_mode = rp.summary_cont(mode_df.groupby(mode_df['mode']), decimals=decimal)
        
        # =============================================================================
        #     Statistical Analysis for AGes taking out VS and VF
        # =============================================================================
        
        adults_o = overground.query("AgeGroup == 'Young'")
        olds_o = overground.query("AgeGroup == 'Older'")
        
        var_ages_m = {item: stats.bartlett(adults_o[item], 
                     olds_o[item]).pvalue for item in dep_vars}
        
        ttest_ages_m = {item: stats.ttest_ind(adults_o[item], olds_o[item], 
                        equal_var=var_ages_m[item] > 0.05).pvalue for item in dep_vars}
        # THERE IS NO STATISTICAL DIFFERENCES BETWEEN ELDERLY AND ADULTS IN GENERAL
        olds_slow = olds_o.query("speed == 'Slow'")
        adults_slow = adults_o.query("speed == 'Slow'")
        olds_free = olds_o.query("speed == 'Free'")
        adults_free = adults_o.query("speed == 'Free'")
        olds_fast = olds_o.query("speed == 'Fast'")
        adults_fast = adults_o.query("speed == 'Fast'")
        groups_ages = [olds_slow, adults_slow, olds_free, 
                       adults_free, olds_fast, adults_fast]
        # Let us determine the normality in subgroups
        normality_ages = pd.concat([pd.DataFrame({item: stats.shapiro(data_ages[item]) \
                         for item in dep_vars}) for data_ages in groups_ages], axis=0)
        normality_ages.index = pd.MultiIndex.from_product([['Slow', 'Free', 'Fast'],
                                                           ['Olds','Adults'],['stats', 'p value']])
        #MOST OF THE SUBGROUPS ARE NORMAL, HOWEVER A CONSIDERABLE PORCENTAGE IS NOT
        
        # Determining statistical differences between speeds on subgroups
        kruskal_speed_AS = {item: kruskal(olds_slow[item].values, 
                                          adults_slow[item].values).pvalue for item in dep_vars}
        #Point 1
        kruskal_speed_AC = {item: kruskal(olds_free[item].values, 
                                          adults_free[item].values).pvalue for item in dep_vars}
        #Point 1 is the only one different (0.04)
        kruskal_speed_AF = {item: kruskal(olds_fast[item].values, 
                                          adults_fast[item].values).pvalue for item in dep_vars}
        #Point 5
        #Analysis between 
        kruskal_speed_olds = {item: kruskal(olds_o.query("speed == 'Slow'")[item].values, 
                                          olds_o.query("speed == 'Free'")[item].values,
                                          olds_o.query("speed == 'Fast'")[item].values).pvalue for item in dep_vars}
        #Stats diff in point 1,3,4,abs and prod, and CP
        # Let us proceed with dunn analysis on those outputs in which 
        dunn_old = pd.concat([sp.posthoc_dunn(olds_o, val_col = item, 
                        group_col= 'speed', p_adjust='holm') for item in dep_vars[np.r_[0,2,3,4,5,6,7]]], axis=0)
        dunn_old.index = pd.MultiIndex.from_product([dep_vars[np.r_[0,2,3,4,5,6,7]],
                                                     list(dunn_old.index.unique())])
        dunn_oldbool = dunn_old.apply(lambda x: x < 0.05)
                           
        
        kruskal_speed_adults = {item: kruskal(adults_o.query("speed == 'Slow'")[item].values, 
                                          adults_o.query("speed == 'Free'")[item].values,
                                          adults_o.query("speed == 'Fast'")[item].values).pvalue for item in dep_vars}
        #Stats diff in point 3,4, abs and prod, CP and LRP
        dunn_adults = pd.concat([sp.posthoc_dunn(adults_o, val_col = item, group_col = 'speed',
                                        p_adjust='holm') for item in dep_vars[np.r_[2,3,5,6,7,10]]], axis=0)
        dunn_adults.index = pd.MultiIndex.from_product([dep_vars[np.r_[2,3,5,6,7,10]],list(dunn_adults.index.unique())])
        dunn_adultsbool = dunn_adults.apply(lambda x: x < 0.05)    
        summary_adults_m = multi_idx('Adults', 
                                     rp.summary_cont(adults_o.groupby(adults_o['speed']), decimals=decimal).T, idx=False)
        summary_old_m = multi_idx('Elderly', 
                                  rp.summary_cont(olds_o.groupby(olds_o['speed']), decimals=decimal).T, 
                                  idx=False)
        summary_over = multi_idx('Overground', 
                                 rp.summary_cont(overground.groupby(overground['speed'])).round(2).T, idx=False)
        summary_tread_adj = multi_idx('Treadmill', 
                                      rp.summary_cont(treadmill_adj.groupby(treadmill_adj['speed'])).round(2).T, idx=False)
    
        #OLS method
        results_mode = {item: ols("Q('{}') ~ C(mode)".format(item), data=mode_df).fit() \
                          for item in dep_vars}
        #If p-value is greater than the alpha (0.05) value then there is no association between the 
        #dependent variable and AgeGroup
        table_ANOVA_mode = pd.Series({item: sm.stats.anova_lm(results_mode[item], 
                                               typ=2).iloc[0,-1] for item in dep_vars})
        #Conclusion: there is an association between Agegroups and work abs, prod and stiffness DP.
        
        # =============================================================================
        # Tukey's Analysis
        # Null hypothesis: There is no significant difference betweem groups
        # =============================================================================
        mc_mode = {item: MultiComparison(mode_df[item], 
                        mode_df['mode']) for item in dep_vars}
        mc_results_mode = pd.DataFrame({item: [mc_mode[item].tukeyhsd().reject[0],
                                       mc_mode[item].tukeyhsd().pvalues[0],
                                       mc_mode[item].tukeyhsd().meandiffs[0]]
                                       for item in dep_vars}, index=['reject','p value','mean diff']).T
        
        # conclusion: We can reject the null hypothesis in which there is no significant 
        # differences in age groups only for the dependent variables work prod, abs and DP,
        # the remaining ones do not have statistical differences.
    
        # =============================================================================
        #     Plot statistics for walking mode
        # =============================================================================
        
        #Seeing statistical differences between gender and the significant
        fig8, axes = plt.subplots(2,4, figsize = (16,8))
        deps_mode = dep_vars[np.r_[1,3:10]]
        labels_mode = np.array(labels)[np.r_[1,3:10]]
        for num, ax in enumerate(np.ravel(axes)):
            sns.boxplot(x='mode', y=deps_mode[num], data=mode_df, ax=ax)
            ax.set_ylabel(labels_mode[num])
        # fig8.suptitle('Variables with statistical differences in Overground vs Treadmill', fontsize = 18)
        fig8.savefig('Fukuchi/stats_diff_mode.png')
        
        # =============================================================================
        #     Differences within speed and mode
        # =============================================================================
        norm_speed_O = {item: stats.shapiro(overground[item]) for item in dep_vars}
        norm_speed_T = {item: stats.shapiro(treadmill_adj[item]) for item in dep_vars}
        #outputs are not normal distributed
        O_slow = overground.query("speed == 'Slow'")
        O_free = overground.query("speed == 'Free'")
        O_fast = overground.query("speed == 'Fast'")
        T_vslow = treadmill.query("speed == 'Very Slow'")
        T_slow = treadmill.query("speed == 'Slow'")
        T_free = treadmill.query("speed == 'Free'")
        T_fast = treadmill.query("speed == 'Fast'")
        T_vfast = treadmill.query("speed == 'Very Fast'")
        
        #Let us see if we see same variances in overground 
        var_speed_O = {item: stats.bartlett(O_slow[item], O_free[item], O_fast[item]).pvalue for item in dep_vars}
        
        #Now for treadmill
        var_speed_T = {item: stats.bartlett(T_vslow[item], T_slow[item], T_free[item], 
                                            T_fast[item], T_vfast[item]).pvalue for item in dep_vars}
        
        # As variances are different we would need to implement a non-parametric method
        # We will apply kruskal wallis
        
        #Null hypothesis
        # the null hypothesis is that the medians of all groups are equal, 
        # and the alternative hypothesis is that at least one population median 
        # of one group is different from the population median of at least one other group.
        
        kruskal_speed_O = {item: kruskal(O_slow[item].values, O_free[item].values, 
                                         O_fast[item].values).pvalue for item in dep_vars}
        kruskal_speed_T = {item: kruskal(T_vslow[item].values, T_slow[item].values, 
                                         T_free[item].values, T_fast[item].values, 
                                         T_vfast[item].values).pvalue for item in dep_vars}
        kruskal_speed_T_adj = {item: kruskal(T_slow[item].values, 
                                     T_free[item].values, 
                                     T_fast[item].values).pvalue for item in dep_vars}
        kruskal_speed_MS = {item: kruskal(O_slow[item].values, 
                                          T_slow[item].values).pvalue for item in dep_vars}
        #Stat diff in points 4, work abs, CP, ERP, LRP, DP
        kruskal_speed_MC = {item: kruskal(O_free[item].values, 
                                          T_free[item].values).pvalue for item in dep_vars}
        #Stats diff in point 5, work prod and abs, CP and DP
        kruskal_speed_MF = {item: kruskal(O_fast[item].values, 
                                          T_fast[item].values).pvalue for item in dep_vars}
        #Stats diff in point 4,5, abs and prod, CP and DP.
        #At 5% the null hypothesis is rejected for:
            # Overground: point 0, point 2, point 3, work abs, and work prod
            # Treadmill: all were rejected excepting DP (0.07)
            
        # Let us proceed with dunn analysis on those outputs in which 
        dunn_O = pd.concat([sp.posthoc_dunn(overground, val_col = item, 
                        group_col= 'speed', p_adjust='holm') for item in dep_vars[np.r_[0,2:8,10]]], axis=0)
        dunn_O.index = pd.MultiIndex.from_product([dep_vars[np.r_[0,2:8,10]],list(dunn_O.index.unique())])
        dunn_Obool = dunn_O.apply(lambda x: x < 0.05)
                           
        dunn_T = pd.concat([sp.posthoc_dunn(treadmill_adj, val_col = item, group_col = 'speed', #Take out adj for Very classes 
                                        p_adjust='holm') for item in dep_vars[np.r_[0,2,3,5,6]]], axis=0)
        dunn_T.index = pd.MultiIndex.from_product([dep_vars[np.r_[0,2,3,5,6]],list(dunn_T.index.unique())])
        dunn_Tbool = dunn_T.apply(lambda x: x < 0.05)
        
        
        
        legend_ = [1,1,0]*3
        fig9, axes = plt.subplots(3,3, figsize = (12,12))
        fig9.tight_layout()
        fig9.subplots_adjust(wspace=.3, left=0.1)
        deps_mod_ = dep_vars[np.r_[0:4,5:10]]
        labels_mod_ = np.array(labels)[np.r_[0:4,5:10]]
        for num,  ax in enumerate(np.ravel(axes)):
            sns.boxplot(x='mode', y=deps_mod_[num], hue='speed', 
                        data=mode_df, ax=ax, hue_order = ['Slow', 'Free', 'Fast'])
            ax.set_ylabel(labels_mod_[num])
            if bool(legend_[num]):
                ax.get_legend().remove()
            else:
                ax.legend(loc='upper right')
            ax.set_xlabel('Environment')
        # fig9.suptitle('Variables with statistical differences in OvsT and speed', fontsize = 18)
        fig9.savefig('Fukuchi/stats_diff_mode_speed.png')
        
        
        # =============================================================================
        #     Differences within speed and gender
        # =============================================================================
        Fem = mode_df[mode_df['Gender'] == 'F']
        Male = mode_df[mode_df['Gender'] == 'M']
        norm_speed_F = {item: stats.shapiro(Fem[item]) for item in dep_vars}
        norm_speed_M = {item: stats.shapiro(Male[item]) for item in dep_vars}
        #Main information
        summary_Fem_m = multi_idx('Female', rp.summary_cont(Fem.groupby(Fem['speed']), 
                                                            decimals=decimal).T, idx=False)
        summary_Male_m = multi_idx('Male', rp.summary_cont(Male.groupby(Male['speed']), 
                                                           decimals=decimal).T, idx=False)
        
        # =============================================================================
        #         Summary concat
        # =============================================================================
        summary_concat = pd.concat([summary_adults_m, summary_old_m, summary_Fem_m, 
                                    summary_Male_m, summary_over, summary_tread_adj], axis=1)
        trials_num = summary_concat.iloc[0,:].astype(np.int64)
        trials_num.name = ('','N')
        #Dropping non independent vars
        summary_concat = summary_concat.loc[idx[dep_vars,:],:]
        #Dropping N,and SE
        summary_concat = summary_concat.loc[idx[:,['Mean', 'SD','95% Conf.', 'Interval']],:]
        summary_concat = change_labels(summary_concat, ['Mean', 'SD','95% CI min', '95% CI max'], 
                                       index=True, level=1)
        # summary_concat = change_labels(summary_concat, labels, 
        #                                index=True, level=0)
        summary_concat = pd.concat([pd.DataFrame(trials_num).T, summary_concat], axis=0)
        #Changing order on level 1 columns
        summary_concat = summary_concat.reindex(['Slow', 'Free', 'Fast'], axis=1, level=1)
        # Export to latex
        with open("Fukuchi/table2.tex", "w+") as pt:
            summary_concat.to_latex(buf=pt, col_space=10, longtable=True, multirow=True, 
                                caption='Cuantitative ankle DJS characteristics at different population groups'+\
                                r' three different gait speeds: Slow ({})'.format(vel_labels[-3])+\
                                r', Free ({}) and Fast({})'.format(vel_labels[-2], vel_labels[-1]),
                                label='tab:table2')
        
        summary_N = summary_concat.iloc[0,:]
        summary_N.name = ('Group', 'Speed')
        summary_N.columns = ['Number N']
        mult_idx_N = pd.MultiIndex.from_product([['Young adult (A) (age $< 31$ years old)',
                                                  'Elder adult (E) (age $\ge 54$ years old',
                                                  'Female (F)',
                                                  'Male (M)',
                                                  'Overground (O)',
                                                  'Treadmill (T)'],
                                                 ['Slow (S) ($v^*\le 0.34$)',
                                                  'Free (C) ($0.37 < v^* \le 0.48$)',
                                                  'Fast (F) ($v^*>0.48$)']])
        summary_N.index = mult_idx_N
        with open("Fukuchi/tableN.tex", "w+") as pt:
            summary_N.to_latex(buf=pt, col_space=10, longtable=False, multirow=True, 
                        caption='caption',
                        label='tab:tableN')
        #To csv
        summary_concat.to_csv('Fukuchi/summary_concat.csv')
        #outputs are not normal distributed
        
        M_slow = mode_df.query("speed == 'Slow' & Gender == 'M'")
        M_free = mode_df.query("speed == 'Free' & Gender == 'M'")
        M_fast = mode_df.query("speed == 'Fast' & Gender == 'M'")
        F_slow = mode_df.query("speed == 'Slow' & Gender == 'F'")
        F_free = mode_df.query("speed == 'Free' & Gender == 'F'")
        F_fast = mode_df.query("speed == 'Fast' & Gender == 'F'")
        
        #Let us see if we see same variances in Female groups
        var_speed_F = {item: stats.bartlett(F_slow[item], F_free[item], F_fast[item]).pvalue for item in dep_vars}
        #Variances are equal in point 1, point 4, work abs and DP
        
        #Now for Males
        var_speed_M = {item: stats.bartlett(M_slow[item], M_free[item], 
                                            M_fast[item]).pvalue for item in dep_vars}
        #Variances are equal in work abs, DP and ERP
        
        # As variances are different we would need to implement a non-parametric method
        # We will apply kruskal wallis
        
        #Null hypothesis
        # the null hypothesis is that the medians of all groups are equal, 
        # and the alternative hypothesis is that at least one population median 
        # of one group is different from the population median of at least one other group.
        
        kruskal_speed_F = {item: kruskal(F_slow[item].values, F_free[item].values, 
                                         F_fast[item].values).pvalue for item in dep_vars}
        #Point 1, point 3, 4, 5, abs, prod, CP, and LRP
        kruskal_speed_M = {item: kruskal(M_slow[item].values, 
                                         M_free[item].values, 
                                         M_fast[item].values).pvalue for item in dep_vars}
        # Points 1,2,3,4,abs, prod, LRP
        kruskal_speed_Gen = {item: kruskal(M_slow[item].values, M_free[item].values, 
                                         M_fast[item].values,F_slow[item].values, 
                                         F_free[item].values, 
                                         F_fast[item].values).pvalue for item in dep_vars}
        
        #We will use kruskal wallis to see the differences between speeds
        kruskal_speed_GenS = {item: kruskal(F_slow[item].values, 
                                            M_slow[item].values).pvalue for item in dep_vars}
        #Statistical differences in point 1, CP, DP, ERP, LRP
        kruskal_speed_GenC = {item: kruskal(F_free[item].values, 
                                            M_free[item].values).pvalue for item in dep_vars}
        # Stats diff DP and ERP
        kruskal_speed_GenF = {item: kruskal(F_fast[item].values, 
                                            M_fast[item].values).pvalue for item in dep_vars}
        # Stats diff DP and ERP
        #At 5% the null hypothesis is rejected for:
            # Females: point 0, point 1, point 2, point 3, work abs, and work prod
            # Males: point 0, point 2, point 3, work abs, and work prod
            # Overall: point 4 is the unique with no statistical differences
        
    # =============================================================================
    #         Determining which values are statistically significant comparing classes at same speed
    # =============================================================================
        def which_significant(dic):
            res = {0.001: [], 0.01:[], 0.05:[]}
            for key, item in dic.items():
                if item <= 0.001:
                    res[0.001].append((key,item))
                elif item <= 0.01:
                    res[0.01].append((key,item))
                elif item <= 0.05:
                    res[0.05].append((key,item))
            return res
        kruskal_etiquete_class = ['{} {}'.format(i,j) for i in ['Age', 'Gender', 'Mode'] for j in ['Slow', 'Free', 'Fast']]
        kruskals_all_class = [kruskal_speed_AS, kruskal_speed_AC, kruskal_speed_AF, 
                        kruskal_speed_GenS, kruskal_speed_GenC, kruskal_speed_GenF,
                        kruskal_speed_MS, kruskal_speed_MC, kruskal_speed_MF]
        kruskals_speed = {key: which_significant(item) for key, item in zip(kruskal_etiquete_class, kruskals_all_class)}
        #Now for classes within the group
        kruskal_etiquete_group = ['Adults', 'Elder', 'Female', 'Male', 'Overground', 'Treadmill']
        kruskal_all_group = [kruskal_speed_adults, kruskal_speed_olds,
                             kruskal_speed_F, kruskal_speed_M,
                             kruskal_speed_O, kruskal_speed_T_adj]
        kruskals_group = {key: which_significant(item) for key, item in zip(kruskal_etiquete_group, 
                                                                            kruskal_all_group)}
        
        # Let us proceed with dunn analysis on those outputs in which 
        dunn_F = pd.concat([sp.posthoc_dunn(Fem, val_col = item, 
                        group_col= 'speed', p_adjust='holm') for item in dep_vars[np.r_[0,2,3,5,6,10]]], axis=0)
        dunn_F.index = pd.MultiIndex.from_product([dep_vars[np.r_[0,2,3,5,6,10]],list(dunn_F.index.unique())])
        dunn_Fbool = dunn_F.apply(lambda x: x < 0.05)
                           
        dunn_M = pd.concat([sp.posthoc_dunn(Male, val_col = item, group_col = 'speed',
                                        p_adjust='holm') for item in dep_vars[np.r_[0,2,3,5,6,7]]], axis=0)
        dunn_M.index = pd.MultiIndex.from_product([dep_vars[np.r_[0,2,3,5,6,7]],list(dunn_M.index.unique())])
        dunn_Mbool = dunn_M.apply(lambda x: x < 0.05)
            
    
        legend_ = [1,0,1]*3
        fig11, axes = plt.subplots(3,3, figsize = (12,12))
        fig11.tight_layout()
        fig11.subplots_adjust(wspace=.3, left=0.1)
        for num,  ax in enumerate(np.ravel(axes)):
            sns.boxplot(x='Gender', y=deps_mod_[num], hue='speed', 
                        data=mode_df, ax=ax, hue_order = ['Slow', 'Free', 'Fast'])
            ax.set_ylabel(labels_mod_[num])
            if bool(legend_[num]):
                ax.get_legend().remove()
            else:
                ax.legend(loc='upper right')
        # fig11.suptitle('Variables with statistical differences in Gender and speed', fontsize = 18)
        fig11.savefig('Fukuchi/stats_diff_gender_speed.png')
        
        # Format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
        cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 
                        'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        
        concat_dunn = [dunn_adults, dunn_old, dunn_F, 
                       dunn_M, dunn_O, dunn_T]
        for num, info in enumerate(concat_dunn):
            fig12, axs = plt.subplots(1,1, figsize=[8,8])
            axs = sp.sign_plot(info, **heatmap_args)
            # axs.set_ylabel(kruskal_etiquete_group[num])
            fig12.savefig('Fukuchi/dunn_results_{}.png'.format(kruskal_etiquete_group[num]))
    
    # Saving a table with the general statistical significance
    #Saving the significance of the speed within the vars of the same class
    significance_group = pd.concat([sp.sign_table(dunn_) for dunn_ in concat_dunn], axis=1)
    significance_group = significance_group.fillna('NS')
    significance_group = significance_group.replace('-', 'NS') 
    significance_group = significance_group.replace('NS', ' ') #Removing NS
    significance_group = significance_group.replace('*', 0.05) #0.05
    significance_group = significance_group.replace('**', 0.01) #0.01
    significance_group = significance_group.replace('***', 0.001) #0.001
    #Changing order on level 1 columns and index
    significance_group.columns = pd.MultiIndex.from_product([kruskal_etiquete_group,['Fast', 'Free', 'Slow']])
    significance_group = significance_group.reindex(['Slow', 'Free', 'Fast'], axis=1, level=1)
    significance_group = significance_group.reindex(['Slow', 'Free', 'Fast'], axis=0, level=1)
    # Reindexing significance
    significance_group = significance_group.reindex(dep_vars[np.r_[0,2,3,4,5,6,7,10]], axis=0, level=0)
    # significance_group = change_labels(significance_group, np.array(labels)[np.r_[0,2,3,4,5,6,7,10]], 
    #                                     level=0)
    #Removing TS as they do not have good significance
    significance_group = significance_group.drop(['point 5'], axis=0, level=0)
    with open("Fukuchi/table3.tex", "w+") as pt:
        significance_group.to_latex(buf=pt, col_space=10, longtable=False, multirow=True, 
                            caption=r'Significant differences (p value) between '+\
                            r'three different gait speeds: Slow ({})'.format(vel_labels[-3])+\
                            r', Free ({}) and Fast({})'.format(vel_labels[-2], vel_labels[-1])+\
                                ' for each population group.',
                            label='tab:table3')

#Meta info_anova to csv
#Creating a categorical value to define speeds from Very Slow to Very Fast
speed_cat_simp = {'OC': 'C', 'OS':'S', 'OF':'F', 'T01': 'VS',
             'T02': 'VS', 'T03': 'S', 'T04': 'S', 'T05': 'C',
             'T06': 'F', 'T07': 'F', 'T08': 'VF'}
meta_info_anova['speed'] = meta_info_anova.index.get_level_values(1).map(speed_cat_simp)
meta_info_anova.to_csv("Fukuchi/meta_info_Fukuchi.csv")
