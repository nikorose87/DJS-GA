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

# =============================================================================
# Settings
# =============================================================================
instances = False
Age_group = False
Age_group_mod = False
Age_group_gen = True
Gender = False

# =============================================================================
# Helper functions
# =============================================================================

def multi_idx(name, df_, idx=True, level=0):
    """
    

    Parameters
    ----------
    name : High level name to set. STR
    df_ : Dataframe to create the multiindex.
    idx : If true It will do over index, otherwise columns.
    level : Either to set in the first level or the second level.

    Returns
    -------
    df_ : Pandas dataframe.

    """
    if idx:
        l_ = df_.index
    else:
        l_ = df_.columns
    if level == 0:
        l_1 = [name]
        l_2 = l_
    else:
        l_1 = l_
        l_2 = [name]
    index_mi = pd.MultiIndex.from_product([l_1, l_2])
    if idx: df_.index = index_mi
    else: df_.columns = index_mi
    return df_
    
def create_df(df_mean, df_std):
    """
    
    Creates specific dataframe according the the mean and SD, in order to return
    a 2 level column dataframe with -1sd, mean and +1sd values
    Parameters
    ----------
    df_mean : TYPE
        DESCRIPTION.
    df_std : TYPE
        DESCRIPTION.

    Returns
    -------
    df_concat : TYPE
        DESCRIPTION.

    """
    _plus = df_mean + df_std
    _minus = df_mean - df_std
    
    #Creating the 2nd level
    _mean = multi_idx('mean', df_mean, level=1, idx=False)
    _plus = multi_idx('+1sd', _plus, level=1, idx=False)
    _minus = multi_idx('-1sd', _minus, level=1, idx=False)
    
    df_concat = pd.concat([_minus, _mean, _plus], axis=1)
    df_concat = df_concat.sort_index(axis=1, level=0)
    
    # Reindexing second level
    df_concat = df_concat.reindex(['-1sd','mean','+1sd'], level=1, axis=1)
    return df_concat

def power(col):
    return col['RAnkleMomentZ']*col['Angular vel [deg / GC]']

def hyperparams(df_, smooth_radius=(4,8), c_radius=(10,14), features=
                ['Ankle Dorsi/Plantarflexion ', 'Vertical Force',
                 'Ankle Dorsi/Plantarflexion',  'Ankle']):
    """
    Parameterization of the curvature settings to see which of them are suitable
    for all the gait instances

    Parameters
    ----------
    df_ : dataframe to analyze the best hyperparameters for Smoothin radious.
    smooth_radius : TYPE, optional
        DESCRIPTION. The default is (4,8).
    c_radius : TYPE, optional
        DESCRIPTION. The default is (10,14).
    features : TYPE, optional
        DESCRIPTION. The default is ['Ankle Dorsi/Plantarflexion ', 'Vertical Force',                 'Ankle Dorsi/Plantarflexion',  'Ankle'].

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
            except ValueError:
                print('parameters {},{} failed'.format(i,j))
                continue
    return df_turn_instance
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
Fukuchi_df_nan = Fukuchi_df.dropna(axis=1)
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
mean_T = {i:[meta_info_valid.query("Type == '{}'".format(i))['Fraude'].mean()] for i in unique_type}
std_T = {i:meta_info_valid.query("Type == '{}'".format(i))['Fraude'].std() for i in unique_type}
for key in mean_T.keys():
    mean_T[key].append(std_T[key]) #Including std in mean

new_dict_labels= ['T01','T02','T03','T04','T05','T06','T07','T08','OS','OC','OF']
for num, old_key in enumerate(mean_T.keys()):
    mean_T[new_dict_labels[num]] = mean_T.pop(old_key)
    
if instances:
    
    # =============================================================================
    #     Let us keep only the piece of df and change labels
    # =============================================================================
    Fukuchi_df_modes = Fukuchi_df.loc[:,idx[:,['T03','OS','T05','OC','T07','OF']]]
    # =============================================================================
    # Performing overall average over modes
    # =============================================================================
    
    # Let us filter only the ankle information of the right foot
    Fukuchi_mean_modes = Fukuchi_df_modes.groupby(level=1, axis=1).mean()
    Fukuchi_sd_modes = Fukuchi_df_modes.groupby(level=1, axis=1).std()
    
    Fukuchi_modes = create_df(Fukuchi_mean_modes, Fukuchi_sd_modes)
    
    
    # =============================================================================
    # Setting variables and plotting in individuals
    # =============================================================================
    
    
    Fukuchi_instance = ankle_DJS(Fukuchi_modes, 
                        features= ['Ankle Dorsi/Plantarflexion ', 
                                  'Vertical Force',
                                   'Ankle Dorsi/Plantarflexion',
                                   'Ankle'], 
                        exp_name = 'Fukuchi Instances variation analysis')
    
    all_dfs_instance = Fukuchi_instance.extract_df_DJS_data(idx=[0,2,1,3])
    all_dfs_instance = Fukuchi_instance.change_labels(["Free O", "Fast O", 'Slow O', 'Slow T', 
                                    'Free T', 'Fast T'])
    df_turn_instance = Fukuchi_instance.get_turning_points(turning_points= 6, 
                                smoothing_radius = 4, cluster_radius= 13)
    
    
    Fukuchi_instance.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_instance.deg_to_rad()
    total_work_instance = Fukuchi_instance.total_work()  
    
    # =============================================================================
    # Obtaining the mechanical work through power instances in regular walking 
    # =============================================================================
    Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
    params = {'sharex':False, 'sharey':False, 'left_margin': 0.2, 'arr_size':12,
              'yticks': np.arange(-0.25, 2.0, 0.25), 'xticks':None}
    
    cols_to_joint ={r'$v* = 0.3 \pm 0.04$': (2,3, False), r'$v* = 0.43 \pm 0.05$': (0,4, True),
                    r'$v* = 0.55 \pm 0.06$': (1,5, True)}
    
    for num, key in enumerate(cols_to_joint.keys()):
        params.update({'hide_labels': (False, cols_to_joint[key][-1])})
        DJS_instances = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=1.5, fig_size=[3.0,2.5], params=params)
        fig5 = DJS_instances.plot_DJS(all_dfs_instance, 
                            cols=list(cols_to_joint[key][:2]), rows= np.r_[0,2],
                            title="Ankle DJS OvsT group comparison at {}".format(key), 
                            legend=True, reg=df_turn_instance,
                            integration= True, rad = True, header= key)
        if num == 0:
            reg_info_ins = pd.DataFrame(DJS_instances.reg_info_df)
            work_ins = pd.DataFrame(DJS_instances.areas)
        else:
            reg_info_ins = pd.concat([reg_info_ins, DJS_instances.reg_info_df])
            work_ins = pd.concat([work_ins, DJS_instances.areas])
    
    reg_info_ins = reg_info_ins.round(3)
    work_ins = work_ins.round(3)

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

if Age_group:   
    
    # The best are 4 and 13
    # df_turn_age = hyperparams(Fukuchi_age)
    
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
    df_turn_ages = Fukuchi_ages.get_turning_points(turning_points= 6, 
                                smoothing_radius = 4, cluster_radius= 13)
    
    
    Fukuchi_ages.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_ages.deg_to_rad()
    total_work_ages = Fukuchi_ages.total_work() 
    
    
    Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
    params = {'sharex':False, 'sharey':True, 'left_margin': 0.2, 'arr_size':12,
              'hide_labels':(False, True), 'yticks': np.arange(-0.25, 2.25, 0.25), 
              'xticks':None}
    
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
    df_turn_ages_mod = Fukuchi_ages_mod.get_turning_points(turning_points= 6, 
                                smoothing_radius = 4, cluster_radius= 13)
    
    
    Fukuchi_ages_mod.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_ages_mod.deg_to_rad()
    total_work_ages = Fukuchi_ages_mod.total_work() 
    
    
    Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
    params = {'sharex':False, 'sharey':True, 'left_margin': 0.2, 'arr_size':12,
              'hide_labels':(False, True), 'yticks': np.arange(-0.25, 2.25, 0.25), 
              'xticks':None}
    
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
    df_turn_ages_gen = Fukuchi_ages_gen.get_turning_points(turning_points= 6, 
                                param_1 = 4, cluster_radius= 13)
    
    
    Fukuchi_ages_gen.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_ages_gen.deg_to_rad()
    total_work_gen = Fukuchi_ages_gen.total_work() 
    
    
    Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
    params = {'sharex':False, 'sharey':True, 'left_margin': 0.2, 'arr_size':12,
              'hide_labels':(False, True), 'yticks': np.arange(-0.25, 2.25, 0.25), 
              'xticks':None}
    
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
                            title="Ankle DJS OvsT age group comparison at {}".format(key), 
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
    df_turn_gender = Fukuchi_gen.get_turning_points(turning_points= 6, 
                                smoothing_radius = 4, cluster_radius= 13)
    
    
    Fukuchi_gen.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_gen.deg_to_rad()
    total_work_gen = Fukuchi_gen.total_work() 
    
    
    Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
    params = {'sharex':False, 'sharey':True, 'left_margin': 0.2, 'arr_size':12,
              'hide_labels':(False, True), 'yticks': np.arange(-0.25, 2.25, 0.25), 
              'xticks':None}
    
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
                            legend=True, reg=df_turn_gender, header=None,
                            integration= True, rad = True)
        if num == 0:
            reg_info_gen = pd.DataFrame(DJS_gender.reg_info_df)
            work_gen = pd.DataFrame(DJS_gender.areas)
        else:
            reg_info_gen = pd.concat([reg_info_gen, DJS_gender.reg_info_df])
            work_gen = pd.concat([work_gen, DJS_gender.areas])
    reg_info_gen = reg_info_gen.round(3)
    work_gen = work_gen.round(3)