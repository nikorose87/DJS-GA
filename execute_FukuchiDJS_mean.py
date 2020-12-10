#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:47:55 2020
Analysing the DJS of Fukuchi's paper 
@author: nikorose
"""


from DJSFunctions import extract_preprocess_data, ankle_DJS
from plot_dynamics import plot_dynamic, plot_ankle_DJS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============================================================================
# Settings
# =============================================================================
instances = True
Age_group = False
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
    count = 0
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
mean_T = {'mean_'+i:meta_info_valid.query("Type == '{}'".format(i))['Fraude'].mean() for i in unique_type}
mean_T.update({'std_'+i:meta_info_valid.query("Type == '{}'".format(i))['Fraude'].std() for i in unique_type})

if instances:
    
    # =============================================================================
    #     Let us keep only the piece of df and change labels
    # =============================================================================
    Fukuchi_df_modes = Fukuchi_df.loc[:,idx[:,['T03','OS','T05','OC','T07','OF']]]
    # =============================================================================
    # Performing overall average over modes
    # =============================================================================
    
    # Let us filter only the ankle information of the right foot
    Fukuchi_mean_modes = Fukuchi_df.groupby(level=1, axis=1).mean()
    Fukuchi_sd_modes = Fukuchi_df.groupby(level=1, axis=1).std()
    
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
    df_turn_instance = Fukuchi_instance.get_turning_points(turning_points= 6, 
                                smoothing_radius = 4, cluster_radius= 13)
    
    
    Fukuchi_instance.energy_calculation()
    #Sensitive results may vary when integrating degrees, the best is to do in radians
    Fukuchi_instance.deg_to_rad()
    total_work_instance = Fukuchi_instance.total_work()  
    
    # =============================================================================
    # Obtaining the mechanical work through power instances in regular walking 
    # =============================================================================
    
    # Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
    params = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 'color_reg':['black']*20, 
              'color_symbols': ['slategray']*20, 'arr_size': 10, 'left_margin': 0.1}
    
    DJS_fuk_instO = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[1,3],
                          alpha=1.5, fig_size=[2,6], params=params, ext='png')
    
    fig2 = DJS_fuk_instO.plot_DJS(Fukuchi_instance.all_dfs_ankle, 
                        cols=[2,0,1], rows= np.r_[0,2],
                        title="Ankle DJS gait instances Fukuchi dataset overground", 
                        legend=True, reg=df_turn_instance,
                        integration= True, rad = True)
    
    DJS_fuk_instT = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[4,2],
                          alpha=6, fig_size=[10,6], params=params, ext='png')
    
    fig3 = DJS_fuk_instT.plot_DJS(Fukuchi_instance.all_dfs_ankle, 
                        cols=np.r_[3:11], rows= np.r_[0,2],
                        title="Ankle DJS gait instances Fukuchi dataset treadmill", 
                        legend=True, reg=df_turn_instance,
                        integration= True, rad = True)

# =============================================================================
# Plotting dynamic parameters
# =============================================================================
# plot_dyn_gen = plot_dynamic(SD=True, save=True, plt_style='bmh')

# # # Plotting power information

# fig1 = plot_dyn_gen.gait_plot(Fukuchi_instance.all_dfs_ankle, 
#                     cols = None,
#                     rows = None,
#                     title='Ankle Dynamics Right Foot Fukuchi data at instances')  

if Age_group:   
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
    Fukuchi_mean_age.columns = Fukuchi_mean_age.columns.map('{0[0]}_{0[1]}'.format)
    Fukuchi_sd_age = Fukuchi_df.rename(columns= age_dict, level=0).std(level=[0,1], axis=1)
    #Merging column labels
    Fukuchi_sd_age.columns = Fukuchi_sd_age.columns.map('{0[0]}_{0[1]}'.format)
    
    Fukuchi_age =  create_df(Fukuchi_mean_age, Fukuchi_sd_age)
    
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
    params = {'sharex':False, 'sharey':True, 'left_margin': 0.2, 'arr_size':12}
    
    # =============================================================================
    # Plotting QS one by one comparing adults and elderly people
    # =============================================================================
    cols_to_joint ={item: (num, num+11) for num, item in enumerate(meta_info_red['Mode'].unique())}
    
    for key in cols_to_joint.keys():
    
        DJS_comp = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=1.5, fig_size=[3.0,2.5], params=params)
        fig5 = DJS_comp.plot_DJS(Fukuchi_ages.all_dfs_ankle, 
                            cols=list(cols_to_joint[key]), rows= np.r_[0,2],
                            title="Ankle DJS age group comparison at {}".format(key), 
                            legend=True, reg=df_turn_ages,
                            integration= True, rad = True)
    

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
    Fukuchi_mean_gender.columns = Fukuchi_mean_gender.columns.map('{0[0]}_{0[1]}'.format)
    Fukuchi_sd_gender = Fukuchi_df.rename(columns= age_dict, level=0).std(level=[0,1], axis=1)
    #Merging column labels
    Fukuchi_sd_gender.columns = Fukuchi_sd_gender.columns.map('{0[0]}_{0[1]}'.format)
    
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
    params = {'sharex':False, 'sharey':True, 'left_margin': 0.2, 'arr_size':12}
    
    # =============================================================================
    # Plotting QS one by one comparing adults and elderly people
    # =============================================================================
    cols_to_joint ={item: (num, num+11) for num, item in enumerate(meta_info_red['Mode'].unique())}
    
    for key in cols_to_joint.keys():
    
        DJS_comp = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=1.5, fig_size=[3.0,2.5], params=params)
        fig5 = DJS_comp.plot_DJS(Fukuchi_gen.all_dfs_ankle, 
                            cols=list(cols_to_joint[key]), rows= np.r_[0,2],
                            title="Ankle DJS gender group comparison at {}".format(key), 
                            legend=True, reg=df_turn_gender,
                            integration= True, rad = True)