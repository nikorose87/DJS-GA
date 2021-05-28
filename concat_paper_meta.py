#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:20:38 2021

@author: enprietop
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
# General task configs
# =============================================================================
wanna_plot = False
wanna_stats = True

def agegroup(age):
    if age <= 18: 
        ageg = 'Children'
    elif age > 18 and age <= 35: 
        ageg = 'YoungAdults'
    elif age > 35 and age <= 54: 
        ageg = 'Adults'
    elif age > 54: 
        ageg = 'Elderly'
    return ageg


Ferra_data = pd.read_csv("Ferrarin2019/meta_info_ferra.csv", index_col=[0,1])
Fukuchi_data = pd.read_csv("Fukuchi/meta_info_Fukuchi.csv", index_col=[0,1])


# =============================================================================
# Modifying Ferrarin
# =============================================================================
indexFerra = 'L-S'+Ferra_data['ID'].apply(lambda x: '{:02d}'.format(int(x)))+'-O-'+\
    Ferra_data['Gender'].apply(lambda x: x[0])+'-'+Ferra_data['Range']
    
Ferra_data['ID'] = indexFerra
Ferra_data['Gender'] = Ferra_data['Gender'].apply(lambda x: x[0])

Ferra_data['ID'] = np.where(Ferra_data['ID'].duplicated(keep=False), 
                      Ferra_data['ID'] + Ferra_data.groupby('ID').cumcount().add(1).astype(str),
                      Ferra_data['ID'])
Ferra_data.index = Ferra_data['ID']

Ferra_data['mode'] = 'Overground'

Ferra_data['AgeGroup'] = Ferra_data['Age'].apply(agegroup)
Ferra_data['Id'] = Ferra_data.index
Ferra_data['Id'] = Ferra_data['Id'].apply(lambda x:x[:5])

Ferra_data_red = Ferra_data[['Id', 'Age', 'AgeGroup', 'Gender', 'Body Height (cm)', 
                             'Body Mass (kg)', 'mode',
                             'Range', 'direction','point 1', 'point 2',
                             'point 3', 'point 4', 'point 5', 'work abs', 'work prod',
                             'CP', 'ERP', 'LRP', 'DP']]

uniform_labels = ['ID','Age','AgeGroup', 'Gender','Height', 'Weight','Mode',
                  'Speed','LoopDirection','initERP', 'initLRP', 'initDP', 
                  'initS', 'initTS', 'WorkAbs', 'WorkNet',
                  'CP', 'ERP', 'LRP', 'DP']
Ferra_data_red.columns = uniform_labels


# =============================================================================
# Modifying  Fukuchi dataset
# =============================================================================

indexFuku = 'F-'+Fukuchi_data.index.get_level_values(0)+'-'+ \
    Fukuchi_data['mode'].apply(lambda x: x[0])+'-'+ Fukuchi_data['Gender']+'-'+ \
        Fukuchi_data['speed']
        

Fukuchi_data['ID'] = indexFuku

Fukuchi_data['ID'] = np.where(Fukuchi_data['ID'].duplicated(keep=False), 
                      Fukuchi_data['ID'] + Fukuchi_data.groupby('ID').cumcount().add(1).astype(str),
                      Fukuchi_data['ID'])
Fukuchi_data.index = Fukuchi_data['ID']

Fukuchi_data['AgeGroup'] = Fukuchi_data['Age'].apply(agegroup)

Fukuchi_data['Id'] = Fukuchi_data.index
Fukuchi_data['Id'] = Fukuchi_data['Id'].apply(lambda x:x[:5])

Fukuchi_data_red = Fukuchi_data[['Id', 'Age', 'AgeGroup', 'Gender', 'Height', 'Mass', 'mode',
                             'speed', 'direction','point 1', 'point 2',
                             'point 3', 'point 4', 'point 5', 'work abs', 'work prod',
                             'CP', 'ERP', 'LRP', 'DP']]

Fukuchi_data_red.columns = uniform_labels


# =============================================================================
# Concatenating datasets
# =============================================================================

concat_data = pd.concat([Fukuchi_data_red, Ferra_data_red])
concat_data.to_csv('ConcatDatasets/DatasetPaper.csv')
#Replacing some values
concat_data['Gender'] = concat_data['Gender'].replace('M', 'Males')
concat_data['Gender'] = concat_data['Gender'].replace('F', 'Females')
vels = ['VS','S','C','F','VF']
groups = ['Children', 'YoungAdults', 'Adults', 'Elderly']
#Ordenating subgroups in a dict
sub_groups_age = {'{}_{}'.format(k1,k2): concat_data.query("AgeGroup == '{}' and Speed == '{}'".format(k1,k2)) for k2 in vels for k1 in groups} 
sub_groups_gen = {'{}_{}'.format(k1,k2): concat_data.query("Gender == '{}' and Speed == '{}'".format(k1,k2)) for k2 in vels for k1 in ['M','F']} 
sub_groups_cond  = {'{}_{}'.format(k1,k2): concat_data.query("Mode == '{}' and Speed == '{}'".format(k1,k2)) for k2 in vels for k1 in ['Overground','Treadmill']} 

Fem_group = concat_data.query("Gender == 'F'")
Male_group = concat_data.query("Gender == 'M'")
#Age groups = 
Age_groups = {key: concat_data.query("AgeGroup == '{}'".format(key)) for key in groups}
num_sub_age = [len(item['ID'].unique()) for item in Age_groups.values()]

#Making formal indexes
labels_samples = [pd.MultiIndex.from_product([['Very Slow', 'Slow', 'Free', 'Fast', 
                'Very Fast'],i]) for i in [groups,['Males','Females'], 
                                         ['Overground', 'Treadmill']]]
                                         
#Making Series if the number of samples
samples_age = pd.Series({key: item.shape[0] for key, item in sub_groups_age.items()})
samples_age.index = labels_samples[0]
samples_gender = pd.Series({key: item.shape[0] for key, item in sub_groups_gen.items()})
samples_gender.index = labels_samples[1]
samples_mode = pd.Series({key: item.shape[0] for key, item in sub_groups_cond.items()})
samples_mode.index = labels_samples[2]
                  
#Concatenating
samples_all = pd.concat([samples_age, samples_gender, samples_mode])
samples_all = samples_all.swaplevel()
samples_all = samples_all.unstack(level=1).reindex(['Children', 'YoungAdults', 
                                                    'Adults', 'Elderly', 'Females', 'Males',
                                                    'Overground', 'Treadmill'])
samples_all = samples_all.reindex(['Very Slow', 'Slow', 'Free', 'Fast', 
                'Very Fast'], axis=1)

samples_all.index = pd.MultiIndex.from_arrays([['Age']*4+['Gender']*2+['Walking Condition']*2, 
                                               list(samples_all.index)])

with open("table_samples.tex", "w+") as pt:
    samples_all.to_latex(buf=pt, col_space=10, longtable=False, multirow=True, 
                        caption=r'Dataset groups by population age, sex, walking '+\
                            'condition and gait speed reporting the number $N$ of '+\
                                r'individuals. Range speeds varies from Very Slow ($v \leq 0.227$)'+\
                                    r',Slow ($0.227 < v* \leq 0.363$), Free ($0.363 < v* \leq 0.500$), '+\
                                        r'Fast ($0.500 < v* \leq  0.636$) and Very Fast ($v* \geq 0.636$)',
                        label='tab:table1')


# =============================================================================
# Analyzing the complete dataset
# =============================================================================

Ferra_dynamics = pd.read_csv("Ferrarin2019/dynamic_data_Lencioni.csv", index_col=[0,1], header=[0,1])
Fukuchi_dynamics = pd.read_csv("Fukuchi/dynamic_data_Fukuchi.csv", index_col=[0,1], header=[0,1])
Fukuchi_dynamics = Fukuchi_dynamics.sort_index(level=[0,1], axis=1)
#Homogenizing the index
index_QS = pd.MultiIndex.from_product([['Ankle Dorsi/Plantarflexion Deg [°]', 
                                        'Ankle Dorsi/Plantarflexion [Nm/kg]'], np.linspace(0,1,303)])
Ferra_dynamics.index = index_QS
Fukuchi_dynamics.index = index_QS

dynamics_concat = pd.concat([Fukuchi_dynamics, Ferra_dynamics], axis=1)

#Making many levels to group
dynamics_concat.columns = pd.MultiIndex.from_arrays([concat_data['ID'], concat_data['AgeGroup'],
                                                     concat_data['Gender'], concat_data['Mode'], 
                                                     concat_data['Speed']])

dynamics_concat.to_csv('ConcatDatasets/dynamics_QS.csv')

# =============================================================================
# Plotting groups DJS plots
# =============================================================================
#General plot configuration 
#Params for all comparatives plots
color_labels = ['blue','red','green','violet','orange','grey','goldenrod']
color_regs = ['dark'+i for i in color_labels]
Colors_tab = [i[1] for i in mcolors.TABLEAU_COLORS.items()]
Color_DJS = [mcolors.CSS4_COLORS[item] for item in color_labels]
Color_reg = [mcolors.CSS4_COLORS[item] for item in color_regs]
params = {'sharex':False, 'sharey':False, 'left_margin': 0.2, 'arr_size':12,
          'yticks': np.arange(-0.25, 1.80, 0.25), 'xticks':None, 
          'color_reg': Color_reg, 'color_symbols': Color_reg, 'color_DJS': Color_DJS,
          'alpha_prod': 0.3, 'alpha_absorb': 0.0, 'DJS_linewidth': 1.5,
          'sd_linewidth': 0.08,'reg_linewidth': 1.0}

times=3
smooth_ = [2,3,4]
cluster_ = range(15*times, 20*times, times)
idx= pd.IndexSlice
if wanna_plot:
    QS_df = {}
    opt = False
    for feat in ['Gender','AgeGroup','Mode']: #
        _QS = dynamics_concat.groupby([feat,'Speed'], axis=1)
        mean_QS = _QS.mean()
        mean_QS.columns = mean_QS.columns.map('{0[0]} {0[1]}'.format)
        std_QS = _QS.std()
        std_QS.columns = std_QS.columns.map('{0[0]} {0[1]}'.format)
        QS_df.update({feat: create_df(mean_QS, std_QS)})
        
        # =============================================================================
        # Setting variables and plotting in individuals
        # =============================================================================
        
        
        _instance = ankle_DJS(QS_df[feat])
        df_instance = _instance.extract_df_QS_data([0,1])
        
        
        #_instance = _instance.change_labels(["Free O", "Fast O", 'Slow O', 'Slow T', 
        #                                 'Free T', 'Fast T'])
    
        if opt == True:
            df_turn_instance = best_hyper(df_instance, save='ConcatDatasets/best_params_instance_{}.csv'.format(feat),
                                      smooth_radius=smooth_,
                                      cluster_radius=cluster_, verbose=False,
                                      rows=[0,1])
        else: 
            df_turn_instance = pd.read_csv('ConcatDatasets/best_params_instance_{}.csv'.format(feat), 
                                           index_col=[0,1])
        #Sensitive results may vary when integrating degrees, the best is to do in radians
        _instance.deg_to_rad()
        total_work_instance = _instance.total_work()  
    
        # # =============================================================================
        # # Obtaining the mechanical work through power instances in regular walking 
        # # =============================================================================
        if feat == 'AgeGroup':
            cols_to_group = [(9,19,4,14, False), (7,17,2,12, True), (5,15,0,10, True), 
                             (6,16,1,11, True), (8,18,3,13, True)]
        else:
            cols_to_group = [(4,9, False), (2,7, True), 
                             (0,5, True), (1,6, True), (3,8, True)]
        cols_to_joint = {r'$v* \leq 0.227$': cols_to_group[0], 
                        r'$0.227 \le v* \leq 0.363$': cols_to_group[1],
                        r'$0.363 \le v* \leq 0.500$': cols_to_group[2],
                        r'$0.500 \le v* \leq 0.636$': cols_to_group[3],
                        r'$v* \ge 0.636$': cols_to_group[4]}
        
        for num, key in enumerate(cols_to_joint.keys()):
            params.update({'hide_labels': (False, cols_to_joint[key][-1])})
            DJS_instances = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                      alpha=1.5, fig_size=[2.2, 2.5], params=params)
            fig = DJS_instances.plot_DJS(df_instance, 
                                cols=list(cols_to_joint[key][:-1]), rows= np.r_[0,1],
                                title="Ankle DJS {} group comparison at {}".format(feat, key), 
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
    
#Perform the LDA Once the plots are done, we are seeing lots of statistical differences.  
