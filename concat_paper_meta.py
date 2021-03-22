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
concat_data.to_csv('DatasetPaper.csv')
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


    