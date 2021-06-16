#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday June 7th

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
import itertools as it

#stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import seaborn as sns
from scipy.stats.mstats import kruskal
import scikit_posthocs as sp

# =============================================================================
# Helper functions 
# =============================================================================

def ttest_(ds1, ds2, dep_vars):
    """
    

    Parameters
    ----------
    ds1 : Dataset 1
    ds2 : Dataset 2
    items : items in a dict format

    Returns
    -------
    None.

    """
    # Assumptions:
    #     1. Independent samples
    #     2. Large enough sample size or observations come from a normally-distributed 
    #     population
    #     3. Variances are equal, if not apply weltch test
    
    # Does the samples come from a normally distributed population
    #Let's perform the Bartetts's test whose Null Hypothesis is that the 
    #variances are equal. We will use a significance level of 5.0%, for lower values the null hypothesis is rejected
    #and the variances are not equal
    
    # Measuring and storing if the samples has the same variance
    var = {item: stats.bartlett(ds1[item], 
                          ds2[item]).pvalue for item in dep_vars}
    # Performing the ttest, if not equal it will perform 
    ttest_ = {item: stats.ttest_ind(ds1[item], ds2[item], 
                    equal_var=var[item] > 0.05).pvalue for item in dep_vars}
    return var, ttest_


#Testing normal distributions
#For values below 5% the hipothesis is rejected and is non-normal distribution
def shapiro_test(ds, dep_vars, name='No name', df=True):
    
    if df == True:
        shapiro_ = {item: stats.shapiro(ds[item]).pvalue > 0.05 for item in dep_vars}
        shapiro_df = pd.Series(shapiro_, name=name)
        return shapiro_df
    else:
        shapiro_ = {item: stats.shapiro(ds[item]).pvalue for item in dep_vars}
        return shapiro_



os.chdir('ConcatDatasets/')
concat_QS = pd.read_csv('DatasetPaper.csv', index_col=[0])

#Defining the labels to use
dep_vars = concat_QS.columns[10:]
#Unique categories of interest
labels = {col: concat_QS[col].unique() for col in ['Mode', 'Speed', 'Origin', 'Gender', 'AgeGroup']}
#Creating the combinations
alllabelNames = sorted(labels)
combinations = it.product(*(labels[Name] for Name in alllabelNames))
# =============================================================================
# Setting individual and specific groups
# =============================================================================
mini_datasets ={}
for comb in combinations:
    name = ''
    for init in comb:
        name += init[0]
    #Acronyms correspond to Agegroup, Gender, Mode, Origin and Speed 
    mini_datasets.update({name: concat_QS.query("AgeGroup == '{0}' and Gender == '{1}' and Mode == '{2}' and Origin == '{3}' and Speed == '{4}'".format(*comb))})

# Removing empty datasets
mini_datasets = {k: v for k, v in mini_datasets.items() if v.shape[0] != 0}
mini_ds_over = {k: v for k, v in mini_datasets.items() if k[2] == 'O'} #Only overground

# Eliminating VS and VF as the most of trials are over treadmill
overground_ds = concat_QS.query("Mode == 'Overground'") 
treadmill_ds = concat_QS.query("Mode == 'Treadmill'")

#As European has no treadmill, the comparison between datasets needs to be done on overground
european_ds =  overground_ds.query("Origin == 'European'")
brazilian_ds =  overground_ds.query("Origin == 'South American'") 

#Gender comparison on overground
male_ds = overground_ds.query("Gender == 'M'")
norm_male = shapiro_test(male_ds, dep_vars, 'Male')
female_ds = overground_ds.query("Gender == 'F'")



# =============================================================================
#     ttest student analysis
# =============================================================================


    




