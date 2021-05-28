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





    