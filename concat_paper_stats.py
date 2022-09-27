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
from itertools import permutations

# stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import seaborn as sns
from scipy.stats.mstats import kruskal
import scikit_posthocs as sp
from statannotations.Annotator import Annotator  # Significant differences

# =============================================================================
# Helper functions
# =============================================================================

print_tables = True
__plots = True
sns.set_context('paper', font_scale=1.5)
sns.set_style("whitegrid")


# =============================================================================
# Class to plot the box plots
# =============================================================================

class obt_box_plot:
    def __init__(self, dep_vars, formal_labels, nrows=3, ncols=3, figsize=(12, 12),
                 wspace=0.3, left_space=0.1, labels_take=np.r_[:4, 5:11]):
        self.ncols = ncols
        self.nrows = nrows
        self.fig1, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        self.fig1.tight_layout()
        self.fig1.subplots_adjust(wspace=wspace, left=left_space)
        self.labels_to_take = labels_take
        self.axes = self.trim_axs(self.axes, len(self.labels_to_take))
        self.ax = None
        self.deps_mod_ = dep_vars[self.labels_to_take]
        self.labels_mod_ = np.array(formal_labels)[self.labels_to_take]
        self.hue_order = [] 

    def plot_boxplot(self, varx,  dataset, xlabel, hue='Speed', order=None,
                     hue_order=["VS", 'S', 'C', 'F', 'VF'], legend_pos=4, rot=0):
        self.hue = hue
        self.hue_order = hue_order
        for num,  self.ax in enumerate(np.ravel(self.axes)):
            if num < (self.nrows * self.ncols) - 1:
                sns.boxplot(x=varx, y=self.deps_mod_[num], hue=self.hue,
                            data=dataset, ax=self.ax, hue_order=hue_order, order=order)
                self.ax.set_ylabel(self.labels_mod_[num])
                if num != legend_pos:
                    self.ax.get_legend().remove()
                else:
                    self.ax.legend(loc='lower right')
                self.ax.set_xlabel(xlabel)
                if rot != 0:
                    plt.setp(self.ax.get_xticklabels(), rotation=rot)
            else:
                continue

        return 
    
    def annotate_(self):
        """
        Work in Progress. 
        To generate automatic stats annotations

        Returns
        -------
        None.

        """
        perm = permutations(self.hue_order)
        annotator = Annotator(self.ax, list(perm))
        annotator.set_pvalues()
        

    def trim_axs(self, axs, N):
        """
        Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
        Copied from pyplot webpage
        """
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    def save_fig(self, fig_name, sup_title=False):
        if sup_title:
            self.fig1.suptitle(
                'Variables with statistical differences in OvsT and speed', fontsize=18)
        self.fig1.savefig(fig_name)
        return self.fig1

# =============================================================================
# Function to do test student test
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
    # Let's perform the Bartetts's test whose Null Hypothesis is that the
    # variances are equal. We will use a significance level of 5.0%, for lower values the null hypothesis is rejected
    # and the variances are not equal

    # Measuring and storing if the samples has the same variance
    var = {item: stats.bartlett(ds1[item],
                                ds2[item]).pvalue for item in dep_vars}
    # Performing the ttest, if not equal it will perform
    ttest_ = {item: stats.ttest_ind(ds1[item], ds2[item],
                                    equal_var=var[item] > 0.05).pvalue for item in dep_vars}
    return var, ttest_


# Testing normal distributions
# For values below 5% the hipothesis is rejected and is non-normal distribution
def shapiro_test(ds, dep_vars, name='No name', df=True):

    if df == True:
        shapiro_ = {item: stats.shapiro(
            ds[item]).pvalue > 0.05 for item in dep_vars}
        shapiro_df = pd.Series(shapiro_, name=name)
        return shapiro_df
    else:
        shapiro_ = {item: stats.shapiro(ds[item]).pvalue for item in dep_vars}
        return shapiro_

# =============================================================================
#    Kruskal Wallis test on ranks
# =============================================================================


def kruskal_groups(ds1, ds2, dep_vars, name):
    kruskal_deps = pd.Series({item: kruskal(ds1[item].values, 
                                              ds2[item].values).pvalue < 0.05 for item in dep_vars})
    kruskal_deps = kruskal_deps.replace(True,1)
    kruskal_deps = kruskal_deps.replace(False,0)
    kruskal_deps.name = name
    return kruskal_deps


os.chdir('ConcatDatasets/')
concat_QS = pd.read_csv('PostProcessedDatasets.csv', index_col=[0])

# Defining the labels to use
dep_vars = concat_QS.columns[10:]
# Unique categories of interest
labels = {col: concat_QS[col].unique() for col in [
    'Mode', 'Speed', 'Origin', 'Gender', 'AgeGroup']}
# Creating the combinations
alllabelNames = sorted(labels)
combinations = it.product(*(labels[Name] for Name in alllabelNames))

# =============================================================================
# Setting individual and specific groups
# =============================================================================

mini_datasets = {}
for comb in combinations:
    name = ''
    for init in comb:
        name += init[0]
    # Acronyms correspond to Agegroup, Gender, Mode, Origin and Speed
    mini_datasets.update({name: concat_QS.query(
        "AgeGroup == '{0}' and Gender == '{1}' and Mode == '{2}' and Origin == '{3}' and Speed == '{4}'".format(*comb))})

# Removing empty datasets
mini_datasets = {k: v for k, v in mini_datasets.items() if v.shape[0] != 0}
# Only overground
mini_ds_over = {k: v for k, v in mini_datasets.items() if k[2] == 'O'}


# Comparing Overground and treadmill
overground_ds = concat_QS.query("Mode == 'Overground'")
treadmill_ds = concat_QS.query("Mode == 'Treadmill'")

# As European has no treadmill, the comparison between datasets needs to be done on overground
european_ds = overground_ds.query("Origin == 'European'")
brazilian_ds = overground_ds.query("Origin == 'South American'")

# Groups by age
children_ds = overground_ds.query("AgeGroup == 'Children'")
younga_ds = overground_ds.query("AgeGroup == 'YoungAdults'")
adults_ds = overground_ds.query("AgeGroup == 'Adults'")
elder_ds = overground_ds.query("AgeGroup == 'Elderly'")

# Gender comparison on overground
male_ds = overground_ds.query("Gender == 'M'")
female_ds = overground_ds.query("Gender == 'F'")

main_groups = [overground_ds, treadmill_ds, european_ds, brazilian_ds,
               children_ds, younga_ds, adults_ds, elder_ds, male_ds, female_ds]

main_labels = ['Overground', 'Treadmill', 'Children', 'Young Adults', 'Adults', 'Elderly',
               'Males', 'Females']

norm_groups = pd.concat([shapiro_test(ds, dep_vars,
                        etiquete) for ds, etiquete in zip(main_groups,
                                                          main_labels)], axis=1)

# =============================================================================
# Kruskal Wallis test
# =============================================================================

# If true, significant differences are detected
kruskal_gen = pd.concat([kruskal_groups(male_ds.query("Speed == '{}'".format(speed)),
                                        female_ds.query(
                                            "Speed == '{}'".format(speed)),
                                        dep_vars, '{}'.format(speed)) for speed in ['S', 'C', 'F']], axis=1)
kruskal_origin = pd.concat([kruskal_groups(european_ds.query("Speed == '{}'".format(speed)),
                                           brazilian_ds.query(
                                               "Speed == '{}'".format(speed)),
                                           dep_vars, '{}'.format(speed)) for speed in ['S', 'C', 'F']], axis=1)

#Only children and young adults are compared
kruskal_age_ch = pd.concat([kruskal_groups(children_ds.query("Speed == '{}'".format(speed)), 
                              younga_ds.query("Speed == '{}'".format(speed)),
                             dep_vars, '{}'.format(speed)) for speed in ['S', 'C', 'F']], axis=1)
kruskal_age_ya = pd.concat([kruskal_groups(younga_ds.query("Speed == '{}'".format(speed)), 
                              adults_ds.query("Speed == '{}'".format(speed)),
                             dep_vars, '{}'.format(speed)) for speed in ['S', 'C', 'F']], axis=1)
kruskal_age_a = pd.concat([kruskal_groups(adults_ds.query("Speed == '{}'".format(speed)), 
                              adults_ds.query("Speed == '{}'".format(speed)),
                             dep_vars, '{}'.format(speed)) for speed in ['S', 'C', 'F']], axis=1)
kruskal_age_e = pd.concat([kruskal_groups(adults_ds.query("Speed == '{}'".format(speed)), 
                              elder_ds.query("Speed == '{}'".format(speed)),
                             dep_vars, '{}'.format(speed)) for speed in ['S', 'C', 'F']], axis=1)
kruskal_mode = pd.concat([kruskal_groups(overground_ds.query("Speed == '{}'".format(speed)), 
                              treadmill_ds.query("Speed == '{}'".format(speed)), 
                             dep_vars, '{}'.format(speed)) for speed in ['VS', 'S', 'C', 'F', 'VF']], axis=1)

# Making fancier columns
m_index_krus_gen = pd.MultiIndex.from_product([['Gender'], ['S', 'C', 'F']])
# m_index_krus_ori = pd.MultiIndex.from_product([['Ethnicity'], ['S', 'C', 'F']])
m_index_krus_mode = pd.MultiIndex.from_product(
    [['Mode'], ['VS', 'S', 'C', 'F', 'VF']])
kruskal_gen.columns = m_index_krus_gen
# kruskal_origin.columns = m_index_krus_ori
kruskal_age_ch.columns = pd.MultiIndex.from_product([['Children'],['S', 'C', 'F']])
kruskal_age_ya.columns = pd.MultiIndex.from_product([['YoungAdults'],['S', 'C', 'F']])
kruskal_age_a.columns = pd.MultiIndex.from_product([['Adults'],['S', 'C', 'F']])
kruskal_age_e.columns = pd.MultiIndex.from_product([['Elderly'],['S', 'C', 'F']])
kruskal_mode.columns = m_index_krus_mode
kruskall_all = pd.concat([kruskal_gen, kruskal_mode], axis=1)
kruskall_age = pd.concat([kruskal_age_ch, kruskal_age_ya,kruskal_age_a, kruskal_age_e], axis=1)


# =============================================================================
# Dunn Posthoc test
# =============================================================================
dunn_all = {}
for label, sub_group in zip(main_labels, main_groups):
    dunn_T = pd.concat([sp.posthoc_dunn(sub_group,
                                        val_col=item,
                                        group_col='Speed',  # Take out adj for Very classes
                                        p_adjust='holm') for item in dep_vars], axis=0)
    dunn_T.index = pd.MultiIndex.from_product(
        [dep_vars, list(dunn_T.index.unique())])
    dunn_Tbool = sp.sign_table(dunn_T)
    dunn_all.update({label: dunn_Tbool})

# =============================================================================
# Main statistic info
# =============================================================================

# Detecting negative CP and LRP
neg_QS = concat_QS.query("CP < 0 ")
neg_LRP = concat_QS.query("LRP < 0 ")
# Detecting how many were cw
cw_samples = concat_QS.query("LoopDirection == 'cw' ")

idx = pd.IndexSlice
decimal = 3
summary = pd.concat([multi_idx(label, rp.summary_cont(group.groupby(group['Speed']),
                                                      decimals=decimal).T, idx=False) for group,
                     label in zip(main_groups, main_labels)], axis=1)
summary = summary.dropna(axis=1)

trials_num = summary.iloc[0, :].astype(np.int64)
trials_num.name = ('', 'N')
# Dropping non independent vars
summary = summary.loc[idx[dep_vars, :], :]
# Dropping N,and SE
summary = summary.loc[idx[:, ['Mean', 'SD', '95% Conf.', 'Interval']], :]
summary = change_labels(summary, ['Mean', 'SD', '95% CI min', '95% CI max'],
                        index=True, level=1)

summary = pd.concat([pd.DataFrame(trials_num).T, summary], axis=0)
# Changing order on level 1 columns
summary = summary.reindex(summary.index.get_level_values(0).unique(),
                          axis=0, level=0)
summary = summary.reindex(['VS', 'S', 'C', 'F', 'VF'], axis=1, level=1)


# Exporting to latex
vel_labels = ['$v*=0.17(0.02)$', '$v*=0.23(0.03)$',
              '$v*=0.3(0.04)$',  '$v*=0.36(0.04)$',
              '$v*=0.43(0.05)$', '$v*=0.49(0.06)$',
              '$v*=0.56(0.07)$', '$v*=0.62(0.07)$',
              '$v*=0.30(0.04)$', '$v*=0.44(0.05)$',
              '$v*=0.55(0.06)$']

with open("table2.tex", "w+") as pt:
    summary.to_latex(buf=pt, col_space=10, longtable=True, multirow=True,
                     caption='Cuantitative ankle DJS characteristics at different population groups' +
                     r' three different gait speeds: Very Slow (VS)[{}], Slow (S)[({})]'.format(vel_labels[0], vel_labels[-3]) +
                     r', Free (C)[{}], Fast (F)[{}] and, Very Fast (VF)[{}]'.format(
                         vel_labels[-2], vel_labels[-1], vel_labels[-4]),
                     label='tab:main_stats_DJS')

groups = [['Overground', 'Treadmill'],
          ['Children', 'Young Adults', 'Adults', 'Elderly'],
          ['Males', 'Females']]

summary = summary.drop(["95% CI min", "95% CI max"], level=1)
groups_label = ["Environment", "Age", "Sex"]

if print_tables:
    for num, subgroup in enumerate(groups):
        sub_summary = summary.loc[:, idx[subgroup, :]]
        with open("table2_{}.tex".format(num), "w+") as pt:
            sub_summary.to_latex(buf=pt, col_space=10, longtable=True, multirow=True,
                                 caption='Cuantitative ankle DJS characteristics at different population groups' +
                                 r' three different gait speeds: Very Slow (VS)[{}], Slow (S)[({})]'.format(vel_labels[0], vel_labels[-3]) +
                                 r', Free (C)[{}], Fast (F)[{}] and, Very Fast (VF)[{}]'.format(
                                     vel_labels[-2], vel_labels[-1], vel_labels[-4]),
                                 label='tab:main_stats_{}'.format(groups_label[num]))

# Building the formal math labels

stiff_labels = ['CP', 'ERP', 'LRP', 'DP']
formal_labels = ['Init {} '.format(
    i)+r'$[\%GC]$' for i in ['ERP', 'LRP', 'DP', 'S', 'TS']]
formal_labels.extend(
    ['Work Absorbed '+r'$\frac{J}{kg}$', 'Net Work '+r'$\frac{J}{kg}$'])
formal_labels.extend(['Stiffness {}'.format(
    stiff)+r'$\frac{Nm}{kg \times rad}$' for stiff in stiff_labels])


if __plots:
    # Plotting the overground vs treadmill boxplots
    mode_class = obt_box_plot(dep_vars, formal_labels,
                              nrows=4, ncols=3, labels_take=np.r_[:11])
    mode_class.plot_boxplot("Mode", dataset=concat_QS, xlabel="Walking mode")
    mode_class.save_fig("stats_diff_mode_speed.pdf")

    # Plotting the Agegroup boxplots
    age_class = obt_box_plot(dep_vars, formal_labels, nrows= 4, ncols = 3, labels_take= np.r_[:11])
    age_class.plot_boxplot("AgeGroup", dataset = concat_QS, xlabel = "Age Groups",
                            order = ["Children","YoungAdults","Adults","Elderly"], rot= 15)
    age_class.save_fig("stats_diff_age_speed.pdf") #Change angle xlabel, try to visualize in a better way

    # Ploting the Gender comparison boxplot
    gender_class = obt_box_plot(dep_vars, formal_labels, nrows= 4, ncols = 3, labels_take= np.r_[:11])
    gender_class.plot_boxplot("Gender", dataset = concat_QS, xlabel = "Sex")
    gender_class.save_fig("stats_diff_gender_speed.pdf") #Change angle xlabel
