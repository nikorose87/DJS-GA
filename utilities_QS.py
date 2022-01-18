#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:24:20 2020

@author: nikorose
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind_from_stats
from DJSFunctions import plot_ankle_DJS, ankle_DJS
import operator


def change_labels(df_, new_labels, level=0, index=True):
    """
    

    Parameters
    ----------
    new_labels : List
        Replace column or index labels  

    Returns
    -------
    all_dfs_ankle.

    """
    if index:
        idx_old = df_.index.get_level_values(level).unique()
        for num, name in enumerate(new_labels):
            df_.index = df_.index.set_levels(\
                    df_.index.levels[level].str.replace(idx_old[num], name), 
                    level=level)
    else:
        idx_old = df_.columns.get_level_values(level).unique()
        for num, name in enumerate(new_labels):
            df_.columns = df_.columns.set_levels(\
                    df_.columns.levels[level].str.replace(idx_old[num], name), 
                    level=level)
    return df_

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


def ttest(df1, df2, samples=[20,20], name='T-test', 
          method='manual', equal_var=False):
    """
    

    Parameters
    ----------
    df1 : Dataframe No 1, all dfs should have a multilabeled column 
          with -1sd, mean and, +1sd.
    df2 : dataframe No 2 with the mentioned above characteristics

    samples : list with the samples of each dataset
    method : 'manual' for an empiric testing, 'scipy' for using
    ttest_ind_from_stats library.
    
    From the last one we're implementing the Welch's test 
    due to we are not assuming equal population variance.

    Returns
    -------
    t_df : T-test for each GC percent

    """
    std_1 = df1['+1sd']-df1['mean']
    std_2 = df2['+1sd']-df2['mean']
    if method == 'manual':
        diff_means = df1['mean'] - df2['mean']
        t_df = pd.Series(diff_means/(np.sqrt(std_1**2/samples[0]+std_2**2/samples[1])),
                         name=name)
        return t_df
    
    elif method == 'scipy':
        res_tt = {'t_value': [], 'p_value': []}
        for ind in std_1.index:
            stats, p_value = ttest_ind_from_stats(mean1 = df1['mean'][ind], 
                                                  std1 = std_1[ind], 
                                                  nobs1 = samples[0], 
                                                  mean2 = df2['mean'][ind], 
                                                  std2 = std_2[ind], 
                                                  nobs2 = samples[1], 
                                                  equal_var=equal_var)
            res_tt['t_value'].append(stats)
            res_tt['p_value'].append(p_value)
        
        res_tt = pd.DataFrame(res_tt, index=std_1.index)
        res_tt = multi_idx(name, res_tt, idx=False)
        return res_tt

def create_df(df_mean, df_std, idx = False):
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
    _plus = df_mean + df_std / 2
    _minus = df_mean - df_std / 2
    
    #Creating the 2nd level
    _mean = multi_idx('mean', df_mean, level=1, idx=idx)
    _plus = multi_idx('+1sd', _plus, level=1, idx=idx)
    _minus = multi_idx('-1sd', _minus, level=1, idx=idx)
    
    df_concat = pd.concat([_minus, _mean, _plus], axis=1)
    df_concat = df_concat.sort_index(axis=1, level=0)
    
    # Reindexing second level
    df_concat = df_concat.reindex(['-1sd','mean','+1sd'], level=1, axis=1)
    return df_concat

def best_hyper(all_dfs, save=None, TP=[5],
               smooth_radius=range(8,20,2),
               cluster_radius=range(18,24,2), verbose=False, rows=[0,1]):
    """
    To generate the best hyperparameters of the ankle DJS

    Parameters
    ----------
    all_dfs : Dataframe with the ankle DJS information
    save : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    best_df_turn : a Dataframe with the best points found

    """
    sub_levels = len(all_dfs.columns.get_level_values(1).unique())
    idx = pd.IndexSlice     
    df_turn_ = { vel: hyperparams(all_dfs.loc[:,idx[vel,:]], TP = TP,
                                  smooth_radius=smooth_radius, 
                                  c_radius=cluster_radius, R2=True, 
                                  verbose=verbose, rows=rows) for 
                vel in all_dfs.columns.get_level_values(0).unique()} #
    #(6,16) and (12,24) were done to adjust Ch and Y
    #Were are going to select the best hyperparams automatically
    max_p = {}
    best_df_turn = []
    for keys, vals in df_turn_.items():
        ver = False
        #If this is True is because it matches with the shape 
        while ver == False:
            try:
                max_val = max(vals['R2'].items(), key=operator.itemgetter(1))
                ver = vals['TP'][max_val[0]].shape[0] == sub_levels and vals['TP'][max_val[0]].shape[1] == 6
                if ver == True:
                    max_p.update({keys: {'hyperparam': max_val[0], 'R2': max_val[1]}})
                    best_df_turn.append(vals['TP'][max_val[0]])
                else:
                    del vals['R2'][max_val[0]]
            except ValueError:
                # If only 5 points were generated
                print('We could not obtain the parameters in {}, adding nan'.format(vals['TP'][max_val[0]].index))
                best_df_turn.append(vals['TP'][max_val[0]])
                break

    best_df_turn = pd.concat(best_df_turn, axis=0)
    #Filling nan with 0
    best_df_turn = best_df_turn.fillna(0).astype(int)
    if save is not None:
        best_df_turn.to_csv(save)
    return best_df_turn

def hyperparams(df_, TP, smooth_radius, c_radius, features=
                ['Ankle Dorsi/Plantarflexion ', 'Ankle Dorsi/Plantarflexion',
                 'Vertical Force', 'Ankle'], R2=False, verbose=True, rows=[0,1]):
    """
    Parameterization of the curvature settings to see which of them are suitable
    for all the gait instances

    Parameters
    ----------
    df_ : dataframe to analyze the best hyperparameters for Smoothing radious.
    smooth_radius : TYPE, optional
        DESCRIPTION. The default is (4,8).
    c_radius : TYPE, optional
        DESCRIPTION. The default is (10,14).
    features : TYPE, optional
        DESCRIPTION. The default is ['Ankle Dorsi/Plantarflexion ', 'Vertical Force',                 
                                     'Ankle Dorsi/Plantarflexion',  'Ankle'].

    Returns
    -------
    df_turn_instance : Dict with dataframes indicating which regression could be done

    """
    df_turn_instance = {}
    df_turn_r2 = {}
    for tp in TP:
        for i in smooth_radius:
            for j in c_radius:
                try:
                    if verbose:
                        print('For {}, {} values'.format(i,j))
                    _instance = ankle_DJS(df_, features= features)
                    #It needs work
                    _instance.extract_df_QS_data(idx=[0,1]) #df idx=[0,2,1,3]
                    df_turn_instance.update({'sr_{}_cr_{}'.format(i,j): _instance.get_turning_points(rows=rows, 
                                                turning_points= tp, param_1 = i, cluster_radius= j)})
                    if R2:
                        DJS_all = plot_ankle_DJS(SD=False, save=False, plt_style='bmh', sep=False)
                        DJS_all.plot_DJS(df_[df_turn_instance['sr_{}_cr_{}'.format(i,j)].index.values], 
                                            cols=None, rows= np.r_[0,1],
                                            title="Individual", 
                                            legend=True, reg=df_turn_instance['sr_{}_cr_{}'.format(i,j)],
                                            integration= False, rad = True)
                        R2 = DJS_all.reg_info_df['R2'].mean() #Consider to establish weights
                        df_turn_r2.update({'sr_{}_cr_{}'.format(i,j): R2})
                except (ValueError,NameError) as e: #(IndexError, KeyError)
                    if verbose:
                        print('parameters {},{} failed, probably because data indexion or because points are the same'.format(i,j))
                    continue
    return {'TP': df_turn_instance, 'R2': df_turn_r2}