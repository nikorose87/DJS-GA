#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:37:14 2020
Ferrain and horst Together
@author: nikorose
"""
from DJSFunctions import plot_ankle_DJS, ankle_DJS, Plotting
import os
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from utilities_QS import ttest, hyperparams, best_hyper, change_labels
import operator
import seaborn as sns


# Testers

ttest_ = False
plot_theory = False
optimize_params = False
plot_pairs = False
plot_sample_per_step = False
plot_fig_paper = True
plot_fig_graphical_abs = False
plot_per_group = False
plot_children = False
# =============================================================================
# Ferrarin execution 
# =============================================================================

#Excluding not regular intentions
exclude_list = ["{} {}".format(i,j) for i in ['Toe', 'Heel', 'Descending', 
                                            'Ascending'] for j in ['A','Y']]

# exclude_list.extend(['Free A', 'Very Slow A', 'Slow A', 'Medium A', 'Fast A'])

Ferrarin_ = ankle_DJS('mmc3.xls', 
                      dir_loc = 'Ferrarin',
                      exp_name = 'Ferrarin analysis',
                      exclude_names = exclude_list)

all_dfs_ferra = Ferrarin_.extract_DJS_data()
#Changing labels
# all_dfs_ferra = Ferrarin_.change_labels([ r'$0.363 < v* < 0.500$ Ch$_{2}$',
#                                           r"$v/h<0.6$ Ch$_{2}$",
#                                           r'$0.6 < v/h < 0.8$ Ch$_{2}$',
#                                           r'$0.8 < v/h < 1$ Ch$_{2}$',
#                                           r'$v/h > 1.0$ Ch$_{2}$', 
#                                            r'$0.363 < v* < 0.500$ A',
#                                            r'$v/h < 0.6$ A', #
#                                            r'$0.6 < v/h < 0.8$ A',
#                                            r'$0.8 < v/h < 1$ A',
#                                            r'$v/h > 1.0$ A'])

# df_turn_ferra = Ferrarin_.get_turning_points(turning_points= 6, 
#                             param_1 = 4, cluster_radius= 15)
# Ferrarin_.deg_to_rad()
Ferrarin_.energy_calculation()
#Sensitive results may vary when integrating degrees, the best is to do in radians
Ferrarin_.deg_to_rad()
total_work_ferra = Ferrarin_.total_work()

# =============================================================================
# Obtaining the mechanical work through power instances in regular walking Ferra
# =============================================================================
idx= pd.IndexSlice
work_df_ferra = Ferrarin_.power_energy.loc[idx[: , 'mean'], :]
zero_ro_ferra = Ferrarin_.energy_fun.min_max_power(Ferrarin_.power_ankle)

# =============================================================================
# Schwartz execution 
# =============================================================================

Schwartz_ = ankle_DJS('Schwartz.xls', 
                      dir_loc = 'Schwartz',
                      exp_name = 'Schwartz analysis',
                      features= ['Ankle Dorsi/Plantarflexion', 
                                  'Vertical',
                                  'Ankle Dorsi/Plantarflexion',
                                  'Ankle'])

all_dfs_schwartz = Schwartz_.extract_DJS_data()
# all_dfs_schwartz = Schwartz_.change_labels([r'$v* < 0.227$ Ch$_{1}$',r'$0.227 < v* < 0.363$ Ch$_{1}$',r'$0.363 < v* < 0.500$ Ch$_{1}$',
#                                             r'$0.500 < v* < 0.636$ Ch$_{1}$','$v* > 0.636$ Ch$_{1}$'])
# df_turn_schwartz = Schwartz_.get_turning_points(turning_points= 6, 
#                            param_1 = 2, cluster_radius= 8)
# Schwartz_.deg_to_rad()
Schwartz_.energy_calculation()
#Sensitive results may vary when integrating degrees, the best is to do in radians
Schwartz_.deg_to_rad()
total_work_schwartz = Schwartz_.total_work()

# =============================================================================
# Obtaining the mechanical work through power instances in regular walking Ferra
# =============================================================================
work_df_schwartz = Schwartz_.power_energy.loc[idx[: , 'mean'], :]
zero_ro_schwartz = Schwartz_.energy_fun.min_max_power(Schwartz_.power_ankle)


# =============================================================================
# concatenating DFs
# =============================================================================

concat_gait = pd.concat([Ferrarin_.all_dfs_ankle, Schwartz_.all_dfs_ankle], axis=1)
concat_gait = concat_gait.interpolate(axis=0)
concat_gait = concat_gait.reindex(Schwartz_.index_ankle.get_level_values(0).unique(), 
                                  level=0, axis=0)

# =============================================================================
# Obtaining new values for the concatenated df
# =============================================================================
concat_ = ankle_DJS(concat_gait, exp_name = 'Concat Ferrarin and Schwartz analysis')

all_dfs = concat_.extract_df_DJS_data(idx=[0,1,2,3], units=False)
times=3
all_dfs = concat_.interpolate_ankledf(times=times, replace=True)


# =============================================================================
# Best params
# =============================================================================

if optimize_params:
    best_df_turn = best_hyper(all_dfs, save='Ferrarin/best_params_all_dfs.csv',
                             smooth_radius=range(4,7),
                             cluster_radius=range(15*times,20*times, times), 
                             verbose=False, rows=[0,2])
    #Fast A did not do well let us optimize again
    df_turn_FA = hyperparams(all_dfs.loc[:,idx['Fast A',:]], 
                         smooth_radius=range(4,7), 
                         c_radius=range(10*times,15*times, times), R2=True,
                         rows=[0,2])
    df_turn_FA['TP']['sr_4_cr_42'].to_csv('Ferrarin/FastA_opt_params.csv')
    best_df_turn.loc['Fast A'] = df_turn_FA['TP']['sr_4_cr_42']
else:
    best_df_turn = pd.read_csv('Ferrarin/best_params_all_dfs.csv', index_col=[0,1])
    df_turn_FA = pd.read_csv('Ferrarin/FastA_opt_params.csv', index_col=[0,1])
    #Replacing values on best
    best_df_turn.loc['Fast A'] = df_turn_FA




best_df_turnGC = best_df_turn.apply(lambda x: np.int64(x/times))
#Sensitive results may vary when integrating degrees, the best is to do in radians
concat_.deg_to_rad()
total_work = concat_.total_work()   
# =============================================================================
# Plotting ankle Quasi-Stiffness
# =============================================================================
if plot_children:
    Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
    params = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 'color_reg':['black']*20, 
            'color_symbols': ['slategray']*20, 'arr_size': 6, 'left_margin': 0.1, 
            'DJS_linewidth': 0.2, 'reg_linewidth': 1.0, 'grid': False}

    DJS_all_ch = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[3,5],
                        alpha=3, fig_size=[9,11], params=params, ext='png')
    DJS_all_ch.colors = Color
    #Previuos config for Y, A and CH
    #All np.r_[10,1,6,11,2,7,12,0,5,13,3,8,14,4,9]
    # config for more rows np.r_[5,1,6,2,7,0,8,3,9,4]
    # Only Ferra np.r_[1,2,0,3,4,10:15]
    # fig4 = DJS_all_ch.plot_DJS(concat_.all_dfs_ankle, 
    #                     cols=np.r_[10,1,6,11,2,7,12,0,5,13,3,8,14,4,9], rows= np.r_[0,2],
    #                     title="Individual ankle DJS Children", 
    #                     legend=True, reg=best_df_turn.loc[idx[:,'mean'],:],
    #                     integration= True, rad = True)

    # reg_info_concat_all = DJS_all_ch.reg_info_df.round(3)
    # reg_info_concat_all = reg_info_concat_all
    #Read the metrics from this
    metrics_all = reg_info_concat_all.mean(axis=0)

    concat_dep = best_df_turnGC.loc[idx[:,'mean'], 'point 1':].droplevel(1)


    #Work data handling
    work_all = DJS_all_ch.areas
    work_all['direction'] = work_all['direction'].replace('cw',0)
    work_all['direction'] = work_all['direction'].replace('ccw',1)
    work_all = work_all.astype(np.float64).round(3)

    concat_dep = pd.concat([concat_dep, reg_info_concat_all['stiffness'].unstack().droplevel(1), work_all], axis=1)
    labels_idx = ['{} {}'.format(i,j) for i in ['Very Slow', 'Slow', 'Free', 'Medium', 'Fast'] for j in ['C', 'Y', 'A']]
    concat_dep = concat_dep.reindex(labels_idx)
    concat_dep.index = pd.MultiIndex.from_product([['Very Slow', 'Slow', 'Free', 'Medium', 'Very Fast'], ['C', 'Y', 'A']], names=['Gait Speed','Group'])
    concat_dep.columns = pd.MultiIndex.from_arrays([[r'Turning Point [$\%GC$]']*5+[r'Stiffness [$\frac{Nm}{kg\times rad}$]']*4+[r'Work $\frac{J}{kg}$']*3,
                                                    ['ERP','LRP','DP','S','TS','CP','ERP','LRP','DP','Abs.', 'Net.', 'Direction']])
    # =============================================================================
    # Plotting the results of concat_dep
    # =============================================================================
    concat_dep_red = concat_dep.xs(r'Stiffness [$\frac{Nm}{kg\times rad}$]', axis=1)


    with open("Ferrarin/ferra_DJS.tex", "w+") as pt:
        concat_dep.to_latex(buf=pt, col_space=10, longtable=False, multirow=True, 
                            caption=r'Cuantitative ankle DJS characteristics depicted in Fig. \ref{fig:comp_speed_QS_ferra} for children and adult groups',
                            label='tab:table_ferra')
    # =============================================================================
    # Showing the DJS results for youths and adults
    # =============================================================================



    DJS_all_ad = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[2,5],
                        alpha=2, fig_size=[3,7], params=params, ext='png')
    DJS_all_ad.colors = Color
    #Previuos config for Y, A and CH
    #np.r_[10,1,6,11,2,7,12,0,5,13,3,8,14,4,9]
    # config for more rows np.r_[5,1,6,2,7,0,8,3,9,4]
    fig5 = DJS_all_ad.plot_DJS(concat_.all_dfs_ankle, 
                        cols=np.r_[1,2,0,3,4,6,7,5,8,9], rows= np.r_[0,2],
                        title="Individual ankle DJS Y v A", 
                        legend=True, reg=best_df_turn.loc[idx[:,'mean'],:],
                        integration= True, rad = True)

    reg_info_concat_ad = DJS_all_ad.reg_info_df.round(3)

# =============================================================================
# Showing in the regular way, comparing Schwartz and Ferrarin 
# =============================================================================

if plot_pairs:
    Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3
    params_c = {'sharex':False, 'sharey':True, 'left_margin': 0.2, 'arr_size':15,
              'hide_labels':(False, True), 'yticks': np.arange(-0.25, 1.75, 0.25), 
              'xticks':None, 'alpha_absorb': 0.2, 'alpha_prod': 0.4, 'line_width':1}
    
    cols_to_joint ={r'Free': (0,-3, False),
                    r'Very Slow': (1,-5, False),
                    r'Slow': (2,-4, False),
                    r'Fast': (3,-2, False),
                    r'Very Fast': (4,-1, False)}
    
    cols_to_joint_a ={r'Free': (0,5, False),
                    r'Very Slow': (1,6, False),
                    r'Slow': (2,7, False),
                    r'Fast': (3,8, False),
                    r'Very Fast': (4,9, False)}
    
    for num, key in enumerate(cols_to_joint.keys()):
        params_c.update({'hide_labels': (False, cols_to_joint[key][-1])})
        DJS_all = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=2.0, fig_size=[3,3], params=params_c)
        fig6 = DJS_all.plot_DJS(concat_.all_dfs_ankle, 
                            cols=list(cols_to_joint[key][:2]), rows= np.r_[0,2],
                            title="Ankle DJS Y vs CH comparison {}".format(key), 
                            legend=True, reg=best_df_turn.loc[idx[:,'mean'],:],
                            integration= True, rad = True)
        if num == 0:
            reg_info_ch = pd.DataFrame(DJS_all.reg_info_df)
            work_ch = pd.DataFrame(DJS_all.areas)
        else:
            reg_info_ch = pd.concat([reg_info_ch, DJS_all.reg_info_df])
            work_ch = pd.concat([work_ch, DJS_all.areas])
    reg_info_ch = reg_info_ch.round(3)
    work_ch = work_ch.round(3)
    
    for num, key in enumerate(cols_to_joint_a.keys()):
        params_c.update({'hide_labels': (False, cols_to_joint_a[key][-1])})
        DJS_all = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=2.0, fig_size=[3,3], params=params_c)
        fig6 = DJS_all.plot_DJS(concat_.all_dfs_ankle, 
                            cols=list(cols_to_joint_a[key][:2]), rows= np.r_[0,2],
                            title="Ankle DJS Y vs A comparison {}".format(key), 
                            legend=True, reg=best_df_turn.loc[idx[:,'mean'],:],
                            integration= True, rad = True)
        if num == 0:
            reg_info_a = pd.DataFrame(DJS_all.reg_info_df)
            work_a = pd.DataFrame(DJS_all.areas)
        else:
            reg_info_a = pd.concat([reg_info_a, DJS_all.reg_info_df])
            work_a = pd.concat([work_a, DJS_all.areas])
    reg_info_a = reg_info_a.round(3)
    work_a = work_a.round(3)

# =============================================================================
# Plotting separately per study
# =============================================================================
if plot_per_group:
    params_to = {'sharex':False, 'sharey':True, 'left_margin': 0.05, 'arr_size':15,
              'hide_labels':(False, False), 'yticks': np.arange(-0.25, 1.75, 0.25), 
              'xticks':None, 'alpha_absorb': 0.08, 'alpha_prod': 0.3, 'line_width':0.6}
    
    groups = {'Children G2': ([1,0,2,3,4], 'Ferrarin Ch'), 
              'Adults' : (np.r_[6,5,7:10], 'Ferrarin A'),
              'Children G1': (np.r_[10:15], 'Schwartz')}
    
    for key, item in groups.items():
        DJS_g = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[1,5],
                                  alpha=3.0, fig_size=[4,20], params=params_to)
        fig6 = DJS_g.plot_DJS(concat_.all_dfs_ankle, 
                            cols=item[0], rows= np.r_[0,2], #[1,0,2,3,4], np.r_[5:10]
                            title="Ankle DJS CH together {}".format(item[1]), 
                            legend=True, reg=best_df_turn.loc[idx[:,'mean'],:],
                            integration= True, rad = True, header= None)

# =============================================================================
# Plotting one sample with labels, theoretical
# =============================================================================

if plot_sample_per_step: 

    cases = {0: [None, 'None'], 
             1:[best_df_turn.loc[idx[:,'mean'],['point 0', 'point 1', 'point 3']], ['Control Plantar-flexion \n (CPF)']],
             2:[best_df_turn.loc[idx[:,'mean'],['point 1', 'point 3', 'point 4']], ['Dorsi-flexion (DF)']],
             3:[best_df_turn.loc[idx[:,'mean'],['point 1', 'point 3', 'point 4', 'point 5']], ['Dorsi-flexion (DF)', 'Plantar-flexion (PF)']],
             4:[best_df_turn.loc[idx[:,'mean'],['point 0', 'point 1', 'point 3', 'point 4', 'point 5']], ['CPF', 'DF', 'PF']],
             5:[best_df_turn.loc[idx[:,'mean'],:], ['CPF', 'Early RP', 'Late RP', 'DP']]}
    for key, case in cases.items():
        params_sample = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 
                     'color_reg':['black']*20, 'color_symbols': ['slategray']*20, 
                     'arr_size': 13, 'left_margin': 0.25, 'DJS_linewidth': 0.2, 
                     'reg_linewidth': 1.0, 'grid': False, 'alpha_prod': 0.4,
                     'alpha_absorb': 0.1, 'text':True, 'instances':case[1], 
                     'tp_labels' : {'':(1.5,0.6)}}
        DJS_sample = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                              alpha=3.0, fig_size=[4,4], params=params_sample)
        fig6 = DJS_sample.plot_DJS(concat_.all_dfs_ankle, 
                            cols=[-1], rows= np.r_[0,2], #[1,0,2,3,4], np.r_[5:10]
                            title="Ankle DJS sample {}".format(key), 
                            legend=True, reg=case[0],
                            integration= True, rad = True, header= None)

if plot_fig_graphical_abs: 

    cases = {"grahicalAbstract":[best_df_turn.loc[idx[:,'mean'],:], 
                                 ['Controlled Plantar-Flexion (CP)', 
                                  'Early Response Phase (ERP)', 'Late Response Phase (LRP)', 
                                  'Descending Phase (DP)']]}
    for key, case in cases.items():
        params_sample = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 
                     'color_reg':['black']*20, 'color_symbols': ['slategray']*20, 
                     'arr_size': 13, 'left_margin': 0.25, 'DJS_linewidth': 0.2, 
                     'reg_linewidth': 1.5, "sd_linewidth": 0.0, 'grid': False, 'alpha_prod': 0.4,
                     'alpha_absorb': 0.1, 'text':True, 'instances':case[1], 
                     'tp_labels' : {'':(1.5,0.6)}}
        DJS_sample = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                              alpha=3.0, fig_size=[3,3], params=params_sample)
        fig6 = DJS_sample.plot_DJS(concat_.all_dfs_ankle, 
                            cols=[-3], rows= np.r_[0,2], #[1,0,2,3,4], np.r_[5:10]
                            title="Ankle DJS sample {}".format(key), 
                            legend="sep", reg=case[0],
                            integration= True, rad = True, header= None)

# =============================================================================
# Plotting one sample with labels, theoretical
# =============================================================================

if plot_fig_paper: 

    cases = {5:[best_df_turn.loc[idx[:,'mean'],:], ['CP', 'ERP', 'LRP', 'DP']]}
    for key, case in cases.items():
        params_sample = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 
                     'color_reg':['black']*20, 'color_symbols': ['slategray']*20, 
                     'arr_size': 13, 'left_margin': 0.25, 'DJS_linewidth': 0.2, 
                     'reg_linewidth': 1.0, 'grid': False, 'alpha_prod': 0.4,
                     'alpha_absorb': 0.1, 'text':True, 'instances':case[1], 
                     'tp_labels' : {'a.':(1.4, 1.4), 'b.':(1.6,1.3), 'c.':(1.2,1.0),
                                    'd.':(1,1.14),'e.':(1.15,1.1)}}
        DJS_sample = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                              alpha=3.0, fig_size=[5,5], params=params_sample)
        fig6 = DJS_sample.plot_DJS(concat_.all_dfs_ankle,
                            cols=[-2], rows= np.r_[0,2], #[1,0,2,3,4], np.r_[5:10]
                            title="Ankle DJS sample {}".format(key), 
                            legend=True, reg=case[0],
                            integration= True, rad = True, header= None)

    #Only regressions
    
    params_simple = {'sharex':False, 'sharey':True, 'color_DJS':['white']*20, 
                     'color_reg':['black']*20, 'color_symbols': ['white']*20, 
                     'arr_size': 13, 'left_margin': 0.15, 'DJS_linewidth': 0.2, 
                     'reg_linewidth': 1.0, 'grid': False, 'alpha_prod': 0.0,
                     'alpha_absorb': 0.0, 'text':False, 'hide_labels': (True,True)}
    
    # DJS_simple = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
    #                           alpha=2.0, fig_size=[2,2], params=params_simple)
    # fig6 = DJS_simple.plot_DJS(concat_.all_dfs_ankle, 
    #                     cols=[-1], rows= np.r_[0,2], #[1,0,2,3,4], np.r_[5:10]
    #                     title="Ankle DJS sample dir", 
    #                     legend=False, reg=best_df_turn.loc[idx[:,'mean'],:],
    #                     integration= True, rad = True, header= None)
    
if ttest_:
    # =============================================================================
    # Obtaining the ttest of children (Schwartz) against youth (Ferrarin)
    # =============================================================================
    
    cols_ch = best_df_turn.index.get_level_values(0).unique()
    cols_ch = cols_ch[np.r_[1,2,0,3,4,10:15]]
    etiquete = ['VS','S','C','F','VF']
    #Dropping GRF and powers, we are interested in QS only
    df_QS_ch = all_dfs
    df_QS_ch = df_QS_ch.drop(['Vertical  Force [%BH]', 'Ankle  [W/kg]'], axis=0, level=0)
    
    #Samples of each experiment
    n_schwartz = [77, 82, 82, 76, 51] #Very Slow Slow Free Fast Very Fast
    n_ferra_y = [34, 76, 111, 71, 100, 83, 51, 75, 67] #XS S Natural M L T H A D
    n_ferra_a = [140, 110, 124, 68, 52, 124, 85, 73, 72] #XS S Natural M L T H A D
    
    tt_ch = pd.concat([ttest(df_QS_ch[cols_ch[i]],
                             df_QS_ch[cols_ch[i+5]], 
                             samples=[n_schwartz[i], n_ferra_y[i]], 
                             name='Ttest_{}'.format(etiquete[i]), method='scipy') for i in range(5)], axis=1)
    
    tt_angles_ch = tt_ch.loc['Ankle Dorsi/Plantarflexion  Deg [°]'].mean(axis=0)
    tt_moments_ch = tt_ch.loc['Ankle Dorsi/Plantarflexion  [Nm/kg]'].mean(axis=0)
    
    
    # =============================================================================
    # Obtaining the ttest of adults vs youth (Ferrarin)
    # =============================================================================
    
    cols_ad = best_df_turn.index.get_level_values(0).unique()
    cols_ad = cols_ad[np.r_[1,2,0,3,4,6,7,5,8,9]]
    etiquete = ['VS','S','C','F','VF']
    #Dropping GRF and powers, we are interested in QS only
    df_QS_ad = all_dfs
    df_QS_ad = df_QS_ad.drop(['Vertical  Force [%BH]', 'Ankle  [W/kg]'], axis=0, level=0)
    
    tt_ad = pd.concat([ttest(df_QS_ad[cols_ad[i]],
                             df_QS_ad[cols_ad[i+5]], 
                             samples= [n_ferra_y[i], n_ferra_a[i]], 
                             name='Ttest_{}'.format(etiquete[i]), 
                             method='scipy') for i in range(5)], axis=1)
    
    tt_angles_ad = tt_ad.loc['Ankle Dorsi/Plantarflexion  Deg [°]'].mean(axis=0)
    tt_moments_ad = tt_ad.loc['Ankle Dorsi/Plantarflexion  [Nm/kg]'].mean(axis=0)
    
    tt_concat = pd.concat([tt_angles_ch, tt_moments_ch, tt_angles_ad, tt_moments_ad], axis=1)
    tt_concat = tt_concat.round(3)
    tt_concat.index = pd.MultiIndex.from_product([['Very Slow', 'Slow', 'Free', 'Medium', 'Very Fast'],['t-value', 'p-value']])
    tt_concat.columns = pd.MultiIndex.from_product([['C vs Y', 'Y vs A'],['Ankle angle', 'Ankle moment']])
    with open("Ferrarin/stats_comp.tex", "w+") as pt:
        tt_concat.to_latex(buf=pt, col_space=10, longtable=False, multirow=True, 
                            caption=r'Statistical Weltch t-test and their respective t-value for the independent variables of the mentioned groups.',
                            label='tab:ferra_stats')
    

#Conclusion
#Significant differences were found in terms of angles, whereas for ankle moment no significant 
#differences were found with p value < 0.05

#The statistic conclusion according with
#https://stats.stackexchange.com/questions/339243/r-welch-two-sample-t-test-t-test-interpretation-help


# Plotting the power work which is lost
if plot_theory:
    plot_power = Plotting()
    plot_power.plot_power_and_work(all_dfs, 'Free Y')
    plot_power.regular_plot(all_dfs, 'Ankle Dorsi/Plantarflexion  Deg [°]', 'Free Y', 'Ankle angle', sd=True)
    plot_power.regular_plot(all_dfs, 'Ankle Dorsi/Plantarflexion  [Nm/kg]', 'Free Y', 'Ankle moment', sd=True)
