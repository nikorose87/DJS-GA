#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:04:57 2021

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

sns.set_context('paper', font_scale=1.5)
sns.set_style("whitegrid")

#plots 
all_plot = False
plot_bechmark = True
plot_sample = False
box_plot = False

def label_(sub_ID, lat, df_):
    try:
        label_turn = pd.MultiIndex.from_arrays([['TF{:02d}'.format(sub_ID)]*df_.shape[0], 
                                     [lat]*df_.shape[0],list(df_.index.get_level_values(0)),
                                     list(df_.index.get_level_values(1))])
    except IndexError:
        label_turn = pd.MultiIndex.from_arrays([['TF{:02d}'.format(sub_ID)]*df_.shape[0], 
                             [lat]*df_.shape[0],list(df_.index.get_level_values(0).unique())])
    return label_turn

meta_data = pd.read_excel('Hood/Subject Information.xlsx', skiprows=np.r_[0,20:30], usecols=np.r_[1:15], index_col=[0])

idx = pd.IndexSlice

Color = [i[1] for i in mcolors.TABLEAU_COLORS.items()]*3

params_mono = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 
                 'color_reg':['black']*20, 'color_symbols': ['slategray']*20, 
                 'arr_size': 20, 'left_margin': 0.1, 'DJS_linewidth': 0.2, 
                 'reg_linewidth': 1.0, 'grid': True, 'alpha_prod': 0.4,
                 'alpha_absorb': 0.1, 'text':False}

params_all = {'sharex':False, 'sharey':True, 'left_margin': 0.30, 'arr_size':5,
          'hide_labels':(False, False), 'yticks': np.arange(-0.25, 1.755, 0.5), 
          'xticks':np.arange(-0.2, 0.3, 0.1), 'alpha_prod':0.0, 'alpha_absorb':0.0, 'line_width': 1.0,
          'grid':False}

params_ind = {'sharex':False, 'sharey':True, 'left_margin': 0.05, 'arr_size':5,
          'hide_labels':(True, True), 'yticks': np.arange(-0.2, 2.30, 0.6), 
          'xticks':None, 'alpha_prod':0.3, 'alpha_absorb':0.1, 
          'line_width': 0.4,
          'grid':True}

count = 0

if all_plot:
    op_ = 'load'
    for sub_ID in np.r_[1,5:21]: #np.r_[1,2,5:21]
        sub_df = pd.read_csv('Hood/Hood_TF{:02d}.csv'.format(sub_ID), index_col=[0,1], header=[0,1,2,3])
        for lat in ['ipsilateral', 'contralateral']: #
            sub_df_ = sub_df.loc[:,idx[:, lat,:,:]]
            sub_df_ = sub_df_.droplevel(level=[0,1], axis=1)
            Hood_sub1 = ankle_DJS(sub_df_, 
                                  dir_loc = 'Hood',
                                  exp_name = 'Above knee amputation analysis')
            
            ipsi_QS = Hood_sub1.extract_df_QS_data(idx=[0,1])
            ipsi_QS = Hood_sub1.invert_sign(idx=1)
            
            #Trial and error hyperparameters
            if op_ == 'manual':
                if lat == 'ipsilateral':
                    tp, sr, cr = [4, 4, 100]
                else:
                    tp, sr, cr = [5, 4, 150]
                
                df_turn = Hood_sub1.get_turning_points(rows=[0,1], turning_points=tp, param_1=sr,
                                                        cluster_radius=cr)
            elif op_ == 'opt':
                df_turn = best_hyper(ipsi_QS, save='Hood/turn_params_{}_TF{:02d}.csv'.format(lat, sub_ID), 
                           TP = [4 if lat == 'ipsilateral' else 5],
                           smooth_radius=range(1,10,2),
                           cluster_radius=range(100,200,25), verbose=True, rows=[0,1])
            elif op_ == 'load':
                df_turn = pd.read_csv('Hood/turn_params_{}_TF{:02d}.csv'.format(lat, sub_ID), 
                                      index_col=[0,1])
                
            
            DJS_vis = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=[1,5],
                                      alpha=1.5, fig_size=[2,6], params=params_ind)
            try:
                fig_vis = DJS_vis.plot_DJS(Hood_sub1.all_dfs_QS, 
                                    cols=None, rows= np.r_[0,1],
                                    title="Ankle DJS amputee {} TF{:02d}".format(lat, sub_ID), 
                                    legend=True, reg= df_turn.loc[idx[:,'mean'],:],
                                    integration= True, rad = True)
            
                df_turn.index = label_(sub_ID, lat, df_turn)
                #Saving work
                area_ = DJS_vis.areas
                area_.index = label_(sub_ID, lat, area_)
                #Regressions
                reg_info = DJS_vis.reg_info_df
                reg_info = reg_info.droplevel(1, axis=0)
                label_turn_reg = label_(sub_ID, lat, reg_info)
                reg_info.index = label_turn_reg
                
                if count == 0:
                    df_turn_all = df_turn
                    work_amp = area_
                    reg_info_all = reg_info
                    count=1
                else:
                    df_turn_all = pd.concat([df_turn_all, df_turn], axis=0)
                    work_amp = pd.concat([work_amp, area_], axis=0)
                    reg_info_all = pd.concat([reg_info_all, reg_info], axis=0)
            except IndexError:
                continue
            
            df_turn_all.to_csv('Hood/Hood_TP.csv')
            work_amp.to_csv('Hood/hood_work_info.csv')
            reg_info_all.to_csv('Hood/Hood_regressions.csv')
else:
    df_turn_all = pd.read_csv('Hood/Hood_TP.csv', header=[0], index_col=[0,1,2,3])
    work_amp = pd.read_csv('Hood/hood_work_info.csv', header=[0], index_col=[0,1,2])
    reg_info_all = pd.read_csv('Hood/Hood_regressions.csv', header=[0], index_col=[0,1,2,3])

#Changing labels in ipsilateral in second and third variables
ipsi_reg_df = reg_info_all.loc[idx[:,'ipsilateral',:,:],:]
ipsi_reg_df =  change_labels(ipsi_reg_df, ['CP','CDF', 'PP'], level=3, index=True)
contra_reg_df = reg_info_all.loc[idx[:,'contralateral',:,:],:]
    
# Main statistics about regression
results = {}
dep_vars = {}
vel_label_ranges = [r'$v* < 0.227$',r'$0.227 < v* < 0.363$',r'$0.363 < v* < 0.500$',
                    r'$0.500 < v* < 0.636$', r'$v* > 0.636$']
def define_range_vel(x):
    if x <= 0.227:
        cel = r'$v* < 0.227$'
    elif 0.227 <= x <= 0.363:
        cel = r'$0.227 < v* < 0.363$'
    elif 0.363 <= x <= 0.500:
        cel = r'$0.363 < v* < 0.500$'
    elif 0.500 <= x <= 0.636:
        cel = r'$0.500 < v* < 0.636$'
    elif x >= 0.636:
        cel = r'$v* > 0.636$'
    
    return cel
        
    
    return
for et, limb in [('ipsilateral', ipsi_reg_df/10), ('contralateral', contra_reg_df)]:
    results.update({et+' summary': rp.summary_cont(limb, decimals=2)})
    reg_res = limb.unstack(level=3)
    # merging into one level
    reg_res.columns =  reg_res.columns.map('{0[0]} {0[1]}'.format)
    #Adding work
    reg_res = pd.concat([reg_res, work_amp.loc[idx[:,et,:],:]], axis=1)
    #Adding points and scaling to 100%
    reg_res = pd.concat([reg_res, df_turn_all.loc[idx[:,et,:,'mean'],:].droplevel(3)/10], axis=1)
    reg_res = reg_res.dropna(axis=1, thresh= 30)
    reg_res = reg_res.dropna(axis=0, how='any')
    #Making vel as column but no level
    reg_res['velocity'] = reg_res.index.get_level_values(2)  
    reg_res['Range'] = reg_res['velocity'].apply(define_range_vel)
    for sub in reg_res.index.get_level_values(0).unique():
        reg_res.loc[idx[sub,:,:],'subject'] = sub
        for col in meta_data.columns:
            reg_res.loc[idx[sub,:,:], col] = meta_data.loc[sub,col]
    if et == 'ipsilateral':
        reg_res_min = reg_res[reg_res.columns[np.r_[3:6,15:18,19:38]]]
        reg_res_min = reg_res_min.reindex(tuple(reg_res_min.columns[np.r_[10:24,5,0:5,6:10]]), axis=1)
        reg_res_min.columns = np.hstack((list(reg_res_min.columns[:-4]),
                                         ['CDF init','DP init','S init','TS init']))
        #Dep vars labels
        dep_vars.update({'ipsilateral': [r'CP $\frac{Nm}{kg\times rad}$', r'CDF $\frac{Nm}{kg\times rad}$',
                                           r'PP $\frac{Nm}{kg\times rad}$', r'Work Abs $\frac{J}{kg}$', 
                                           r'Work Net $\frac{J}{kg}$','init CDF', 'init PP','init S','init TS']})
        #dep_vars to show in plots
    else: 
        reg_res_min = reg_res[reg_res.columns[np.r_[4:8,20,21,22,24:44]]]
        reg_res_min = reg_res_min.reindex(tuple(reg_res_min.columns[np.r_[12:26,6,0,2,3,1,4,5,7:12]]), axis=1)
        reg_res_min.columns = np.hstack((list(reg_res_min.columns[:-5]),
                                         ['ERP init','LRP init','DP init','S init','TS init']))
        dep_vars.update({'contralateral': [r'CP $\frac{Nm}{kg\times rad}$', r'ERP $\frac{Nm}{kg\times rad}$',
                                           r'LRP $\frac{Nm}{kg\times rad}$',r'DP $\frac{Nm}{kg\times rad}$',
                                           r'Work Abs $\frac{J}{kg}$', r'Work Net $\frac{J}{kg}$',
                                           'init ERP', 'init LRP','init DP','init S','init TS']})
    results.update({et+' results all': reg_res})
    results.update({et+' results': reg_res_min})
    
    
#Plotting brands QS
if plot_sample:
    params_sample = {'sharex':False, 'sharey':True, 'color_DJS':['slategray']*20, 
                     'color_reg':['black']*20, 'color_symbols': ['slategray']*20, 
                     'arr_size': 13, 'left_margin': 0.15, 'DJS_linewidth': 0.2, 
                     'reg_linewidth': 1.0, 'grid': False, 'alpha_prod': 0.4,
                     'alpha_absorb': 0.1, 'text':True, 'tp_labels': {'I.C.':(3,3),'CDF':(1,1),
                     'PP':(1,1),'S':(1,1), ' ':(1,1)}, 'instances': ['CP','CDF', 'PP','S']}
    sub = 14
    for lat in ['ipsilateral', 'contralateral']:
        sub_df = pd.read_csv('Hood/Hood_TF{:02d}.csv'.format(sub), index_col=[0,1], header=[0,1,2,3])
        sub_df_ = sub_df.loc[:,idx[:, lat,:,:]]
        sub_df_ = sub_df_.droplevel(level=[0,1], axis=1)
        Hood_sub1 = ankle_DJS(sub_df_, 
                              dir_loc = 'Hood',
                              exp_name = 'Above knee amputation analysis')
        ipsi_QS = Hood_sub1.extract_df_QS_data(idx=[0,1])
        ipsi_QS = Hood_sub1.invert_sign(idx=1)
        if lat == 'contralateral':
            params_sample.update({'tp_labels' : {'I.C.':(1,1),'ERP':(1,1),
                     'LRP':(1.0,1.0),'DP':(1,1.1),'S':(1.0,1.0), 
                     ' ':(1,1)}, 'instances': ['CP','ERP', 'LRP', 'DP', 'S']})
        DJS_sample = plot_ankle_DJS(SD=True, save=True, plt_style='bmh', sep=False,
                                  alpha=3.0, fig_size=[4,4], params=params_sample)
        fig_sample = DJS_sample.plot_DJS(Hood_sub1.all_dfs_QS, 
                            cols=np.r_[3], rows= np.r_[0,1], #[1,0,2,3,4], np.r_[5:10]
                            title="amputee_convention_{}".format(lat), 
                            legend=True, 
                            reg=df_turn_all.loc[idx['TF{:02d}'.format(sub), lat,:,'mean'],:].dropna(axis=1),
                            integration= True, rad = True, header= None)

series_brands = meta_data['Foot Prosthesis']
brands = series_brands.value_counts()

# load specific subjects
if plot_bechmark:
    for sub_ID in np.r_[5,6,8,11,13,14,17,18,20]: #np.r_[5,6,8,11,13,14,17,18,20]
        sub_df = pd.read_csv('Hood/Hood_TF{:02d}.csv'.format(sub_ID), index_col=[0,1], header=[0,1,2,3])
        for lat in ['ipsilateral', 'contralateral']: #
            sub_df_ = sub_df.loc[:,idx[:, lat,:,:]]
            sub_df_ = sub_df_.droplevel(level=[0,1], axis=1)
            Hood_sub1 = ankle_DJS(sub_df_, 
                                  dir_loc = 'Hood',
                                  exp_name = 'Above knee amputation analysis')
            
            ipsi_QS = Hood_sub1.extract_df_QS_data(idx=[0,1])
            ipsi_QS = Hood_sub1.invert_sign(idx=1)
            
            # For plotting speeds
            if lat == 'ipsilateral':
                DJS_all = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=False,
                                          alpha=1.5, fig_size=[1,1.5], params=params_all)
                try:
                    fig_all = DJS_all.plot_DJS(Hood_sub1.all_dfs_QS, 
                                        cols=None, rows= np.r_[0,1],
                                        title="Ankle DJS amputee {} TF{:02d} concat".format(lat, sub_ID), 
                                        legend=True, reg= False, #df_turn_all.loc[idx['TF{:02d}'.format(sub_ID),lat,:,'mean'],:].dropna(axis=1),
                                        integration= True, rad = True)
                except IndexError:
                    continue
            DJS_limb = plot_ankle_DJS(SD=False, save=True, plt_style='bmh', sep=False,
                                      alpha=1.5, fig_size=[1,1.2], params=params_ind)
            
            try:
                fig_limb = DJS_limb.plot_DJS(Hood_sub1.all_dfs_QS, 
                                    cols=[4 if meta_data.loc['TF{:02d}'.format(sub_ID)]['K-Level'] else 1], rows= np.r_[0,1],
                                    title="Ankle DJS amputee {} TF{:02d} ind".format(lat, sub_ID), 
                                    legend= False, 
                                    reg=df_turn_all.loc[idx['TF{:02d}'.format(sub_ID),lat,:,'mean'],:].dropna(axis=1).astype(np.int64),
                                    integration= True, rad = True)
            except IndexError:
                    continue
#Reading natural references from Fukucho
old_labels_cl = results['contralateral results'].columns[-11:]
old_labels_il = results['ipsilateral results'].columns[-9:]
fukuchi_ref = pd.read_csv('Fukuchi/summary_concat.csv', index_col=[0,1], header=[0,1])

treadmill_ref_cl = fukuchi_ref.loc[idx[:,['95% CI min', '95% CI max']],'Treadmill']
labels_over = treadmill_ref_cl.index.get_level_values(0).unique()
#Creating a reference table for contralateral
treadmill_ref_cl = treadmill_ref_cl.reindex(labels_over[np.r_[7:11,5,6,0:5]], level=0)
treadmill_ref_cl = change_labels(treadmill_ref_cl, old_labels_cl, level=0)
#Creating for ipsilateral
treadmill_ref_il = pd.concat([treadmill_ref_cl, multi_idx('stiffness CDF', 
                    (treadmill_ref_cl.loc['stiffness ERP']+treadmill_ref_cl.loc['stiffness LRP'])/2)], axis=0)
treadmill_ref_il = treadmill_ref_il.drop(['stiffness ERP', 'stiffness LRP', 'LRP init'])
labels_over_il = treadmill_ref_il.index.get_level_values(0).unique()
treadmill_ref_il = treadmill_ref_il.reindex(labels_over_il[np.r_[0,8,1:8]], level=0)
treadmill_ref_il = change_labels(treadmill_ref_il, old_labels_il, level=0)
#Plotting statistics
#ipsilateral
if box_plot:
    for comp_, label, lat in zip([treadmill_ref_il, treadmill_ref_cl], 
                                 [old_labels_il, old_labels_cl],
                                 ['ipsilateral','contralateral']): #
        fig1, axes = plt.subplots([8 if lat == 'ipsilateral' else 10][0],1, figsize = (10,16))
        for num_i, ax in enumerate(np.ravel(axes)):
            dep_var = dep_vars[lat]
            # order = sorted(list(results['{} results'.format(lat)]['velocity'].unique())) order for velocity
            #Filling regular values interval
            # ax.fill_between(np.arange(-1,10), comp_.loc[idx[dep_var[num_i],'95% CI min'], 'Fast'],
            #                 comp_.loc[idx[dep_var[num_i],'95% CI max'], 'Fast'], 
            #                 color='lightsalmon', label='non pathological fast')
            ax.fill_between(np.arange(-1,10), comp_.loc[idx[label[num_i],'95% CI min'], 'Free'],
                            comp_.loc[idx[label[num_i],'95% CI max'], 'Free'], 
                            color='rosybrown', label='non pathological free')
            ax.fill_between(np.arange(-1,10), comp_.loc[idx[label[num_i],'95% CI min'], 'Slow'],
                            comp_.loc[idx[label[num_i],'95% CI max'], 'Slow'], 
                            color='khaki', label='non pathological slow')
            sns.barplot(x='Foot Prosthesis', y=label[num_i], hue='Range', 
                        data=results['{} results'.format(lat)], palette="rocket", 
                        ax=ax, hue_order=vel_label_ranges[:3])
            ax.legend().set_visible(False)
    
            if lat == 'ipsilateral' and num_i != 7 or lat == 'contralateral' and num_i != 9:
                ax.set(xticklabels=[])
            ax.set(xlabel='')
            ax.set_ylabel(ylabel = dep_var[num_i])
        plt.xticks(rotation=20)
        # plt.subplots_adjust(left=0.1, bottom=0.3)
        handles, labels = ax.get_legend_handles_labels()
        fig1.legend(handles, labels, loc='upper center', ncol=2, 
                    bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
        plt.tight_layout(h_pad=1.5, rect=[0,0.18,1,1])
        # fig6.suptitle('Variables with statistical differences in gender', fontsize = 18)
        fig1.savefig('Hood/vars_bars_in_prosthesis_{}.png'.format(lat), bbox_inches="tight")


# Table for statistics
for lat in ['ipsilateral', 'contralateral']:
    data_ = results['{} results'.format(lat)]
    data_.columns = list(data_.columns[:[-9 if lat == 'ipsilateral' else -11][0]])+(dep_vars[lat])
    decimal=3
    summary_= multi_idx(lat, rp.summary_cont(data_.groupby(['Range']), 
                                                        decimals=decimal).T, idx=False)
    trials_num = summary_.iloc[0,:].astype(np.int64)
    trials_num.name = ('','N')
    summary_ = summary_.loc[idx[dep_vars[lat][:[5 if lat == 'ipsilateral' else 6][0]],
                                        ['Mean', 'SD','95% Conf.', 'Interval']],:]
    summary_ = change_labels(summary_, ['Mean', 'SD','95% CI min', '95% CI max'], 
                                           index=True, level=1)
    summary_ = pd.concat([pd.DataFrame(trials_num).T, summary_], axis=0)
    #Changing order on level 1 columns
    summary_ = summary_.reindex(vel_label_ranges[:3], axis=1, level=1)
    with open("Hood/table_stats_{}.tex".format(lat), "w+") as pt:
        summary_.to_latex(buf=pt, col_space=10, longtable=True, multirow=True, 
                        caption='Cuantitative mechanical ankle DJS characteristics for {} group'.format(lat),
                        label='tab:stats_{}'.format(lat))


#Exporting to chapter two
fukuchi_ref =  fukuchi_ref.drop(['point 1',   'point 2',   'point 3',   'point 4',
         'point 5'], axis=0)
fukuchi_ref = fukuchi_ref.reindex([np.nan, 'CP', 'ERP', 'LRP', 'DP', 'work abs', 'work prod'], axis=0, level=0)
fukuchi_ref = fukuchi_ref.loc[:,idx['Treadmill',:]]
fukuchi_ref.index = summary_.index
fukuchi_ref_contra = pd.concat([summary_, fukuchi_ref], axis=1)
old_idex_order =  np.array(fukuchi_ref_contra.columns)[np.r_[0,3,1,4,2,5]]
fukuchi_ref_contra = fukuchi_ref_contra.reindex(old_idex_order, axis=1)
new_index_fuk = pd.MultiIndex.from_product([['$v* < 0.227$', '$0.227 < v* < 0.363$', 
                                              '$0.363 < v* < 0.500$'],['Contralateral', 'Able-bodied']])
fukuchi_ref_contra.columns = new_index_fuk

with open("Hood/table_stats_contra.tex", "w+") as pt:
    fukuchi_ref_contra.to_latex(buf=pt, col_space=10, longtable=True, multirow=True, 
                    caption='Cuantitative mechanical ankle DJS characteristics for contralateral group'+\
                        'and their comparison against able-bodied groups at same gait speed.',
                    label='tab:stats_concat_contra')
