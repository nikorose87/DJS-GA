#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:15:13 2020
Python script to process ankle DJS from a c3d file directly
@author: nikorose
"""


import numpy as np
import pandas as pd
from pathlib import PurePath
import os
from c3d_functions import process_C3D

  
   
# =============================================================================
# Charging the folders 
# =============================================================================
root_dir = PurePath(os.getcwd())
info_dir = '/home/nikorose/enprietop@unal.edu.co/Tesis de Doctorado/Gait Analysis Data/Downloaded/Horst et al./Nature_paper/Gait_rawdata_c3d'
# info_dir = os.path.join( root_dir, 'C3D/Gait_rawdata_c3d')
amputee_dir = root_dir / "TRANSTIBIAL/Transtibial  Izquierda/Gerson Tafud/Opensim files"

# =============================================================================
# Loading gaits
# =============================================================================
S1_static = "S01_0001_Static.c3d"
S1_gait = "S01_0002_Gait.c3d"
S2_gait = "S01_0018_Gait.c3d"
GT_gait = "0143~ab~Walking_01.c3d"

anthro_info = pd.read_csv('C3D/Gait_subject_info.csv', sep=";",
                          decimal=',')
anthro_info.columns = ['ID','mass','age','gender','height']     
        
#options to run n the algorithm 
regression = False  
Obt_GC = False   
save=False 
gen_df_per_coords =True  
# =============================================================================
# Generating all HS and TO points in Horst Data
# =============================================================================
Gait_cycle_points = {'HS1r':[],'HS2r':[],'HSl':[],
                     'TOr':[], 'TO1l':[],'TO2l':[]}
labels = []
df_coord_mean = {}
os.chdir(info_dir)
for sub in range(1,58):
    subject = 'S{}'.format(str(sub).zfill(2))
    #Dict where to store the values
    complete_coords = {}
    for trial in range(2,26):
        sub_plus_trial = '{}_{}_Gait.c3d'.format(subject, str(trial).zfill(4))

        try:
            S = process_C3D(sub_plus_trial, run=True, forces=True, 
                              **anthro_info.iloc[0].to_dict(), printing=False)
            #Adding the trial to the ID
            S.ID = sub_plus_trial[:-4]
            S.markers_GaitCycle('R FOOT MED','R HEEL','SACRUM')
            S.markers_GaitCycle('L FOOT MED','L HEEL','SACRUM')
            #Changing labels to be synchronized with Opensim
            S.labels = [i.replace(' ', '_') for i in S.labels]
            #Saving the multiindex dataframe from IC to IC in right foot
            S.data_point_mk_multi(S.HSidx[0], S.HSidx[1])
            #Fitting to 100 samples so that having the same size in all trials
            if regression:
                try:
                    S.data_points_gc()
                except ValueError:
                    print('Trial {} could not be fitted'.format(sub_plus_trial))
                #Saving the data into the local dir
                #Getting the local directory for the trial
                if save:
                    os.chdir(sub_plus_trial[:-4])
                    S.point_data_df_gc.to_csv('{}_markers_gc.csv'.format(sub_plus_trial[:-4]))
                    S.r2_.to_csv('{}_markers_R2.csv'.format(sub_plus_trial[:-4]))
                    os.chdir('../')
            else: #Doing through interpolation
                S.arrange_df()
                #Generating the indexer
                if gen_df_per_coords:
                    #Making slicers
                    idx = pd.IndexSlice
                    X_df = S.point_data_df_gc.loc[:, idx[:, 'X']]
                    Y_df = S.point_data_df_gc.loc[:, idx[:, 'Y']]
                    Z_df = S.point_data_df_gc.loc[:, idx[:, 'Z']]
                    #Changing the sublevel
                    for num, df_ in enumerate([X_df, Y_df, Z_df]):
                        if num == 0:
                            coord='X'
                        elif num == 1:
                            coord='Y'
                        elif num==2:
                            coord='Z'
                        df_.columns = df_.columns.set_levels( \
                                    df_.columns.levels[1].str.replace(coord, 
                                                                      S.ID), level=1)
                        df_.columns.set_names=['Marker', 'Coord']
                        #Storing in one dict
                        if trial == 2:
                            complete_coords[coord] = [df_]
                        else:
                            complete_coords[coord].append(df_)
                #Getting the local directory for the trial
                if save:
                    os.chdir(sub_plus_trial[:-4])
                    S.point_data_df_gc.to_csv('{}_markers_gc.csv'.format(sub_plus_trial[:-4]))
                    os.chdir('../')
            
            if Obt_GC:
                try:
                    concat_HS_TO = np.array([S.HSidx, S.TOidx]).reshape(1,6)
                    #Appending to the dict in order to transform in df
                    for i, key in enumerate(Gait_cycle_points.keys()):
                        Gait_cycle_points[key].append(concat_HS_TO[0,i])
                    labels.append(sub_plus_trial[:-4])
                except ValueError:
                    print('Trial {} has more than 1 gait cycle'.format(trial))
                    print(S.HSidx, S.TOidx)
                
                Gait_cp_df = pd.DataFrame(Gait_cycle_points, index=labels)
        except OSError:
            print('Trial {} does not exist. Is it static?'.format(sub_plus_trial))
            continue

    if gen_df_per_coords:
        for coord in ['X','Y','Z']:
            complete_coords[coord] = pd.concat(complete_coords[coord], axis=1)
            x_mean = complete_coords[coord].groupby(level=0, axis=1).mean()
            x_std = complete_coords[coord].groupby(level=0, axis=1).std()
            #Creating Multi index for those df
            x_idx = pd.MultiIndex.from_product([[coord],[subject],x_mean.columns])
            x_mean.columns = x_idx
            x_std.columns = x_idx
            complete_coords[coord] = [x_mean, x_std]
        
        df_coord_mean = pd.concat([complete_coords[coord][0] for coord in ['X','Y','Z']], axis=1)
        df_coord_std = pd.concat([complete_coords[coord][1] for coord in ['X','Y','Z']], axis=1)
        if sub == 1:
            df_mean_concat = df_coord_mean
            df_std_concat = df_coord_std
        else:
            df_mean_concat = pd.concat([df_mean_concat, df_coord_mean], axis=1)
            df_std_concat = pd.concat([df_std_concat, df_coord_std], axis=1)
        #Setting names for multilevel
        df_mean_concat.columns.set_names =['Coordinate', 'subject', 'Marker name']
        df_std_concat.columns.set_names =['Coordinate', 'subject', 'Marker name']

#Saving data to CSV

# Gait_cp_df.to_csv('Gait_cycle_bounds.csv')
# df_mean_concat.to_csv('../Horst_markers_mean.csv')
# df_std_concat.to_csv('../Horst_markers_std.csv')

