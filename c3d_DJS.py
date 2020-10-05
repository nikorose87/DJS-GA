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
from ezc3d import c3d
from scipy.signal import argrelextrema
#Libraries for regression task
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error



class process_C3D():
    def __init__(self, file, run=True, forces=True, ID='NN', mass=80, 
                 age=30, gender='female', height=1.60, printing=True):
        self.forces = forces
        self.printing = printing
        self.C3D = c3d(file, extract_forceplat_data=self.forces)
        self.ID = ID
        self.mass = mass
        self.gender = gender
        self.height = height
        if run:
            self.data_points_markers()
            if self.forces:
                self.data_points_platform()
    

    def data_point_mk_multi(self, begin=0, finish=None, gc_idx=True):
        """
        Generates the concatenated dataframe for markers data, arranged in a Multi
        Index

        Parameters
        ----------
        begin : TYPE, optional
            initial point to take in DF. The default is 0.
        finish : TYPE, optional
            final desired index. The default is None.
        gc_idx : BOOL, optional
            Generate the index in gait cycle percentage. The default is True.

        Returns
        -------
        None.

        """
        if not hasattr(self, 'labels'):
            self.main_info()
        self.point_data_mk = self.C3D['data']['points']
        self.idx = np.linspace(0, self.tot_time, self.frames)
        iterables = [self.labels,['X','Y','Z']]
        x_arr = self.point_data_mk[0,:,:].T
        y_arr = self.point_data_mk[1,:,:].T
        z_arr = self.point_data_mk[2,:,:].T
        for it in range(x_arr.shape[1]):
            if it == 0:
                _arr = x_arr[:,it]
            else:
                _arr = np.vstack((_arr, x_arr[:,it]))
            _arr = np.vstack((_arr, y_arr[:,it]))
            _arr = np.vstack((_arr, z_arr[:,it]))
        
        _arr = _arr.T
        
        self.columns = pd.MultiIndex.from_product(iterables, 
                                                   names=['Label', 'Coordinate'])
        self.point_data_df = pd.DataFrame(_arr, index=self.idx, 
                                          columns=self.columns)
        self.point_data_df.index.name = 'Time [s]'
        if begin != 0:
            self.point_data_df = self.point_data_df.loc[begin:,:]
        if finish is not None:
            self.point_data_df = self.point_data_df.loc[:finish,:]
        if gc_idx:
            self.idx = np.linspace(0,1, self.point_data_df.shape[0])
            self.point_data_df.index = self.idx
            self.point_data_df.index.name = 'Gait Cycle [%]'
        return
    
    def right_fit(self, var_x, var_y, var_to_pred, col, alpha=0.03):
        '''
        In order to give the same size for all trials, we need to adjust all 
        to the same index

        Parameters
        ----------
        var_x : Pandas series or array
            Data in x axis
        var_y :Pandas series or array
            Data in y axis
        var_to_pred:Pandas series or array
            Data in x axis to predict

        Returns
        -------
        R2_sim : float
            Coefficient of Determination
        pred_sim : float
            array with the predicted data

        '''
        order = 0
        #Making regression for simulated data
        r2_ = 0.0
        while order <= 12 and (bool(r2_ < 1.01) != bool(r2_ > 0.97)):
            #A pipeline between a polynomial features and ridge regression was perform 
            y_linear_lr = make_pipeline(PolynomialFeatures(order), linear_model.Ridge(alpha=alpha))
            y_linear_lr.fit(var_x, var_y)
            r2_ = y_linear_lr.score(var_x, var_y)
            pred_sim = y_linear_lr.predict(var_to_pred)
            if order == 0:
                R2_init = np.array([r2_])
                pred_vals = pred_sim
            else:
                R2_init = np.hstack((R2_init, r2_))
                pred_vals = np.hstack((pred_vals, pred_sim))
            order +=1
        best_R2 = np.argmax(R2_init)
        if self.printing:
            print('for column {}, the best poly order was {}'.format(col, 12-best_R2))
        
        return R2_init[best_R2], pred_vals[:,best_R2]
    
    def data_points_gc(self, samples=100):
        if not hasattr(self, 'point_data_df'):
            print('Dataframe from gait cycle has not been generated',
                  'please run data_point_mk_multi first')
            return
        time_gc = np.linspace(0,1,samples)
        self.point_data_df_gc = pd.DataFrame(np.zeros((time_gc.shape[0], 
                                                    self.point_data_df.shape[1])),
                                             index = time_gc, 
                                             columns = self.point_data_df.columns)
        self.r2_ = pd.Series(np.zeros(self.point_data_df.shape[1]), 
                       index=self.point_data_df.columns, name='R2 values')

        for num, col in enumerate(self.point_data_df.columns):
            self.r2_[num], self.point_data_df_gc[col] = self.right_fit(self.idx.reshape(-1,1), 
                                         self.point_data_df[col].values.reshape(-1, 1),
                                         time_gc.reshape(-1,1), col)
            
    def data_points_markers(self, begin=0, finish=None):
        """
        Generates the dataframes to process easily the information in the three 
        coordinates 

        Returns
        -------
        None.

        """
        if not hasattr(self, 'labels'):
            self.main_info()
        self.point_data_mk = self.C3D['data']['points']
        self.idx = np.linspace(0, self.tot_time, self.frames)
        self.point_data_xdf = pd.DataFrame(self.point_data_mk[0,:,begin:finish],
                    index = self.labels, columns= self.idx).T
        self.point_data_ydf = pd.DataFrame(self.point_data_mk[1,:,begin:finish],
                    index = self.labels, columns= self.idx).T
        self.point_data_zdf = pd.DataFrame(self.point_data_mk[2,:,begin:finish],
                    index = self.labels, columns= self.idx).T
        
        return 
    
    def data_points_platform(self):
        if not hasattr(self, 'labels'):
            self.main_info()
        self.point_data_pf = self.C3D['data']['platform']
        self.platform_data = []
        for platform in self.point_data_pf:
            self.platform_data.append({key:platform[key] for key in platform.keys()})
            
        
    def main_info(self):
        """
        Separates the needed information of the C3D file

        Returns
        -------
        None.

        """
        self.params_points = self.C3D['parameters']['POINT']
        self.params_analog = self.C3D['parameters']['ANALOG']
        self.analogs = self.C3D['data']['analogs']
        self.params_platform = self.C3D['parameters']['FORCE_PLATFORM']
        self.npoints = self.params_points['USED']['value'][0]
        self.nframes = self.params_points['FRAMES']['value'][0]
        self.labels = self.params_points['LABELS']['value']
        self.rate = self.params_points['RATE']['value'][0]
        self.frames = self.params_points['FRAMES']['value'][0]
        self.tot_time = self.frames / self.rate
        
    def change_YZ(self):
        #Swapping marker position
        if not hasattr(self, 'point_data_mk'):
            self.data_points_markers()
        Y = np.copy(self.point_data_mk[1,:,:])
        Z = np.copy(self.point_data_mk[2,:,:])
        #Changing coordinates
        self.point_data_mk[2,:,:] = Y
        self.point_data_mk[1,:,:] = Z
        self.C3D['data']['points'] = self.point_data_mk
        self.data_points_markers()
        #Swapping force platform information
        if not hasattr(self, 'platform_data'):
            self.data_points_platform()
        for platform in self.platform_data:
            for key in ['origin', 'force', 'moment', 'center_of_pressure',
                        'Tz']:
                platform[key][[1,2]] = platform[key][[2,1]]
        
        self.C3D['data']['platform'] = self.platform_data
        self.params_points['X_SCREEN']['value'][0] = '+X'
        self.params_points['Y_SCREEN']['value'][0] = '+Y'
        
    def detectHS(self, heel_str, sacrum_str, order=5):
        """
        From the paper
        Zeni, J. A., Richards, J. G., & Higginson, J. S. (2008). 
        https://doi.org/10.1016/j.gaitpost.2007.07.007
        
        They have indicated the following algorithm  to detect the Heel Strike 
        instant so that determining the gait cycle.
        
        $t_{hs} = argmax(X_{heel} - X_{sacrum})$
        
        Parameters
        ----------
        heel_str : Str
            Label name of the Heel marker.
        sacrum_str : TYPE
            Label name of the sacrum marker.

        Returns
        -------
        None.

        """
        
        diff = pd.Series(self.point_data_xdf[heel_str] - 
                         self.point_data_xdf[sacrum_str])
        
        HS = argrelextrema(diff.values, np.greater, order=order)
        if hasattr(self, 'HSidx'):
            self.HSidx.extend(list(self.point_data_xdf.index[HS]))
        else:
            self.HSidx= list(self.point_data_xdf.index[HS])
        return self.HSidx
    
    def detectTO(self, toe_str, sacrum_str, order=5):
        """
        From the paper
        Zeni, J. A., Richards, J. G., & Higginson, J. S. (2008). 
        https://doi.org/10.1016/j.gaitpost.2007.07.007
        
        They have indicated the following algorithm  to detect the Toe Off 
        instant so that determining the stance phase in the gait cycle.
        
        $t_{toe} = \argmin(X_{toe} - X_{sacrum})$
        
        Parameters
        ----------
        toe_str : Str
            Label name of the Toe marker.
        sacrum_str : TYPE
            Label name of the sacrum marker.

        Returns
        -------
        None.

        """
        diff = pd.Series(self.point_data_xdf[toe_str] - 
                         self.point_data_xdf[sacrum_str])
        
        TO = argrelextrema(diff.values, np.less, order=order)
        if hasattr(self, 'TOidx'):
            self.TOidx.extend(list(self.point_data_xdf.index[TO]))
        else:
            self.TOidx = list(self.point_data_xdf.index[TO])
        return self.TOidx
    
    def markers_GaitCycle(self, toe_str, heel_str, sacrum_str):
        self.detectHS(heel_str, sacrum_str)
        self.detectTO(toe_str, sacrum_str)
    
    def interpolate(self, df_, samples = 100):
        #first we're gonna add columns
        time_gc = np.linspace(0,1,samples)
        point_data_df_gc = pd.DataFrame(np.empty((time_gc.shape[0], 
                                                    df_.shape[1])),
                                             index = time_gc, 
                                             columns = df_.columns)
        point_data_df_gc[:] = np.nan
        df_ = df_.append(point_data_df_gc)
        #sorting by index
        df_ = df_.sort_index(axis = 0)
        #Creating new data with interpolation
        df_ = df_.interpolate(method='polynomial', order= 3, axis=0)
        #Extracting only the 100 samples
        df_ = df_.loc[time_gc, :]
        df_.index = np.round(df_.index, 4)
        #Avoid duplicated points 
        df_ = df_[~df_.index.duplicated(keep='first')]
        return  df_
            
    def arrange_df(self, swap_cols=False):
        df_gc = self.interpolate(self.point_data_df)
        df_gc.name = self.ID
        if swap_cols:
            df_gc = df_gc.stack()
            df_gc.index = df_gc.index.swaplevel()
        self.point_data_df_gc = df_gc
        return self.point_data_df_gc
        
    
    
        
   
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
                              **anthro_info.iloc[0].to_dict(), printing=False) # , extract_forceplat_data=True
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

