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



class process_C3D():
    def __init__(self, file, run=True, forces=True, ID='NN', mass=80, 
                 age=30, gender='female', height=1.60):
        self.forces = forces
        self.C3D = c3d(file, extract_forceplat_data=self.forces)
        self.ID = ID
        self.mass = mass
        self.gender = gender
        self.height = height
        if run:
            self.data_points_markers()
            if self.forces:
                self.data_points_platform()
    
    def data_points_markers(self):
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
        self.point_data_xdf = pd.DataFrame(self.point_data_mk[0,:,:],
                    index = self.labels, columns= self.idx).T
        self.point_data_ydf = pd.DataFrame(self.point_data_mk[1,:,:],
                    index = self.labels, columns= self.idx).T
        self.point_data_zdf = pd.DataFrame(self.point_data_mk[2,:,:],
                    index = self.labels, columns= self.idx).T
        
        return 
    
    def data_points_platform(self):
        if not hasattr(self, 'labels'):
            self.main_info()
        self.point_data_pf = self.C3D['data']['platform']
        self.point_data_pf_df = []
        for platform in self.point_data_pf:
            self.point_data_pf_df.append(platform['Tz'])
            
        
    def main_info(self):
        """
        Separates the needed information of the C3D file

        Returns
        -------
        None.

        """
        self.points = self.C3D['parameters']['POINT']
        self.analog_params = self.C3D['parameters']['ANALOG']
        self.analogs = self.C3D['data']['analogs']
        self.force_platform = self.C3D['parameters']['FORCE_PLATFORM']
        self.npoints = self.points['USED']['value'][0]
        self.nframes = self.points['FRAMES']['value'][0]
        self.labels = self.points['LABELS']['value']
        self.rate = self.points['RATE']['value'][0]
        self.frames = self.points['FRAMES']['value'][0]
        self.tot_time = self.frames / self.rate
        
    
    def detectHS(self, heel_str, sacrum_str):
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
        
        HS = argrelextrema(diff.values, np.greater)
        self.HSidx= self.point_data_xdf.index[HS]
        return self.HSidx
    
    def detectTO(self, toe_str, sacrum_str):
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
        
        TO = argrelextrema(diff.values, np.less)
        self.TOidx = self.point_data_xdf.index[TO]
        return self.TOidx
    
    def markers_GaitCycle(self, toe_str, heel_str, sacrum_str):
        self.detectHS(heel_str, sacrum_str)
        self.detectTO(toe_str, sacrum_str)
        
    
    
        
   
# =============================================================================
# Charging the folders 
# =============================================================================
root_dir = PurePath(os.getcwd())
info_dir = root_dir / 'C3D/Gait_rawdata_c3d'
amputee_dir = root_dir / "TRANSTIBIAL/Transtibial  Izquierda/Gerson Tafud/Opensim files"

# =============================================================================
# Loading gaits
# =============================================================================
S1_static = os.path.join(info_dir, "S01_0001_Static.c3d")
S1_gait = os.path.join(info_dir, "S01_0002_Gait.c3d")
S1_gait = os.path.join(info_dir, "S01_0002_Gait.c3d")
S2_gait = os.path.join(info_dir, "S02_0003_Gait.c3d")
GT_gait = os.path.join(amputee_dir, "0143~ab~Walking_01.c3d")


anthro_info = pd.read_csv('C3D/Gait_subject_info.csv', sep=";",
                          decimal=',')
anthro_info.columns = ['ID','mass','age','gender','height']     
        
        
# =============================================================================
# Reading the data from c3d with ezc3d
# We could not integrate it with opensim
# =============================================================================
S1 = process_C3D(S1_gait, run=True, forces=True, 
                 **anthro_info.iloc[0].to_dict()) # , extract_forceplat_data=True
S1.markers_GaitCycle('R FOOT MED','R HEEL','SACRUM')
S2 = process_C3D(S2_gait, run=True, forces=True, 
                 **anthro_info.iloc[1].to_dict()) 
S1.markers_GaitCycle('R FOOT MED','R HEEL','SACRUM')
# points_residuals = c['data']['meta_points']['residuals']
# analog_data = c['data']['analogs']

Gerson_C3D = process_C3D(GT_gait, run=True, forces=False)
Gerson_C3D.markers_GaitCycle('l met','l heel','sacrum')

# tables = osim.opensim.C3DFileAdapter.readFile(info_dir, 1)
# markers = tables['markers']
# forces = tables['forces']


