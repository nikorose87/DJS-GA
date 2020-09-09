#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:38:54 2020
Trying to read c3d and implement ID in opensim
@author: nikorose
"""

import opensim as osim
import os
import pandas as pd
import numpy as np
from pathlib import PurePath
from XML_write_files import XML_scaling, XML_IK, XML_ID
import path
import sys
import matplotlib.pyplot as plt
import shutil

# =============================================================================
# Opensim handling class
# =============================================================================
class opensim_handling():
    
    def __init__(self, filename, work_dir=os.getcwd(), TRC=True, 
                 num_platforms=2, rotate=None):
        self.root_dir = os.getcwd()
        self.work_dir = work_dir
        self.filename = filename
        self.rotate = {'x':0, 'y':0, 'z':0}
        if rotate is not None:
            self.rotate.update(rotate)
        self.num_platforms =num_platforms
        os.chdir(self.work_dir)
        #Chack if file exists before
        if os.path.isfile(filename):
            if not os.path.exists(filename[:-4]):
                os.makedirs(filename[:-4])
        self.working_dir = os.path.join(self.work_dir, filename[:-4])
        self.C3D_tables()
        self.forces_info()
        os.chdir(self.root_dir)
        #C3D_tables must be run at first
        if TRC:
            self.TRC_write(self.markersTable)
        else:
            self.writing_file(self.markers_flat, 'markers')
        self.writing_file(self.forces_flat, 'forces')
        
    
    def TRC_write(self, markersVec3):
        Filename = '{}_{}.trc'.format(self.filename[:-4], 'markers')
        os.chdir(self.working_dir)
        osim.TRCFileAdapter_write(markersVec3, Filename)
        os.chdir(self.work_dir)
    
    def rotate_data_table(self, table, axisString, deg):
        
        """
        Rotate OpenSim::TimeSeriesTableVec3 entries using an axis and angle.
    
        Parameters
        ----------
        table: OpenSim.common.TimeSeriesTableVec3
    
        axis: 3x1 vector
    
        deg: angle in degrees
    
        """
            #set up the transform
            
        if axisString == 'x':
            axis = [1,0,0]
        elif axisString == 'y':
            axis = [0,1,0]
        elif axisString == 'z':
            axis = [0,0,1]
        R = osim.Rotation(np.deg2rad(deg),
                             osim.Vec3(axis[0], axis[1], axis[2]))
        for i in range(table.getNumRows()):
            vec = table.getRowAtIndex(i)
            vec_rotated = R.multiply(vec)
            table.setRowAtIndex(i, vec_rotated)
        return table
        
        

        
    def C3D_tables(self):
        c3dFileAdapter = osim.C3DFileAdapter()
        c3dFileAdapter.setLocationForForceExpression( \
                                osim.C3DFileAdapter.ForceLocation_CenterOfPressure)
        self.tables = c3dFileAdapter.read('{}'.format(self.filename))
        self.markersTable = c3dFileAdapter.getMarkersTable(self.tables)
        #Rotating the data in markers
        for key, item in self.rotate.items():
            if item != 0:
                self.markersTable = self.rotate_data_table(self.markersTable, 
                                                           key, item)
        self.marker_names = self.markersTable.getColumnLabels()
        self.freq = float(self.markersTable.getTableMetaDataAsString('DataRate'))
        self.frames = self.markersTable.getNumRows()
        #We will have to replace space by _, take care because is for this example only
        self.marker_names = [i.replace(' ', '_') for i in self.marker_names]
        self.markersTable.setColumnLabels(self.marker_names)
        #Getting time information
        self.time_array = self.markersTable.getIndependentColumn()
        self.init_time = self.time_array[0]
        self.finish_time = self.time_array[-1]

        self.forcesTable = c3dFileAdapter.getForcesTable(self.tables)
        #Rotating the data in markers
        for key, item in self.rotate.items():
            if item != 0:
                self.forcesTable = self.rotate_data_table(self.forcesTable, 
                                                           key, item)
        self.num_platforms = int(self.forcesTable.getNumColumns()/3)
        labels = []
        for i in range(1,self.num_platforms+1):
            labels.extend(['ground_force_{}_v'.format(i), 
                           'ground_force_{}_p'.format(i), 
                           'ground_moment_{}_m'.format(i)])
        self.forcesTable.setColumnLabels(labels)
        self.mm2m(self.forcesTable)
        self.forces_flat = self.forcesTable.flatten()
        self.markers_flat = self.markersTable.flatten()
        return

    def mm2m(self, table):
        nRows  = table.getNumRows()
        labels = table.getColumnLabels()

        for i in range(table.getNumColumns()):
            # % All force columns will have the 'f' prefix while point
            # % and moment columns will have 'p' and 'm' prefixes,
            # % respectively.
            if labels[i][-1] != 'v':
                columnData = table.updDependentColumnAtIndex(i)
                for n in range(nRows):
                    dat = columnData.getElt(n,0)
                    # % Divide by 1000
                    for i in range(3):#Because it is size 3
                        dat.set(i,dat.get(i)/1000)
        return


    def forces_info(self):
        self.fpCalMats = self.forces_flat.getTableMetaDataVectorMatrix("CalibrationMatrices")
        self.fpCorners = self.forces_flat.getTableMetaDataVectorMatrix("Corners")
        self.fpOrigins = self.forces_flat.getTableMetaDataVectorMatrix("Origins")
    
    def writing_file(self, var, name, ext='sto'):
        # Make sure flattenned marker data is writable/readable to/from file.
        Filename = '{}_{}.{}'.format(self.filename[:-4], name, ext)
        stoAdapter = osim.STOFileAdapter()
        os.chdir(self.working_dir)
        stoAdapter.write(var, Filename)
        os.chdir(self.work_dir)
    
    def transform2df(self, file_name, name):
        if not hasattr(self, 'dfs'):
            self.dfs = {}
        self.current_dir = os.getcwd()
        os.chdir(self.working_dir)
        break_line = self.endheader_line(file_name)
        self.dfs[name] = pd.read_table(file_name,
                              skiprows=break_line, delim_whitespace=True, 
                              engine='python', decimal='.', index_col=0)
        os.chdir(self.current_dir)
        return self.dfs[name]
    
    def endheader_line(self, file):
        with open(file,'r') as f:
            lines = f.read().split("\n")
        
        word = 'endheader' # dummy word. you take it from input
        
        # iterate over lines, and print out line numbers which contain
        # the word of interest.
        for i,line in enumerate(lines):
            if word in line: # or word in line.split() to search for full words
                #print("Word \"{}\" found in line {}".format(word, i+1))
                break_line = i+1
        return break_line

        
# =============================================================================
# Locating files 
# =============================================================================
root_dir = PurePath(os.getcwd())
info_dir = '/home/nikorose/enprietop@unal.edu.co/Tesis de Doctorado/Gait Analysis Data/Downloaded/Horst et al./Nature_paper/Gait_rawdata_c3d'

#Changing location dir 
os.chdir(info_dir)
#Reading anthropometric charateristics
anthro_info = pd.read_csv('../Gait_subject_info.csv', sep=";",
                          decimal=',')
anthro_info.columns = ['ID','gender','age','mass','height']

# =============================================================================
# Generating the ID process for obtaining the Quasi-Stiffness
# =============================================================================


for sub in range(34,58): #Because there are 57 subjects
    # fig, ax = plt.subplots()
    subject = 'S{}'.format(str(sub).zfill(2))
    #Defining the Osim files dir
    osim_files_dir = os.path.join(info_dir, 'osim_files_{}'.format(subject))
    if not os.path.exists(osim_files_dir):
        os.makedirs(osim_files_dir)
        #there is a folder with the custom files
        osim_custom_dir = '../osim_files/'
        list_files = os.listdir(osim_custom_dir)
        for file in list_files:
            shutil.copy2(osim_custom_dir+file, osim_files_dir)
        
    trials = range(1,26)
    for trial in trials:
        sub_plus_trial = '{}_{}'.format(subject, str(trial).zfill(4))
        try:
            if trial == 1:
                subject_static_dir = os.path.join(info_dir, '{}_Static'.format(sub_plus_trial))
                S_static_osim = opensim_handling('{}_Static.c3d'.format(sub_plus_trial), 
                                                 info_dir, TRC=True, rotate={'y':-90, 'z':-90})
                S01_scaling = XML_scaling(time = S_static_osim.finish_time,
                                          mass= anthro_info.iloc[sub-1,3],
                                        height = anthro_info.iloc[sub-1,4],
                                        age = anthro_info.iloc[sub-1,2],
                                        static_marker_data= os.path.join( subject_static_dir, 
                                            '{}_{}_Static_markers.trc'.format(subject, str(trial).zfill(4))),
                                        output_model_file_name = '{}_model.osim'.format(subject),
                                        save_path_file = "custom_{}_Setup_Scale_File.xml".format(subject),
                                        output_scale_name="{}_scaled".format(subject),
                                        output_motion_file_name = "{}_static_output.mot".format(subject),
                                        source_dir= osim_files_dir)
                #move to the scaling directory
                os.chdir(osim_files_dir)
                scaletool = osim.ScaleTool('custom_{}_Setup_Scale_File.xml'.format(subject))
                scaletool.run()
                os.chdir(root_dir)
            else:  
                S_gait_c3d = '{}_Gait.c3d'.format(sub_plus_trial)
                S_gait_osim = opensim_handling(S_gait_c3d, info_dir, TRC=True, rotate={'x': -90})
                subject_dir = os.path.join(info_dir, S_gait_c3d[:-4])
                S01_IK = XML_IK(time= S_gait_osim.finish_time, 
                                IK_output_name = "{}_IK".format(sub_plus_trial),
                                save_IK_path_file = os.path.join(subject_dir,
                                                    "custom_{}_Setup_IK_File.xml".format(sub_plus_trial)),
                                input_model_file_name = os.path.join(osim_files_dir,
                                                                     '{}_model.osim'.format(subject)),
                                gait_marker_file = os.path.join(subject_dir, 
                                                '{}_Gait_markers.trc'.format(sub_plus_trial)),
                                output_motion_file_name = os.path.join(subject_dir,
                                                        "{}_IK.mot".format(sub_plus_trial)),
                                setup_IK_file = "Setup_IK.xml",
                                source_dir = osim_files_dir)
                
                S01_ID = XML_ID(time= S_gait_osim.finish_time, 
                                ID_output_name = "{}_ID".format(sub_plus_trial), 
                                save_ID_path_file = os.path.join(subject_dir,
                                                    "custom_{}_Setup_ID_File.xml".format(sub_plus_trial)),
                                input_model_file_name = os.path.join(osim_files_dir,
                                                        '{}_model.osim'.format(subject)),
                                GRF_Setup_file = os.path.join(subject_dir,
                                                              "custom_{}_Setup_GRF_File.xml".format(sub_plus_trial)),
                                IK_coordinates_file = os.path.join(subject_dir,
                                                        "{}_IK.mot".format(sub_plus_trial)),
                                GRF_mot_file = os.path.join(subject_dir,
                                                        "{}_Gait_forces.sto".format(sub_plus_trial)),
                                output_motion_file_name = "{}_ID.sto".format(sub_plus_trial),
                                setup_ID_file = "Setup_ID.xml",
                                setup_GRF_file = "Setup_GRF.xml",
                                cutoff_freq = 10,
                                source_dir= osim_files_dir,
                                save_dir = subject_dir)
                
                # =============================================================================
                # Executing process 
                # =============================================================================
                
                #move to the scaling directory
                os.chdir(osim_files_dir)
                #Trying to add geometry models
                Path2GeometryDir = "/home/nikorose/Workspace/Opensim/opensim-models/Geometry"
                #Adding the path for visualization
                osim.ModelVisualizer.addDirToGeometrySearchPaths(Path2GeometryDir)
                      
                Invtool = osim.InverseKinematicsTool(os.path.join( subject_dir,
                    'custom_{}_Setup_IK_File.xml'.format(sub_plus_trial)))
                Invtool.run()
                #Transforming to DataFrames
                # IK_df = S_gait_osim.transform2df('{}_IK.mot'.format(sub_plus_trial), 'IK') 
                # GRF_df = S_gait_osim.transform2df('{}_Gait_forces.sto'.format(sub_plus_trial), 'GRF')
                IDtool = osim.InverseDynamicsTool(os.path.join( subject_dir,
                            'custom_{}_Setup_ID_File.xml'.format(sub_plus_trial)))
                scaled_model = osim.Model('{}_model.osim'.format(subject)) 
                model_state = scaled_model.initSystem()
                IDtool.run()
                # ID_df = S_gait_osim.transform2df('{}_ID.sto'.format(sub_plus_trial), 'ID')
                os.chdir(root_dir)
            
                # Plotting the QS
                # ax.plot(IK_df['ankle_angle_l'], -ID_df['ankle_angle_l_moment'])
        except RuntimeError:
            print('Trial {} does not exist.'.format(sub_plus_trial)+
                  ' Look if something is wrong')
            continue

    # ax.set_ylabel('Ankle Moment [Nm]')
    # ax.set_xlabel('Ankle Angle [Deg]')
    # ax.set_title('{} Quasi-stiffness'.format(subject))

# =============================================================================
# Plotting some results
# =============================================================================
# n = [0,3,6]
# [GRF_df.iloc[:, np.r_[0:i, i+6:i+9]].plot() for i in n]

# IK_df.loc[:,['ankle_angle_l','ankle_angle_r']].plot()
# IK_df.loc[:,['knee_angle_l','knee_angle_r']].plot()

# ID_df.loc[:,['ankle_angle_l_moment','ankle_angle_r_moment']].plot()

# =============================================================================
# Amputee location dir|
# =============================================================================

# amp_dir = os.path.join(root_dir, 
#         "TRANSTIBIAL/Transtibial  Izquierda/Gerson Tafud/0143~ab~Walking_01.c3d")

