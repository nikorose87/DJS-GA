#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:38:54 2020
Trying to read c3d and implement IK in opensim
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

# =============================================================================
# Opensim handling class
# =============================================================================
class opensim_handling():
    
    def __init__(self, filename, work_dir=os.getcwd(), TRC=True, num_platforms=2):
        self.current_dir = os.getcwd()
        self.work_dir = work_dir
        self.filename = filename
        self.num_platforms =num_platforms
        os.chdir(self.work_dir)
        if not os.path.exists(filename[:-4]):
            os.makedirs(filename[:-4])
        self.working_dir = os.path.join(self.work_dir, filename[:-4])
        self.C3D_tables()
        self.forces_info()
        os.chdir(self.current_dir)
        #C3D_tables must be run at first
        if TRC:
            self.TRC_write(self.markersTable)
        self.writing_file(self.markers_flat, 'markers')
        self.writing_file(self.forces_flat, 'forces')
        
    
    def TRC_write(self, markersVec3):
        Filename = '{}_{}.trc'.format(self.filename[:-4], 'markers')
        os.chdir(self.working_dir)
        osim.TRCFileAdapter_write(markersVec3, Filename)
        os.chdir(self.work_dir)
        
    def C3D_tables(self):
        c3dFileAdapter = osim.C3DFileAdapter()
        c3dFileAdapter.setLocationForForceExpression( \
                                osim.C3DFileAdapter.ForceLocation_CenterOfPressure)
        self.tables = c3dFileAdapter.read('{}'.format(self.filename))
        self.markersTable = c3dFileAdapter.getMarkersTable(self.tables)
        self.marker_names = self.markersTable.getColumnLabels()
        #We will have to replace space by _, take care because is for this example only
        self.marker_names = [i.replace(' ', '_') for i in self.marker_names]
        self.markersTable.setColumnLabels(self.marker_names)
        self.forcesTable = c3dFileAdapter.getForcesTable(self.tables)
        labels = []
        for i in range(1,self.num_platforms+1):
            labels.extend(['ground_force_{}_v'.format(i), 
                           'ground_force_{}_p'.format(i), 
                           'ground_moment_{}_m'.format(i)])
        self.forcesTable.setColumnLabels(labels)
        self.forces_flat = self.forcesTable.flatten()
        self.markers_flat = self.markersTable.flatten()
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
    
        
        
# =============================================================================
# Locating files 
# =============================================================================
root_dir = PurePath(os.getcwd())
info_dir = root_dir / 'C3D/Gait_rawdata_c3d'

S1_static = "S01_0001_Static.c3d"
S1_gait = "S01_0002_Gait.c3d"
S2_gait = "S02_0003_Gait.c3d"
anthro_info = pd.read_csv('C3D/Gait_subject_info.csv', sep=";",
                          decimal=',')
anthro_info.columns = ['ID','gender','age','mass','height']
S1_gait_osim = opensim_handling(S1_gait, info_dir, TRC=False)
S1_static_osim = opensim_handling(S1_static, info_dir, TRC=True)

# =============================================================================
# Generating the scaling XML file
# =============================================================================
osim_files_dir = os.path.join(info_dir, 'osim_files_S01')
subject_static_dir = os.path.join(info_dir, S1_static[:-4])

sub = 1
subject = 'S{}'.format(str(sub).zfill(2))
trial = 1
S01_scaling = XML_scaling(mass= anthro_info.iloc[0,3],
                        height = anthro_info.iloc[0,4],
                        age = anthro_info.iloc[0,2],
                        static_marker_data= os.path.join( subject_static_dir, 
                            '{}_{}_Static_markers.trc'.format(subject, str(trial).zfill(4))),
                        output_model_file_name = '{}_model.osim'.format(subject),
                        save_path_file = "custom_{}_Setup_Scale_File.xml".format(subject),
                        output_scale_name="{}_scaled".format(subject),
                        output_motion_file_name = "{}_static_output.mot".format(subject),
                        source_dir= osim_files_dir)

trial += 1
sub_plus_trial = '{}_{}'.format(subject, str(trial).zfill(4))
subject_dir = os.path.join(info_dir, S1_gait[:-4])
S01_IK = XML_IK(IK_output_name = "{}_IK".format(sub_plus_trial),
                save_IK_path_file = "custom_{}_Setup_IK_File.xml".format(sub_plus_trial),
                input_model_file_name = os.path.join(osim_files_dir,
                                                     '{}_model.osim'.format(subject)),
                gait_marker_file = os.path.join(subject_dir, 
                                '{}_Gait_markers.sto'.format(sub_plus_trial)),
                output_motion_file_name = os.path.join(subject_dir,
                                        "{}_IK.mot".format(sub_plus_trial)),
                setup_IK_file = "Setup_IK.xml",
                source_dir = osim_files_dir)

S01_ID = XML_ID(ID_output_name = "{}_ID".format(sub_plus_trial), 
                save_ID_path_file = "custom_{}_Setup_ID_File.xml".format(sub_plus_trial),
                input_model_file_name = os.path.join(subject_dir,
                                        "{}_IK.mot".format(sub_plus_trial)),
                GRF_Setup_file = "custom_{}_Setup_GRF_File.xml".format(subject),
                IK_coordinates_file = os.path.join(subject_dir,
                                        "{}_IK.mot".format(sub_plus_trial)),
                GRF_mot_file = os.path.join(subject_dir,
                                        "{}_Gait_forces.sto".format(sub_plus_trial)),
                output_motion_file_name = os.path.join(subject_dir,
                                        "{}_ID.sto".format(sub_plus_trial)),
                setup_ID_file = "Setup_ID.xml",
                setup_GRF_file = "Setup_GRF.xml",
                source_dir= osim_files_dir)
# =============================================================================
# Executing process 
# =============================================================================

#move to the scaling directory
os.chdir(osim_files_dir)
#Trying to add geometry models
path2geometryDir = "/home/nikorose/Workspace/Opensim/opensim-models/Geometry"
scaletool = osim.ScaleTool('custom_{}_Setup_Scale_File.xml'.format(subject))
scaletool.run()

Invtool = osim.InverseKinematicsTool('custom_{}_Setup_IK_File.xml'.format(sub_plus_trial))
Invtool.run()

IDtool = osim.InverseDynamicsTool('custom_{}_Setup_ID_File.xml'.format(sub_plus_trial))
scaled_model = osim.Model('{}_model.osim'.format(subject)) 
model_state = scaled_model.initSystem()
IDtool.run()
os.chdir(root_dir)
#ModelVisualizer.addDirToGeometrySearchPaths(Path2GeometryDir)

# =============================================================================
# Amputee location dir|
# =============================================================================

amp_dir = os.path.join(root_dir, 
        "TRANSTIBIAL/Transtibial  Izquierda/Gerson Tafud/0143~ab~Walking_01.c3d")
