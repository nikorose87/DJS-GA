#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:08:30 2020
The purpose of this script is to write the XML files needed for Scaling, IK or 
ID process in Opensim
@author: nikorose and Carlos Cuervo
"""
import xml.dom.minidom as md
import os
# Input Scaling model Parameters

class XML_scaling():
    
    def __init__(self, mass=80.0, height= 1.65, age=30,
                 model_file_name="custom_model.osim",
                 static_marker_data= "subject_Static.trc",
                 marker_set_xml_file = "Scale_MarkerSet.xml",
                 output_model_file_name = "subject_model.osim",
                 output_motion_file_name = "Static_output.mot",
                 output_scale_name = "subject-scaled",
                 save_path_file = "Custom_Setup_Scale_File.xml",
                 setup_scale_file = "Setup_Scale_File.xml", 
                 source_dir=None ):
    
        self.current_dir = os.getcwd()
        if source_dir is not None:
            self.source_dir = source_dir
        else:
            self.source_dir = self.current_dir
        self.mass_value = str(mass)
        self.height = str(height)
        self.age = str(age)
        self.model_file_name = os.path.join (self.source_dir, model_file_name)
        self.static_marker_data = static_marker_data
        self.marker_set_xml_file = os.path.join (self.source_dir, 
                                                 marker_set_xml_file)
        self.output_model_file_name = os.path.join (self.source_dir, 
                                                    output_model_file_name)
        self.output_motion_file_name = os.path.join (self.source_dir,
                                                     output_motion_file_name)
        self.output_scale_name = output_scale_name
        self.save_path_file = os.path.join (self.source_dir, save_path_file)
        self.setup_scale_file = setup_scale_file
        self.writing_scaling()
        
    def writing_scaling(self):
        os.chdir(self.source_dir)
        # Load .xml file
        file = md.parse(self.setup_scale_file)
        # Set output name
        output_name = file.getElementsByTagName('ScaleTool')
        output_name[0].setAttribute('name', self.output_scale_name)
        # Print unedited tag values
        print( "Unedited <mass>: " + file.getElementsByTagName("mass" )[ 0 ].firstChild.nodeValue )
        print( "Unedited <model_file>: " + file.getElementsByTagName( "model_file" )[ 0 ].firstChild.nodeValue )
        marker_file_name = file.getElementsByTagName( "marker_file" )
        output_file_name = file.getElementsByTagName( "output_model_file" )
        for i in marker_file_name: 
            print("Unedited Static <marker_file>"+i.firstChild.nodeValue)
            # Modifies <marker_file> tag values
            i.firstChild.nodeValue = self.static_marker_data
        print("Unedited <marker_set_file>: "+file.getElementsByTagName( \
                "marker_set_file" )[ 0 ].firstChild.nodeValue)
        for i in output_file_name: 
            print("Unedited <output_model_file>: "+i.firstChild.nodeValue)
            # Modifies <output_model_file> tag values
            i.firstChild.nodeValue = self.output_model_file_name
        print("Unedited <output_motion_file>: "+file.getElementsByTagName( \
            "output_motion_file" )[ 0 ].firstChild.nodeValue)
        # Modifies scale setup tag values
        features = {"mass": self.mass_value, "height": self.height, 
                    "age": self.age , "model_file": self.model_file_name, 
                    "marker_set_file": self.marker_set_xml_file, 
                    "output_motion_file": self.output_motion_file_name}
        for key, value in features.items():
            file.getElementsByTagName(key)[0].childNodes[0].nodeValue = value
 
        # Print Customized tag values
        print( "Custom <mass>: " + file.getElementsByTagName( "mass" )[ 0 ].firstChild.nodeValue )
        print( "Custom <model_file>: " + file.getElementsByTagName( "model_file" )[ 0 ].firstChild.nodeValue )
        for i in marker_file_name: 
            print("Custom Static <marker_file>: "+i.firstChild.nodeValue)
        print("Custom <marker_set_file>: "+file.getElementsByTagName( "marker_set_file" )[0].firstChild.nodeValue)
        for i in output_file_name: 
            print("Custom <output_model_file>: "+i.firstChild.nodeValue)
        print("Custom <output_motion_file>: "+file.getElementsByTagName( "output_motion_file" )[0].firstChild.nodeValue)
        # writing the changes in "file" object to  
        # the "Custom_Setup_Scale_File.xml" file 
        with open( self.save_path_file, "w" ) as fs:  
            fs.write(file.toxml())
            fs.close()
            
        os.chdir(self.current_dir)
        
class XML_IK():
    def __init__(self, IK_output_name = "subject_IK",
        save_IK_path_file = "custom_Setup_IK_File.xml",
        input_model_file_name = "subject_model.osim",
        gait_marker_file = "subject_gait.sto",
        output_motion_file_name = "subject_IK.mot",
        setup_IK_file = "Setup_IK.xml",
        source_dir = None):
                

        self.IK_output_name = IK_output_name
        self.save_IK_path_file = save_IK_path_file
        self.input_model_file_name = input_model_file_name
        self.gait_marker_file = gait_marker_file
        self.output_motion_file_name = output_motion_file_name
        self.setup_IK_file = setup_IK_file
        self.current_dir = os.getcwd()
        if source_dir is not None:
            self.source_dir = source_dir
        else:
            self.source_dir = self.current_dir
        self.writing_IK()
        
    def writing_IK(self):
        os.chdir(self.source_dir)
        # Load Setup_IK.xml
        file = md.parse(self.setup_IK_file)
        # Set output name
        output_name = file.getElementsByTagName('InverseKinematicsTool')
        output_name[0].setAttribute('name', self.IK_output_name)
        # Print unedited tag values
        print( "Unedited <model_file>: " + file.getElementsByTagName( \
                                "model_file" )[0].firstChild.nodeValue )
        print( "Unedited <marker_file>: " + file.getElementsByTagName( \
                               "marker_file" )[0].firstChild.nodeValue )
        print( "Unedited <output_motion_file>: " + file.getElementsByTagName( \
                        "output_motion_file" )[0].firstChild.nodeValue )
        # Set Custom IK tag values
        file.getElementsByTagName( \
             "model_file" )[0].childNodes[0].nodeValue = self.input_model_file_name
        file.getElementsByTagName( "marker_file" )[0].childNodes[\
                                         0].nodeValue = self.gait_marker_file
        file.getElementsByTagName("output_motion_file" )[\
                    0].childNodes[0].nodeValue = self.output_motion_file_name
        print( "Edited <model_file>: " + 
              file.getElementsByTagName( "model_file" )[0].firstChild.nodeValue )
        print( "Edited <marker_file>: " + 
              file.getElementsByTagName( "marker_file" )[0].firstChild.nodeValue )
        print( "Edited <output_motion_file>: " + 
              file.getElementsByTagName( "output_motion_file" )[0].firstChild.nodeValue )
        # writing the changes in "file" object to  
        # the "Custom_Setup_IK_File.xml" file 
        with open( self.save_IK_path_file, "w" ) as fs:  
            fs.write(file.toxml())
            fs.close()
        os.chdir(self.current_dir)


class XML_ID():
    def __init__(self,  ID_output_name = "subject_ID",
        save_ID_path_file = "Custom_Setup_ID_File.xml",
        input_model_file_name = "subject_model.osim",
        GRF_Setup_file = "Custom_Setup_GRF_File.xml",
        IK_coordinates_file = "subject_IK.mot",
        GRF_mot_file = "subject_gait_grf.mot",
        output_motion_file_name = "subject_ID.sto",
        setup_ID_file = "Setup_ID.xml",
        setup_GRF_file = "Setup_GRF.xml",
        source_dir=None):
        
        self.ID_output_name = ID_output_name
        self.save_ID_path_file = save_ID_path_file
        self.input_model_file_name = input_model_file_name
        self.GRF_Setup_file = GRF_Setup_file
        self.IK_coordinates_file = IK_coordinates_file
        self.GRF_mot_file = GRF_mot_file
        self.output_motion_file_name = output_motion_file_name
        self.setup_ID_file =setup_ID_file
        self.setup_GRF_file = setup_GRF_file
        
        self.current_dir = os.getcwd()
        if source_dir is not None:
            self.source_dir = source_dir
        else:
            self.source_dir = self.current_dir
        self.writing_ID()
        self.writing_GRF()
        
    def writing_ID(self):
        os.chdir(self.source_dir)
        # Load Setup_ID.xml
        file = md.parse(self.setup_ID_file)
        # Set output name
        output_name = file.getElementsByTagName('InverseDynamicsTool')
        output_name[0].setAttribute('name', self.ID_output_name)
        # Print unedited tag values
        print( "Unedited <model_file>: " + file.getElementsByTagName( \
                              "model_file" )[ 0 ].firstChild.nodeValue )
        print( "Unedited <external_loads_file>: " + file.getElementsByTagName( \
                            "external_loads_file" )[ 0 ].firstChild.nodeValue )
        print( "Unedited <coordinates_file>: " + file.getElementsByTagName( \
                                "coordinates_file" )[ 0 ].firstChild.nodeValue )
        print( "Unedited <output_gen_force_file>: " + file.getElementsByTagName( \
                            "output_gen_force_file" )[ 0 ].firstChild.nodeValue )
        
        # Set Custom ID tag values
        file.getElementsByTagName( "model_file" )[ 0 ].childNodes[ \
                            0 ].nodeValue = self.input_model_file_name
        file.getElementsByTagName( "external_loads_file" )[ 0 ].childNodes[ \
                            0 ].nodeValue = self.GRF_Setup_file
        file.getElementsByTagName( "coordinates_file" )[ 0 ].childNodes[ \
                            0 ].nodeValue = self.IK_coordinates_file
        file.getElementsByTagName( "output_gen_force_file" )[ 0 ].childNodes[ \
                            0 ].nodeValue = self.output_motion_file_name
        
        # Print Custom tag values
        print( "Edited <model_file>: " + file.getElementsByTagName( \
                         "model_file" )[ 0 ].firstChild.nodeValue )
        print( "Edited <external_loads_file>: " + file.getElementsByTagName( \
                        "external_loads_file" )[ 0 ].firstChild.nodeValue )
        print( "Edited <coordinates_file>: " + file.getElementsByTagName( \
                        "coordinates_file" )[ 0 ].firstChild.nodeValue )
        print( "Edited <output_gen_force_file>: " + file.getElementsByTagName( \
                        "output_gen_force_file" )[ 0 ].firstChild.nodeValue )
        # writing the changes in "file" object to  
        # the "Custom_Setup_ID_File.xml" file 
        with open( self.save_ID_path_file, "w" ) as fs:  
            fs.write( file.toxml() )
            fs.close()
        
        os.chdir(self.current_dir)
        
    def writing_GRF(self):
        os.chdir(self.source_dir)
        # Load Setup_GRF.xml
        file = md.parse( self.setup_GRF_file)
        # Print unedited tag values
        print( "Unedited <datafile>: " + file.getElementsByTagName( "datafile" )[ 0 ].firstChild.nodeValue )
        # Set GRF test file
        file.getElementsByTagName( "datafile" )[ 0 ].childNodes[ 0 ].nodeValue = self.GRF_mot_file
        # Print Custom tag values
        print( "Edited <datafile>: " + file.getElementsByTagName( "datafile" )[ 0 ].firstChild.nodeValue )
        # writing the changes in "file" object to  
        # the "Custom_Setup_GRF_File.xml" file 
        with open( self.GRF_Setup_file, "w" ) as fs:  
            fs.write( file.toxml() )
            fs.close()
        os.chdir(self.current_dir)