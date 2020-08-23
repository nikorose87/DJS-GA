# Python Script for Automatic Scaling, IK, and ID computing on OpenSim 4.1

# First, import os.path library to be able to use file and folder functions

import os.path

# Then, configure the model folder to the one where your OpenSim model is located, as follows:
# ModelFolder = os.path.join(u"Absolute path Folder"). Each directory must be written preceded by \\
# ModelName	= os.path.join(ModelFolder, 'ModelName.osim')
# scaleSetuoPath = os.path.join(modelFolder, "Custom_Setup_Scale_File.xmll");
ModelFolder = os.path.join(u'D:\\Mis Documentos\\Documentos\\IA\\articulo\\Horst data\\Gait_rawdata_c3d\\S01 data\\TRC files')
ModelName = os.path.join(ModelFolder,'custom_model.osim')
scaleSetup	=	os.path.join(ModelFolder, "Custom_Setup_Scale_File.xml");

scaleModelName	=	os.path.join(ModelFolder, "S01_model.osim");
# Load model 
loadModel(ModelName)

## Scaling Tool
# Create the scale tool object from existing xml
scaleTool = modeling.ScaleTool(scaleSetup)
scaleTool.run();

## load Scaled Model
# Load model 
loadModel(scaleModelName)