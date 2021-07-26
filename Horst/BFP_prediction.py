#!/usr/bin/env python
# coding: utf-8

# # Code description
# 
# In the following code a Body Fat Percentage Prediction is developed using kinematic marker trajectories on overground walking as input features. The dataset used for this prediction can be found in [Horst et al.](https://data.mendeley.com/datasets/svx74xcrjr/3).
# 
# The dataset used consists on kinematic trajectory information of 54 different body markers at each instance of gait cycle [from gait start (0%) to full gait (100%)] along 3 axes (X,Y,Z) for 57 subjects, with each subject performing 20$\pm$2 gait trials. 
# 
# 
# Each gait trial information will be considered as a sample for the desgined regressor [rows], and its corresponding trajectory marker information as sample features [columns]. This means that input data array for the regressor will have:
# * 57 subjects x 20$\pm$2 gait trials $\approx$ 1140 rows
# * 100 gait cycle percentages x 54 markes x 3 axes = 16200 columns
# 
# The code is divided into three sections:
# 1. Input features preparation: In this part, the raw dataset is uploaded and processed in order to obtain input_features array as described above.
# 2. Target data preparation: Target body fat percentage is calculated for each sample using the equation developed by [Gallagher et al.](https://academic.oup.com/ajcn/article/72/3/694/4729363)
# 3. Body fat percentage prediction: A regressor is trained and tested using input_features and target_body_fat_percentage. Regressor performance is shown with a predicted values vs MSE per sample plot and Bland-Altman plot.

# ## Import libraries
# 
# Libraries used along all sections are imported

# In[1]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import statsmodels.api as sm


# # 1. Input features preparation
# 
# In this section, input_features array described above is built from [Horst et al.](https://data.mendeley.com/datasets/svx74xcrjr/3) gait_rawdata dataset.

# [Horst et al.](https://data.mendeley.com/datasets/svx74xcrjr/3) dataset is uploaded to marker_data DataFrame. Then, raw values are extracted and stored on raw_marker_values array.

# In[2]:


marker_data = pd.read_csv('Horst_markers_raw.csv',sep=',',low_memory=False)
raw_marker_values = marker_data.iloc[1:,1:].values.astype(float)


# The markers placed on the test subjects are:

# In[3]:


markers = marker_data.columns.values[1:55] # Information only
markers


# The following code is used to determine the quantity of gait trials each subject performed:

# In[4]:


'''
s_trials contains the quantity of trials for each subject.
    s_trials[0] = 20 -> 20 trials on subject 01
    s_trials[1] = 18 -> 18 trials on subject 02
    s_trials[2] = 21 -> 21 trials on subject 03
    .
    .
    .
    s_trials[k] =  m trials on subject k+1
'''
s_trials = []
for i in range(1,58):
    if i<=9:
        s_trials.append(marker_data.iloc[0,1:].str.contains('S0'+str(i), regex=False).sum())
    else:
        s_trials.append(marker_data.iloc[0,1:].str.contains('S'+str(i), regex=False).sum())
s_trials = np.asarray(s_trials)/162
print('The number of samples on input_features array is '+str(int(s_trials.sum())))


# raw_marker_values array is procesed to shape it into input_features array taking into account the exact number of gait trials performed by subject.

# In[5]:


input_features = pd.DataFrame([])
i = 0
j = 0
for j in range(57):
    k = 0
    subject_data = pd.DataFrame([])
    m = s_trials[j].astype(int)
    for n in range(1,4):
        flattened_data = []
        while k<(54*m*n):
            flattened_data.append(raw_marker_values[:,i:i+54].reshape((1,5400)))
            i = i+54
            k = k+54
        flattened_data = pd.DataFrame(np.asarray(flattened_data).reshape((m,5400)))
        subject_data = pd.concat([subject_data, flattened_data], ignore_index=True, axis = 1)
    input_features = pd.concat([input_features, subject_data], ignore_index=True, axis = 0)
print('input_features array has '+str(np.asarray(input_features).shape[0])+' rows and '+str(np.asarray(input_features).shape[1])+' columns')


# # 2. Target data preparation
# As each subject's body fat percentage is not explicitly known, [Gallagher et al.](https://academic.oup.com/ajcn/article/72/3/694/4729363) equation will be used to calculate this value using BMI, Age and sex as follows:
#     
#     BF% = 76.0 - 1097.8 * (1/BMI) - 20.6 * sex
#           + 0.053 * age + 154 * sex * (1/BMI)
#           + 0.034 * sex * age
#           
#     BMI -> kg/m^2 = weight / height^2
#     Age -> years
#     Sex -> Male = 1, Female = 0
# 
# It is important to notice that in this equation, ethnicity has been already evaluated as all subjects are not from Asian ethnicity.

# Subject's BMI is calculated using weight and height data provided by [Horst et al.](https://data.mendeley.com/datasets/svx74xcrjr/3) in Gait_subject_info.csv

# In[6]:


weight = np.asarray(pd.read_csv("Gait_subject_info.csv",sep=';')['body mass [kg]'].to_list())[:,np.newaxis]
height = np.asarray(pd.read_csv("Gait_subject_info.csv",sep=';')['body size [m]'].to_list())[:,np.newaxis]
bmi = weight/np.power(height,2)


# Subject's age is extracted from [Horst et al.](https://data.mendeley.com/datasets/svx74xcrjr/3) Gait_subject_info.csv

# In[7]:


age = np.asarray(pd.read_csv("Gait_subject_info.csv",sep=';')['age [years]'].to_list())[:,np.newaxis]


# Subject's sex is extracted from [Horst et al.](https://data.mendeley.com/datasets/svx74xcrjr/3) Gait_subject_info.csv. Taking into account that in [Gallagher et al.](https://academic.oup.com/ajcn/article/72/3/694/4729363) equation male subjects must be identified as 1 and female subjects as 0, Gait_subject_info.csv gender column must be logically inverted.

# In[8]:


sex = np.logical_not(np.asarray(pd.read_csv("Gait_subject_info.csv",sep=';')['gender [1=female / 0=male]'].to_list(), dtype=bool)[:,np.newaxis])


# Finally, each sample body fat percentage is calculated using [Gallagher et al.](https://academic.oup.com/ajcn/article/72/3/694/4729363) equation.

# In[9]:


bfp = 76.0 - 1097.8 * (1/bmi) - 20.6 * sex + 0.053 * age + 154 * sex * (1/bmi) + 0.034 * sex * age
bfp_per_trial = pd.DataFrame([])
for j in range(57):
    m = s_trials[j].astype(int)
    for n in range(m):
        bfp_per_trial=pd.concat([bfp_per_trial, pd.DataFrame([bfp[j]])], ignore_index=True, axis = 0)
bfp = bfp_per_trial


# # 3. Body Fat Percentage prediction
# 
# Body Fat Percentage regresion is trained and tested using kinematic marker values as input features.

# bfp array is horizontally stacked to input_features

# In[10]:


input_data = pd.concat([input_features,bfp],ignore_index=True,axis=1)


# Rows with NaN values on input_data DataFrame are dropped

# In[11]:


input_data = input_data.dropna()
input_data = np.asarray(input_data)


# input_data is split into train and test datasets

# In[12]:


train_features, test_features, train_target, test_target = train_test_split(input_data[:,:-1],input_data[:,-1],
                                                    train_size=0.8, test_size=0.2, random_state=42)


# Regressor was built using [TPOT](http://epistasislab.github.io/tpot/) optimization tool 

# In[13]:


exported_pipeline = make_pipeline(
    PCA(iterated_power=2, svd_solver="randomized"),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=5, min_child_weight=11, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=1.0, verbosity=0)),
    LassoLarsCV(normalize=False)
)

set_param_recursive(exported_pipeline.steps, 'random_state', 0)

exported_pipeline.fit(train_features, train_target)


# In[14]:


prediction = exported_pipeline.predict(test_features)
print("MSE = "+str(mean_squared_error(test_target, prediction)))
print("R2 = "+str(r2_score(test_target, prediction)))
errors = list()
for i in range(len(prediction)):
    err = (test_target[i] - prediction[i])**2
    errors.append(err)
plt.scatter(prediction, errors)
plt.xlabel('Predicted Value')
plt.ylabel('Squared Error')
plt.title('Predicted values vs SE per sample')
plt.show()
test_target = test_target[:,np.newaxis]
prediction = prediction[:,np.newaxis]
errors = np.asarray(errors).reshape((227,1))
results_df = pd.DataFrame(data=np.hstack((test_target,prediction,errors)),columns=['Test target BF%', 'Predicted BF%', 'SE'])

f, ax = plt.subplots(1, figsize = (7,5))
sm.graphics.mean_diff_plot(test_target, prediction, ax = ax)
plt.title('Mean difference plot')
plt.show()

