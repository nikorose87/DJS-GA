#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:12:08 2018


@author: nikorose

"""

from scipy.integrate import simps
import pandas as pd
import numpy as np
import scipy.optimize as optimization
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from bokeh.palettes import Spectral6, Pastel2, Set3, Category20
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.glyphs import Text
from scipy.constants import golden
from scipy.signal import argrelextrema
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from matplotlib.patches import Polygon
from plot_dynamics import plot_dynamic, plot_ankle_DJS


class EnergyFunctions():
    """ In this routine we are going to compile all the needed functions 
    to obtain the total work in the Dynamic Joint Stiffness of any case."""
    def __init__(self, dx=0.5, decimals=4):
        #Integration rate
        self.dx = 0.5
        #output decimals
        self.decimals = 4
    
    def integration_two_features(self, var1, var2, Min = 0, Max = None):
        """
        Simpson rule integration based on two variables

        Parameters
        ----------
        var1 : Dataframe
            DF in the x-axis
        var2 : Dataframe
            DF in the y-axis
        column : string
            Name of the feature to calculate in both DF
        Min : int, optional
            Beginning index where you want to start integrating. The default is 0.
        Max : int, optional
            ending point in df index where you want to end integration. The default is 0

        Returns
        -------
        integration area results

        """
        if Min != 0 and Max is None:
            var1 = var1[:][Min:]
            var2 = var2[:][Min:]
        if Max is not None:
            var1 = var1[:][Min:Max]
            var2 = var2[:][Min:Max]
        self.res_two = np.around(np.float64(simps(var1.values, 
                                              var2.values, 
                                              dx=self.dx)), decimals=self.decimals)
        return self.res_two
    
    def integration_one_feature(self, var1, Min = 0, Max = None):
        """
        Simpson rule integration based on one variables

        Parameters
        ----------
        var1 : Dataframe
            DF in the y-axis
        column : string
            Name of the feature to calculate in both DF
        Min : int, optional
            Beginning index where you want to start integrating. The default is 0.
        Max : int, optional
            ending point in df index where you want to end integration. The default is 0

        Returns
        -------
        integration area results

        """
        if Min != 0 and Max is None:
            var1 = var1[Min:]
        if Max is not None:
            var1 = var1[Min:Max]
        self.res_one = np.around(np.float64(simps(var1, dx=self.dx)), 
                                 decimals=self.decimals)
        return self.res_one

    
    def zero_detection(self, power, vertices =4):
        """
        In order to calculate the ankle subphases 'Heel strike','Roll over',
        'Push off', and 'Swing'. It is suggested that each one changes when the 
        trajectory crosses the 0-axis in x in the ankle power plot.
        
        This statement only works for regular gait intentions at different speed.
        However, this piece of code is able to calculate all zero cross points.
        Parameters.
        
        This is a pre-requisite step for calculating the mechanical work
        ----------
        power : TYPE
            DESCRIPTION.
        vertices : TYPE, optional
            DESCRIPTION. The default is 4.

        Returns
        -------
        numpy array with the indexes where it crosses 

        """
        for diff in power.columns:
            #Eliminating the first two indexes due to are causing problems 
            # power = power.iloc[2:,:]
            # To detect signs of numbers
            asign = np.sign(power[diff]) 
            # To detect the change of signs
            signchange = ((np.roll(asign,1)-asign) !=0).astype(int) 
            # To detect the position where sign changes
            j, = np.where(signchange == 1) 
            try: 
                if "a" not in locals():
                    a = j[:vertices]
                else: 
                    a = np.vstack((a, j[:vertices]))
            except ValueError:
                print ("The number of times in which the data cross zeros is ",
                       "less than the number of vertices especified in:"+str(diff))
                j1 = np.zeros(vertices-j.shape[0])
                j = np.append(j,j1)
                a = np.vstack((a,j))
        self.zeros_ = a
        return self.zeros_
    
    def min_max_power(self, power, proportions=[0.1, 0.5, 0.7]):
        """
        
        The purpose of this is to get the max local in the first portion of the 
        gait, later we will obtain the zero cross point in the roll over zone,
        at the end we will to obtain the min local in the energy release zone.

        Parameters
        ----------
        power : DF
            Power ankle information with different colums
        proportions : integer list, optional
            DESCRIPTION. The default is [0.1, 0.5, 0.7].

        Returns
        -------
        None.

        """
        #Let us define the the portion in which we should obtain max min locals
        # Index df size
        power_size = power.index.shape[0]
        power_mid_foot = power.iloc[:int(proportions[0]*power_size),:]
        power_roll_over = power.iloc[int(proportions[0]*power_size): \
                                     int(proportions[1]*power_size),:]
        power_release = power.iloc[int(proportions[1]*power_size): \
                                     int(proportions[2]*power_size),:]
        min_max = np.zeros((4,power.columns.shape[0])).T
        for j, col in enumerate(power.columns):
            min_max[j,1] = power_mid_foot[col].argmax() 
            min_max[j,3] = power_release[col].argmin() + \
                            int(proportions[1]*power_size)
            zero_ro = np.where(((np.roll(np.sign(power_roll_over[col]),
                                1)-np.sign(power_roll_over[col])) \
                                   !=0).astype(int) == 1)[0]
            try:
                min_max[j,2] = zero_ro[-1] + \
                int(proportions[0]*power_size)
            except IndexError:
                min_max[j,2] = 0
        self.zeros_ = min_max
        return min_max
            
    
    def work(self, df_, columns = ['Heel strike','Roll over','Push off',
                                             'Total'], min_max=False):
        
        """
        This piece of code gets the partial work between zero delimiters.
        Parameters
        ----------

        df_: Dataframe containing power information.
        
        columns : TYPE, optional
            DESCRIPTION. The default is ['Heel strike','Roll over','Push off',
                                         'Total'].

        Returns
        -------
        PartialWorkdf : 
            Dataframe that contains the work portion on each stage
        """
        if min_max == True:
            self.min_max_power(df_)
        else:
            self.zero_detection(df_)
            
        PartialWork = np.zeros(self.zeros_.shape)
        for i in range(self.zeros_.shape[0]):
            for j in range(self.zeros_.shape[1]-1):
                try:
                    PartialWork[i,j]= self.integration_one_feature(df_[df_.columns[i]].values,                                                                   
                            Min = int(self.zeros_[i,j]), 
                            Max = int(self.zeros_[i,j+1]))
                except IndexError:
                    PartialWork[i,j] = 0

        PartialWork[:,-1] = np.sum(np.abs(PartialWork), axis = 1)
        PartialWorkdf = pd.DataFrame(PartialWork, index= df_.columns , 
                                     columns= columns)
        return PartialWorkdf

class Regressions:
    """Main functions to do linear regression in the DJS slope, 
    also we're detecting points to obtain the phases of gait"""
    ## Calculating points to detect each sub-phase of gait
    def CrennaPoints(self, Moments, Angles, labels, 
                     percent= 0.05, threshold = 1.7):
        ### Incremental Ratio between Moments and Angles
        Thres1 = []
        Thres2 = []
        ERP = []
        LRP = []
        DP = []
        for columna in labels:
            ## Index value where the moment increase 2% of manimum moment
            Thres1.append(np.argmax(Moments[columna].values > \
                                    np.max(Moments[columna].values, 
                                    axis=0) * percent, axis = 0)) 
            Si = np.concatenate([(Moments[columna][i:i+1].values - \
                                  Moments[columna][i+1:i+2].values) / 
                                 (Angles[columna][i:i+1].values - 
                                  Angles[columna][i+1:i+2].values) 
                                  for i in range(Moments[columna].values.shape[0])])
            SiAvg = Si / np.abs(np.average(Si.astype('Float64'), axis=0))
            ## Moving the trigger 
            ERP.append(np.argmax(SiAvg[Thres1[labels.index(columna)]:].astype('Float64') 
                        > threshold, axis= 0) + Thres1[labels.index(columna)])
            asignLRP = np.sign(SiAvg[int(Moments[columna].shape[0]*0.25):].astype('Float64'))
            LRP.append(np.argmin(asignLRP.astype('Float64'), axis = 0) + 
                       int(Moments[columna].shape[0]*0.25))
            ## Index value where the moment increases 95% of maximum moment
            Thres2.append(np.argmax(Moments[columna].values > np.max(Moments[columna].values, 
                        axis=0) * (1-percent), axis = 0)) 
            ## Index value where the moment decreases to 2% of manimum moment after reaching the peak
            DP.append(np.argmax(Moments[columna][int(Moments[columna].shape[0] * 
                    0.55):].values < np.max(Moments[columna].values, 
                    axis=0) * percent, axis = 0) + 
                    int(Moments[columna].shape[0]*0.55)) 
        CompiledPoints = np.concatenate((Thres1, ERP, LRP, Thres2, DP)).reshape((5,len(Thres1)))
        CompiledPoints = pd.DataFrame(CompiledPoints, 
                                      index= ['Thres1','ERP', 'LRP', 'Thres2', 'DP'], 
                                      columns= labels)
        return CompiledPoints
 
class Plotting():
    def __init__(self):
        """The main purpose of this function is to plot with the bokeh library 
        the Quasi-stiffness slope """    
        list(Category20[20]).pop(1)
        # Matplotlib params
        
        self.params = {'backend': 'ps',
                  'axes.labelsize': 11,
                  'axes.titlesize': 14,
                  'font.size': 14,
                  'legend.fontsize': 10,
                  'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'text.usetex': True,
                  'font.family': 'sans-serif',
                  'font.serif': ['Computer Modern'],
                  'figure.figsize': (6.0, 6.0 / golden),
                  }
        plt.rcParams.update(self.params)
       
    def QuasiRectiLine(self, name1, name2, compiled, regression, labels, minN = None, maxN = None, grid_space= 250, 
                       leyend= "top_left", x_label= 'Angle (Deg)', 
                       y_label= 'Moment (Nm/kg)', size=5, 
                       title= 'General Quasi-Stiffness plot'):
        """Bokeh plot of Quasi-stiffness points plus the linear regression"""
        color, count = 0, 0
        Points = compiled
        figures = []
        for diff in labels[minN:maxN]:
            f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = diff)
            for j in range(len(compiled.index)-1):
                try:
                    f.line(regression[j+count][:,0], regression[j+count][:,1], line_width=2, color=Category20[20][color])
                except TypeError:
                    pass
            count +=len(compiled.index)-1
            f.ray(x=np.min(name1[diff]), y=name2[diff][compiled[diff]['Thres1']], length=100, angle=0, color=Category20[20][color],line_dash="4 4")
            f.ray(x=np.min(name1[diff]), y=name2[diff][compiled[diff]['Thres2']], length=100, angle=0, color=Category20[20][color],line_dash="4 4")
            f.circle(x = name1[diff], y=name2[diff], size= size, color= Category20[20][color], alpha=0.5)
            f.circle(x = name1[diff][Points[diff]], y=name2[diff][Points[diff]], size= size*2, color= Category20[20][color], alpha=0.5)
            color += 1
            figures.append(f)
        return figures
    
    def PowerLine(self, name1, name2, vertices, x_label='Cycle (Percent.)', 
                  y_label= 'Power (W/kg)', grid_space=450, title='Power cycle plot at the joint',
                  minN=None, maxN=None, size=2, leyend= 'top_left', vertical = True, 
                  text = False):
        """Bokeh plot of Power joint with respect to the gait cycle, plus zero points detection"""
        f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = title)
        color = 0
        for diff in name1.columns[minN:maxN]:
            f.line(x = name2,y=name1[diff], color = Spectral6[color], line_width = size, legend = diff)
            if vertical:
                for i in vertices: 
                    f.ray(x=i, y=-1.5, length=100, angle=1.57079633, color=Pastel2[8][color],line_dash="4 4")
            if text:                  
                instances = ColumnDataSource(dict(x = [-5, 15, 45, 75], y = np.ones(4)*1.5, text = ["Heel Strike", "Roll Over", "Push Off", "Swing"]))                    
                glyph = Text(x = "x", y = "y", text = "text", angle = 0.0, text_color = "black")
                f.add_glyph(instances, glyph)
            color += 1
        f.legend.location = leyend
        return f

    def GRF(self, name1, name2, labels, points=pd.DataFrame([]), points2=pd.DataFrame([]), std=None, x_label='Gait Cycle (%)', 
            y_label= 'GRF (Nm/kg)', grid_space=450, title='GRF plot at the Ankle joint', 
            minN=None, maxN=None, size=2, leyend= 'top_right', individual = False):
        plots = []
        """Bokeh plot of GRF in ankle joint with respect to the gait cycle"""
        if not individual:
            f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = title)
        color = 0
        for diff in labels[minN:maxN]:
            if name2.shape != (100,):
                name3 = name2[diff]
            else:
                name3 = name2
            if individual:
                f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = title)
            f.line(x = name3, y=name1[diff], color = Category20[20][color], line_width = size, legend = diff)
            if std:
                y1 = std[0][diff].values 
                y2 = std[1][diff].values[::-1]
                y = np.hstack((y1,y2))
                x = np.hstack((name3.values, name3.values[::-1]))
                f.patch(x, y, alpha=0.5, line_width=0.1, color= Category20[20][color])
            if points.empty != True:
                for j in points[diff]:
                    if np.isnan(j):
                        pass
                    else:
                        f.circle(x = name3[j], y=name1[diff][j], 
                                 size= size*5, color= Category20[20][color], alpha=0.5)                           
            if points2.empty != True:
                for i in points2[diff]:
                    if np.isnan(i):
                        pass
                    else:
                        f.triangle(x = name3[i], y=name1[diff][i], 
                                   size= size*5, color= Category20[20][color], alpha=0.5)
            color += 1
            plots.append(f)
        f.legend.location = leyend
        if individual:
                return plots
        else:
                return f
    

    def plot_power_and_work(self, df_, label, sd = False):
        
        font = 30
        idx =pd.IndexSlice
        a, b = 5, 65 # integral limits
        x = df_.index.get_level_values(1).unique().values
        y = df_.loc[idx['Ankle  [W/kg]',:],idx[label, 'mean']]

        fig, ax1 = plt.subplots(1,1, figsize=(8, 8))
        line = ax1.plot(x, y, 'r', linewidth=2, label='Ankle Joint Power')
        if sd:
            y_1 = df_.loc[idx['Ankle  [W/kg]',:],idx[label, '-1sd']]
            y_2 = df_.loc[idx['Ankle  [W/kg]',:],idx[label, '+1sd']]
            errorPower = [y-y_1, y_2-y]
            error = ax1.errorbar(x, y, yerr=errorPower, fmt='+', label='SD', 
                                 linewidth=0.2)
        
        # Make the shaded region
        ix = np.linspace(a/100, b/100, (b-a)*3)
        iy = y[a*3:b*3]
        verts = [(a/100, 0)] + list(zip(ix, iy)) + [(b/100, 0)]
        poly = Polygon(verts, facecolor='0.9', edgecolor='0.5', label='Integrated Area')
        ax1.add_patch(poly)
        
        plt.figtext(0.25, 0.6, r"$\int_a^b f(x)\mathrm{d}x$", fontsize=font)
        
        plt.figtext(0.46, 0.1, '$x$', fontsize=font)
        plt.figtext(0.1, 0.9, '$y$', fontsize=font)
        plt.figtext(0.15, 0.36, 'Heel Strike', fontsize=font)
        plt.figtext(0.35, 0.25, 'Rollover', fontsize=font)
        plt.figtext(0.45, 0.50, 'Push Off', fontsize=font)
        plt.figtext(0.65, 0.36, 'Swing', fontsize=font)
        
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.xaxis.set_ticks_position('bottom')
        
        ax1.set_xticks((a/100, b/100))
        ax1.set_xticklabels(('$a$', '$b$'), fontsize=font)
        ax1.set_yticks([])
        ax1.legend(loc = 1, fontsize=font)
        ax1.set_ylabel(r'Ankle  $[\frac{W}{kg}]$', fontsize=font)
        fig.savefig('Power plot work.png')
        plt.show()
        
    def regular_plot(self, df_, var, label, name, sd = False):
        
        font = 30
        idx =pd.IndexSlice
        a, b = 5, 65 # integral limits
        x = df_.index.get_level_values(1).unique().values
        y = df_.loc[idx[var,:],idx[label, 'mean']]

        fig, ax1 = plt.subplots(1,1, figsize=(8, 8))
        ax1.plot(x, y, 'r', linewidth=2, label=name)
        if sd:
            y_1 = df_.loc[idx[var,:],idx[label, '-1sd']]
            y_2 = df_.loc[idx[var,:],idx[label, '+1sd']]
            errorPower = [y-y_1, y_2-y]
            ax1.errorbar(x, y, yerr=errorPower, fmt='+', label='SD', 
                                 linewidth=0.2)
                
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.xaxis.set_ticks_position('bottom')
        
        ax1.set_yticks([])
        ax1.set_xticks((a/100, b/100))
        ax1.set_xticklabels(('$a$', '$b$'), fontsize=font)
        ax1.legend(loc = 1, fontsize=font)
        ax1.set_ylabel(var, fontsize=font)
        ax1.set_xlabel(r'Gait Cycle [\%]', fontsize=font)
        fig.savefig('{}.png'.format(name))
        plt.show()
        
class normalization():
    def __init__(self, height, mass):
        ''' Equations to transform data into normalized gait data'''
        self.g = 9.81    
        self.h = height
        self.mass = mass
    def power_n(self, power):
        return power/(self.mass*self.g**(3.0/2.0)*np.sqrt(self.h))
    def moment_n(self, moment):
        return moment / (self.mass*self.g*self.h)
    def force_n(self,force):
        return force / (self.mass * self.g)
    def all_norm(self, power, moment, force):
        return np.array((self.power_n(power), self.moment_n(moment), 
                         self.force_n(force))) 

class extract_preprocess_data():
    
    def __init__(self, file_name, 
                 exp_name = 'Gait Cycle analysis',
                 dir_loc=None, 
                 header=[0,1]):
        if dir_loc is not None:
            current_dir = os.getcwd()
            os.chdir(dir_loc)
        self.exp_name = exp_name
        #Check if dataframe is inserted instead
        if isinstance(file_name, pd.DataFrame):  
            self.all_dfs = file_name
            self.sheet_names = None
        else:
                
            if file_name[-3:] == 'xls' or file_name[-3:] == 'lsx':
                self.overall_data = pd.ExcelFile(file_name)
                self.sheet_names = pd.Series(self.overall_data.sheet_names[1:])
            
            elif file_name[-3:] == 'csv':
                self.all_dfs = pd.read_csv(file_name, index_col=[0,1], 
                                           header=header)
                self.sheet_names = None
        
        
        if dir_loc is not None: os.chdir(current_dir)
        
    def complete_data(self):
        """
        Compiles in a dictionary all the sheets content for Joint rotations,
        Moments, GRF and power

        Returns
        -------
        None.

        """
        #Storing the angles object
        if self.sheet_names is None:
            self.all_dfs = processing_sheet(self.sheet_names, 
                                         self.overall_data)
        else:
            self.angles = processing_sheet(self.sheet_names[0], 
                                                self.overall_data)
            self.angles_df = self.angles.get_dataset()
            #Storing the moments object
            self.moments = processing_sheet(self.sheet_names[2], 
                                                 self.overall_data)
            self.moments_df = self.moments.get_dataset()
            #Storing the GRF object
            self.GRF = processing_sheet(self.sheet_names[1], 
                                             self.overall_data)
            self.GRF_df = self.GRF.get_dataset()
            #Storing the power object
            self.power = processing_sheet(self.sheet_names[3], 
                                             self.overall_data)
            self.power_df = self.power.get_dataset()
            
            self.all_dfs = pd.concat([self.angles_df, self.moments_df, self.GRF_df,
                                      self.power_df], axis=0)
        return self.all_dfs
    
    
class processing_sheet():
    def __init__(self, sheet, data):
        self.sheet = sheet
        self.overall_data = data
        
    def get_labels_horizon(self):
        """
        Creates Multiindex pandas for  columns

        Parameters
        ----------
        sheet : String
            Sheet name within the excelfile
            
        Returns
        -------
        Pandas Multi-Index columns of the dataset

        """
        # Extracting only the first level labels in the first sheet
        self.labels_first_col = self.overall_data.parse(self.sheet, 
                                            header=None).iloc[0,:].dropna()
        # Extrating the y label in all plots
        self.y_label =self.labels_first_col.pop(0)
        # Name of what is going in the loop
        self.cycle_title = self.labels_first_col.pop(1)
        # Converting to np array
        self.labels_first_col = self.labels_first_col.values
        #Arranging the multiIndex horizontally
        self.labels_second_col = self.overall_data.parse(self.sheet, 
                                    header=None).iloc[1,:].dropna().unique()
        multi_labels= [self.labels_first_col, self.labels_second_col]
        #Creating the multi-index for the columns 
        self.columns = pd.MultiIndex.from_product(multi_labels, 
                                                  names=['Speed', 'Measure'])
        return self.columns
        
    def get_labels_rows(self):
        """
        Creates Multiindex pandas either for  rows in the DF

        Parameters
        ----------
        sheet : String
            Sheet name within the Excelfile
        Returns
        -------
        Pandas Multi-Index indexes of the dataset

        """
        #Extracting only the first level labels in the first sheet
        df_rows = self.overall_data.parse(self.sheet, 
                                            header=0).iloc[:,0:2].dropna()
        self.labels_first_rows = df_rows.iloc[:,0].unique()
        self.labels_second_rows = df_rows.iloc[:,1].unique()
        multi_labels= [self.labels_first_rows, self.labels_second_rows]
        #Creating the multi-index for the columns 
        self.index_ = pd.MultiIndex.from_product(multi_labels, 
                                                  names=df_rows.columns)
        return self.index_
    
    def get_df(self):
        """
        Parameters
        ----------
        sheet : string
            Sheet name within the Excelfile

        Returns
        -------
        raw_df : Returns the Multi-Index Dataframe

        """
        raw_df = self.overall_data.parse(self.sheet,skiprows=[0,1], 
                                         header=None).iloc[:,2:]
        raw_df.columns = self.columns
        raw_df.index = self.index_
        return raw_df
        
    def get_dataset(self):
        """
        
        Calls the functions to build the Multi-Index DF
        Parameters
        ----------
        sheet : string
            Sheet name within the Excelfile

        Returns
        -------
        df_ : Returns the Multi-Index Dataframe
        """
        # Calling the multi-indexes
        self.get_labels_rows()
        self.get_labels_horizon()
        self.df_ = self.get_df()
        return self.df_ 


class ankle_DJS(extract_preprocess_data):
    def __init__(self, file_name, dir_loc=None, header=[0,1],
                 features= ['Ankle Dorsi/Plantarflexion', 
                                  'Vertical',
                                  'Ankle Dorsi/Plantarflexion',
                                  'Ankle'],
                 exp_name = 'Gait Cycle analysis',
                 units = [' Deg [°]', ' Force [%BH]', ' [Nm/kg]', ' [W/kg]'],
                 exclude_names=None):
        
        super().__init__(file_name, exp_name, dir_loc, header)
        self.features = features
        self.idx = pd.IndexSlice
        self.exclude_names= exclude_names
        self.units = units
        self.header = header
        
    def extract_DJS_data(self):
        """
        Function to extract specific labeled data from DJS features

        Returns
        -------
        None.

        """
        super().complete_data()
        if self.exclude_names is not None:
            self.exclude_features()
        self.angles_ankle =  self.angles_df.loc[self.features[0]]
        self.GRF_vertical =  self.GRF_df.loc[self.features[1]]
        self.moment_ankle =  self.moments_df.loc[self.features[2]]
        self.power_ankle =   self.power_df.loc[self.features[3]]
        self.all_dfs_ankle = pd.concat([self.angles_ankle, self.GRF_vertical, 
                                  self.moment_ankle, self.power_ankle], axis=0)
        #Changing the features and adding units
        for i in range(4):
            self.features[i] += ' '+self.units[i] #Ankle angle
        self.index_ankle = pd.MultiIndex.from_product([self.features,
                                                       self.angles_ankle.index], 
                                        names=['Feature', 'Gait cycle %'])
        self.all_dfs_ankle.index = self.index_ankle
        
        return self.all_dfs_ankle
    
    def extract_df_DJS_data(self, idx = [0,1,2,3], units=True):
        """
        As we like to read either preprocessed or raw data. It is needed to build
        this object so as to define the objects from a dataframe

        Parameters
        ----------
        idx : list index position containing the indexes of the level 0 in this order:
            Angles (0), Moments (2), GRF (1), Power (3), optional
            DESCRIPTION. The default is [0,1,2,3].

        Returns
        -------
        The same dataframe already adapted to process DJS

        """
        
        index = self.all_dfs.index.get_level_values(0).unique()
        
        self.angles_ankle =  self.all_dfs.loc[index[idx[0]]]
        self.GRF_vertical =  self.all_dfs.loc[index[idx[1]]]
        self.moment_ankle =  self.all_dfs.loc[index[idx[2]]]
        self.power_ankle =   self.all_dfs.loc[index[idx[3]]]
        self.all_dfs_ankle = pd.concat([self.angles_ankle, self.GRF_vertical, 
                                  self.moment_ankle, self.power_ankle], axis=0)
        
        if not units: #If you do not need to add units
            self.units = ['','','','']
        for i in idx:
            self.features[i] += self.units[i] #Ankle angle
        self.index_ankle = pd.MultiIndex.from_product([self.features,
                                                       self.angles_ankle.index], 
                                        names=['Feature', 'Gait cycle %'])
        self.all_dfs_ankle.index = self.index_ankle
        first_lev = self.all_dfs_ankle.index.get_level_values(0).unique()
        self.all_dfs_QS = self.all_dfs_ankle.loc[self.idx[[first_lev[0],first_lev[2]],:]]
        self.get_levels(self.all_dfs_ankle)
        return self.all_dfs_ankle
    
    def get_levels(self, df_):
        self.first_lev_index = df_.index.get_level_values(0).unique()
        self.second_lev_index = df_.index.get_level_values(1).unique()
        self.first_lev_columns = df_.columns.get_level_values(0).unique()
        self.second_lev_columns = df_.columns.get_level_values(1).unique()
        
        
    def extract_df_QS_data(self, idx = [0,1], units=True):
        """
        As we like to read either preprocessed or raw data. It is needed to build
        this object so as to define the objects from a dataframe only for the Quasi-stiffness

        Parameters
        ----------
        idx : list index position containing the indexes of the level 0 in this order:
            Angles (0), Moments (1)
            DESCRIPTION. The default is [0,1].

        Returns
        -------
        The same dataframe already adapted to process DJS in the QS

        """
        
        index = self.all_dfs.index.get_level_values(0).unique()
        index_1 = self.all_dfs.index.get_level_values(1).unique()
        
        self.angles_ankle =  self.all_dfs.loc[index[idx[0]]]
        self.moment_ankle =  self.all_dfs.loc[index[idx[1]]]
        self.all_dfs_QS = pd.concat([self.angles_ankle, self.moment_ankle], axis=0)
        if units:
            self.features[0] += self.units[0] #Ankle angle
            self.features[2] += self.units[2] #Ankle moment
        self.index_ankle = pd.MultiIndex.from_product([[self.features[0],self.features[2]],
                                                       list(np.round(index_1,3))], 
                                        names=['Feature', 'Gait cycle %'])
        self.all_dfs_QS.index = self.index_ankle
        self.get_levels(self.all_dfs_QS)
        self.all_dfs_ankle = self.all_dfs_QS
        return self.all_dfs_QS
    
    def invert_sign(self, idx):
        if hasattr(self, 'all_dfs_QS'):
            self.all_dfs_QS.loc[self.idx[self.first_lev_index[idx],:],:] = \
                self.all_dfs_QS.loc[self.idx[self.first_lev_index[idx],:],:].apply(lambda x: -x)
            return self.all_dfs_QS
        if hasattr(self, 'all_dfs_ankle'):
            self.all_dfs_ankle.loc[self.idx[self.first_lev_index[idx][idx],:],:] = \
                self.all_dfs_ankle.loc[self.idx[self.first_lev_index[idx][idx],:],:].apply(lambda x: -x)
            return self.all_dfs_ankle
    
    def interpolate_ankledf(self, times=3, replace=False):
        
        if not hasattr(self, 'all_dfs_ankle'):
            self.extract_df_DJS_data()
        
        idx_old = self.all_dfs_ankle.index.get_level_values(0).unique()
        idx_old_1 = self.all_dfs_ankle.index.get_level_values(1).unique()
        bounds = [idx_old_1[0], idx_old_1[-1]]
        new_shape = times*len(idx_old_1)
        mat = np.empty((new_shape*len(idx_old), self.all_dfs_ankle.shape[1]))
        mat[:] = np.nan
        new_idx = pd.MultiIndex.from_product([list(idx_old), 
                                              np.linspace(bounds[0], bounds[1], new_shape).round(3)])
        new_dfs_ankle = pd.DataFrame(mat, index=new_idx, 
                                     columns=self.all_dfs_ankle.columns)
        #Interpolating values
        for index_ in idx_old:
            for colname, col in self.all_dfs_ankle.iteritems():
                new_dfs_ankle.loc[self.idx[index_,:], 
                                  colname] = interp1d(idx_old_1, 
                                col[index_])(np.linspace(bounds[0], bounds[1],new_shape).round(3))
                                                      
        if replace:
            self.all_dfs_ankle = new_dfs_ankle
        
        return new_dfs_ankle
        
    def change_labels(self, new_labels, level=0):
        """
        

        Parameters
        ----------
        new_labels : List
            Replace column labels in the df_ankle 

        Returns
        -------
        all_dfs_ankle.

        """
        if not hasattr(self, 'all_dfs_ankle'):
            self.extract_df_DJS_data()
        
        idx_old = self.all_dfs_ankle.columns.get_level_values(level).unique()
        for num, name in enumerate(new_labels):
            self.all_dfs_ankle.columns = self.all_dfs_ankle.columns.set_levels(\
                    self.all_dfs_ankle.columns.levels[level].str.replace(idx_old[num], name), level=level)
        return self.all_dfs_ankle
    
    def deg_to_rad(self):
        """
        Converts angle information from degrees to radians 

        Returns
        -------
        None.

        """
        self.angles_ankle = self.angles_ankle.apply(np.deg2rad, axis=1)
    
    def exclude_features(self):
        """
        It drops the multi-index level 0 columns

        Parameters
        ----------
        col_num : list with string names
            the column names to remove must match

        Returns
        -------
        None.

        """
        
        self.angles_df = self.angles_df.drop(columns = self.exclude_names)
        self.moments_df = self.moments_df.drop(columns = self.exclude_names)
        self.GRF_df = self.GRF_df.drop(columns = self.exclude_names)
        self.power_df = self.power_df.drop(columns = self.exclude_names)
    
    def energy_calculation(self, min_max=True):
        self.energy_fun = EnergyFunctions()
        self.power_energy = self.energy_fun.work(self.power_ankle, min_max=min_max)
        
    def work_produced(self):
        if not hasattr(self, 'energy_fun'):
            self.energy_fun = EnergyFunctions()
        work_prod = []
        for i in self.angles_ankle.columns:
            work_prod.append( \
                self.energy_fun.integration_two_features(self.angles_ankle[i],
                                                    self.moment_ankle[i]))
        self.work_prod = pd.Series(work_prod, index = self.angles_ankle.columns,
                                   name='Work produced')
        return self.work_prod
    
    def min_max_idx(self, df_):
        _max_idx = df_.idxmax(axis=0)
        _min_idx = df_.idxmin(axis=0)
        return _max_idx, _min_idx
    
    def work_absorbed(self):
        if not hasattr(self, 'energy_fun'):
            self.energy_fun = EnergyFunctions()
        self.angles_max_idx, self.angles_min_idx = \
            self.min_max_idx(self.angles_ankle)
        work_abs = []
        for i in self.angles_ankle.columns:
            work_abs.append( \
                self.energy_fun.integration_two_features(self.angles_ankle[i],
                                                    self.moment_ankle[i], 
                Min= 0, Max=self.angles_max_idx[i]))
        self.work_abs = pd.Series(work_abs, index = self.angles_ankle.columns,
                                  name='Work Absorbed')
        return self.work_abs
    
    def total_work(self):
        w_prd = self.work_produced()
        w_abs = self.work_absorbed()
        self.total_work = pd.concat([self.work_abs, self.work_prod], axis=1)
        self.total_work['Work Total'] = w_prd.apply(np.abs) + w_abs.apply(np.abs)
        return self.total_work
        
        
    def first_derivative(self,x) :
        return x[2:] - x[0:-2]
    
    def second_derivative(self,x) :
        return x[2:] - 2 * x[1:-1] + x[:-2]
    
    def curvature(self, x, y) :
        x_1 = self.first_derivative(x)
        x_2 = self.second_derivative(x)
        y_1 = self.first_derivative(y)
        y_2 = self.second_derivative(y)
        return np.abs(x_1 * y_2 - y_1 * x_2) / np.sqrt((x_1**2 + y_1**2)**3)
    

    def turning_points(self, x, y, turning_points=10, smoothing_radius=False,
                            cluster_radius=10) :
        """ 
        =============================================================================
        # Some explaining in the following order:
        # You will probably want to smooth your curve out first, then calculate the curvature, 
        # then identify the highest curvature points:
        #     turning_points is the number of points you want to identify
        #     smoothing_radius is the radius of a smoothing convolution to be 
              applied to your data before computing the curvature
        #     cluster_radius is the distance from a point of high curvature selected 
              as a turning point where no other point should be considered as a candidate.
        """ 
        if smoothing_radius:
            weights = np.ones(2 * smoothing_radius + 1)
            new_x = convolve1d(x, weights, mode='constant', cval=0.0)
            new_x = new_x[smoothing_radius:-smoothing_radius] / np.sum(weights)
            new_y = convolve1d(y, weights, mode='constant', cval=0.0)
            new_y = new_y[smoothing_radius:-smoothing_radius] / np.sum(weights)
        else:
            new_x, new_y = x, y
        # Applying curvature after smoothing
        shape = new_x.shape[0]
        k = np.atleast_1d(self.curvature(new_x[:int(0.95*shape)], new_y[:int(0.95*shape)]))
        # We are removing the last 5 % due to there's another curvature
        # sorting by the max argument from last position in array
        turn_point_idx = np.argsort(k, axis=0)[::-1]
        t_points = []
        while len(t_points) < turning_points and len(turn_point_idx) > 0:
            t_points += [turn_point_idx[0]]
            idx = np.abs(turn_point_idx - turn_point_idx[0]) > cluster_radius
            turn_point_idx = turn_point_idx[idx]
        t_points = np.array(t_points)
        t_points += smoothing_radius #+ 1
        
        #If there are no more points It should fill with zeros
        while t_points.shape[0] < turning_points -1:
            t_points = np.append(t_points, [0])
        #Sorting by order
        t_points = t_points[np.argsort(t_points, axis=0)].astype(np.int64)
        return t_points
    
    def get_curvature_points(self, x, y, k=3, error=0.001, 
                             turning_points=6, cluster_radius=10):
        # Applying curvature after smoothing
        k = self.curvature_splines(x, y, k, error)
        # sorting by the max argument from last position in array
        turn_point_idx = np.argsort(k, axis=0)[::-1]
        t_points = []
        while len(t_points) < turning_points and len(turn_point_idx) > 0:
            t_points += [turn_point_idx[0]]
            idx = np.abs(turn_point_idx - turn_point_idx[0]) > cluster_radius
            turn_point_idx = turn_point_idx[idx]
        t_points = np.array(t_points)
        
        #If there are no more points It should fill with zeros
        while t_points.shape[0] < turning_points -1:
            t_points = np.append(t_points, [0])
        #Sorting by order
        t_points = t_points[np.argsort(t_points, axis=0)].astype(np.int64)
        return t_points
    
    def curvature_splines(self, x, y, k=2, error=0.001):
        """Calculate the signed curvature of a 2D curve at each point
        using interpolating splines.
        Parameters
        ----------
        x,y: numpy.array(dtype=float) representing the DJS array
        error : float
            The admisible error when interpolating the splines
        k: Degree of the fit function, default = 4
        Returns
        -------
        curvature: numpy.array shape (n_points, )
        Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
        but more accurate, especially at the borders.
        
        Piece of code taken from 
        https://gist.github.com/elyase/451cbc00152cb99feac6
        
        This is more accurate 
        """
    
        # handle list of complex case
        if y is None:
            x, y = x.real, x.imag
    
        t = np.arange(x.shape[0])
        std = error * np.ones_like(x)
    
        fx = UnivariateSpline(t, x, k=k, w=1 / np.sqrt(std))
        fy = UnivariateSpline(t, y, k=k, w=1 / np.sqrt(std))
    
        xˈ = fx.derivative(1)(t)
        xˈˈ = fx.derivative(2)(t)
        yˈ = fy.derivative(1)(t)
        yˈˈ = fy.derivative(2)(t)
        curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
        
        return curvature
    
        
    
    def get_turning_points(self, rows=[0,2], cols=None, method = 'normal', turning_points= 6, 
                           cluster_radius= 15, param_1 = 4, param_2=0.001):
        """
        This piece of code will return the df with points where the loop changes
        the most 

        Parameters
        ----------
        rows : np integer array, optional
            List of two elements with indexes positions. 
            The first index should be the x data, the second index is the w data
            The default is None, which means that angle and moments are located 
            at those positions, frequently.
        cols : np integer array, optional
            List with the first level order columns. The default is None.

        Returns
        -------
        None.

        """
        cols_labels = self.all_dfs_ankle.columns
        rows_labels = self.first_lev_index
        if cols is not None:
            cols_labels = self.first_lev_columns[cols]
        if rows is not None:
            rows_labels = self.first_lev_index[rows]
        #We will calculate turning points with the mean
        if len(self.header) != 1:
            df_turn = self.all_dfs_ankle.loc[self.idx[rows_labels,:], 
                                             cols_labels]
        else:
            df_turn = self.all_dfs_ankle.loc[self.idx[rows_labels,:], :]
        # df_turn = df_turn.columns.droplevel(1)

        for num, col in enumerate(df_turn.columns):
            if method == 'splines':
                #This method is not well configured yet
                TP = self.get_curvature_points(df_turn.loc[self.idx[rows_labels[0],:], 
                                                      col].values,
                      df_turn.loc[self.idx[rows_labels[1], :], col].values, param_1, param_2,
                      turning_points, cluster_radius)
            elif method == 'normal':
                TP = self.turning_points(df_turn.loc[self.idx[rows_labels[0],:], 
                                                          col].values,
                          df_turn.loc[self.idx[rows_labels[1], :], col].values, 
                          turning_points, param_1, cluster_radius)
            if num == 0:
                TP_df = np.array(TP)
            else:
                try:
                    TP_df = np.vstack((TP_df, np.array(TP)))
                except ValueError:
                    cols_labels = cols_labels.drop(col)
                    print("Turning points for trial {} could not be generated".format(col))
        TP_arr = np.atleast_2d(TP_df)
        #Adding the point (0,0) to point the init Controlled Plantar Flexion
        TP_arr = np.hstack((np.zeros((TP_arr.shape[0],1)),TP_arr))
        self.TP_df = pd.DataFrame(TP_arr, index=cols_labels, 
                    columns=['point {}'.format(i) for i in range(TP_arr.shape[1])]) 
        self.TP_df = self.TP_df.astype(np.int64)
        return self.TP_df
    