#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:34:07 2020
kinematics and kinetics diagrams for multi-index dataframes in human gait
@author: nikorose
"""

# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import math
from itertools import combinations
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import inspect
import pandas as pd
import numpy as np
from curvature import smoothness
from shapely.geometry import MultiPoint, Polygon
from descartes import PolygonPatch
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from pathlib import PurePath
import os
import pylab

class plot_dynamic:
    def __init__(self, SD = False, ext='png', 
                 dpi=500, save=False, plt_style='seaborn', alpha=1.5,
                 folder='Figures', axs_size=None, fig_size=[5,7]):
        
        """
        Parameters
        ----------
        SD : bool, optional
            DESCRIPTION. The default is False.
        ext : str, optional
            DESCRIPTION. The default is 'png'.
        dpi : int, optional
            DESCRIPTION. The default is 500.
        save : bool, optional
            DESCRIPTION. The default is False.
        plt_style : TYPE, optional
            DESCRIPTION. The default is 'seaborn'.
        alpha : TYPE, optional
            DESCRIPTION. The default is 1.5.

        Returns
        -------
        None.

        """
        self.sd = SD
        self.ext = ext
        self.dpi = dpi
        self.save = save
        self.alpha = alpha
        self.axs_size = axs_size
        self.fig_size = fig_size
        plt.style.use(plt_style)
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*5
        self.root_path = PurePath(os.getcwd())
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.save_folder =  self.root_path / folder
        
    
    def rc_params(self, proportion = 1, nrows=5, ncols=7):
        mpl.rcParams["text.usetex"] = True #To use latex
        if proportion >= 4:
            # Adjusting plot parameters for bigger figures
            mpl.rcParams['axes.titlesize'] = 4*proportion
            mpl.rcParams['axes.labelsize'] = 6*proportion
            # mpl.rcParams['lines.linewidth'] = proportion*0.3
            # mpl.rcParams['lines.markersize'] = 4*proportion
            mpl.rcParams['xtick.labelsize'] = 5*proportion
            mpl.rcParams['ytick.labelsize'] = 5*proportion
            mpl.rcParams['legend.fontsize'] = 7*proportion
            mpl.rcParams['legend.handlelength'] = proportion*3
        else:
            # Adjusting plot parameters for small figures
            mpl.rcParams['axes.titlesize'] = 4*proportion
            mpl.rcParams['axes.labelsize'] = 5*proportion
            # mpl.rcParams['lines.linewidth'] = proportion
            mpl.rcParams['lines.markersize'] = 4*proportion
            mpl.rcParams['xtick.labelsize'] = 4*proportion
            mpl.rcParams['ytick.labelsize'] = 4*proportion
            mpl.rcParams['legend.fontsize'] = 5*proportion
            mpl.rcParams['legend.handlelength'] = proportion / 2
            
        mpl.rcParams['figure.figsize'] = ncols, nrows
        #backgroud color
        mpl.rcParams['axes.facecolor'] = 'white'
    
    def get_mult_num(self, integer):
      """
      Function to generate the best combination among a number of plots
      From https://stackoverflow.com/questions/54556363/finding-two-integers-that-multiply-to-20-can-i-make-this-code-more-pythonic
      """
      given_comb = [1,1,2,2,3,3,4,4,5,5,6,6,7,7]
      self.given_comb = [(i, j) for i, j in list(combinations(given_comb,2)) \
              if i * j == integer]
      return self.given_comb
  
    def if_object(self, df_object, cols, rows):
        # Getting speed labels
        if cols is not None:
            columns_0 = df_object.labels_first_col[cols]
        else:
            columns_0 = df_object.labels_first_col
        # Getting positions
        if rows is not None:
            rows_0 = df_object.labels_first_rows[rows]
        else:
            rows_0 = df_object.labels_first_rows
        # Mean and SD
        columns_1 = df_object.labels_second_col
        #Gait cycle discretization
        rows_1 = df_object.labels_second_rows
        # Dataframe from the object
        df_= df_object.df_
        y_label = df_object.y_label
        cycle_title = df_object.cycle_title
        return columns_0, columns_1, rows_0, rows_1, df_, y_label, cycle_title
    
    def if_df(self, df_object, cols, rows):
        # Getting speed labels
        columns_0 = df_object.columns.get_level_values(0).unique()
        if cols is not None:
            columns_0 = columns_0[cols]
        # Getting positions
        rows_0 = df_object.index.get_level_values(0).unique()
        if rows is not None:
            rows_0 =  rows_0[rows]
        # Mean and SD
        columns_1 = df_object.columns.get_level_values(1).unique()
        #Gait cycle discretization
        rows_1 = df_object.index.get_level_values(1).unique()
        # Dataframe from the object
        df_= df_object
        cycle_title = df_object.index.names[1]
        y_label = ''
        return columns_0, columns_1, rows_0, rows_1, df_, y_label, cycle_title

        
      
    def gait_plot(self, df_object, cols=None, rows=None, title=False, legend=True,
                  show=True):
        """
        Parameters
        ----------
        df_object : object
            Object of the data that containts DFs, labels name, titles, options
        cols : int list
            Integer list with column numbers positions. 
        rows : TYPE
            Integer list with index numbers positions.
        title : str, optional
            Figure suptitle. The default is False.

        Returns
        -------
        fig : fig object
            fig object to combine later

        """
        # Cleaning the plots
        self.clear_plot_mem()
        if isinstance(df_object, pd.DataFrame): # <- if is an object
            columns_0, columns_1, rows_0, rows_1, df_, y_label, cycle_title = \
                self.if_df(df_object, cols, rows)
        else:
            columns_0, columns_1, rows_0, rows_1, df_, y_label, cycle_title = \
                self.if_object(df_object, cols, rows)
        if self.axs_size is not None:
            nrows, ncols = self.axs_size
        else:
            # Suitable distribution for plotting
            nrows, ncols = self.get_mult_num(len(rows_0))[-1]
        # Adjusting the plot settings
        self.rc_params(self.alpha, 8, 10) #Letter size
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                squeeze=False, figsize=self.fig_size) 
        fig.tight_layout(pad=2*self.alpha/ncols)
        count = 0
        for k, ax in np.ndenumerate(axs):
            for i in columns_0:
                if self.sd:
                    sd_min = df_[i,columns_1[0]][rows_0[count]].values
                    mean = df_[i,columns_1[1]][rows_0[count]].values
                    sd_max = df_[i,columns_1[2]][rows_0[count]].values
                    ax.fill_between(rows_1, sd_min, sd_max, alpha=0.2)
                else:
                    try:
                        mean = df_[i,columns_1[1]][rows_0[count]].values
                    except IndexError:
                        mean = df_[i,columns_1[0]][rows_0[count]].values
                ax.plot(rows_1, mean, '-')
                ax.set_xlabel(cycle_title)
                ax.set_ylabel('{} {}'.format(rows_0[count], y_label))
            count += 1
        if legend:
            fig.legend(columns_0, bbox_to_anchor=[0.5, 0.5], loc='center',
                       ncol=int(len(columns_0)/2), fancybox=True)
        #In case y label is not displyed vary this parameter
        plt.subplots_adjust(left=0.05)
        if title:  plt.suptitle('{} {}'.format(title, y_label))
        if show: plt.show()
        if self.save:
            self.save_fig(fig, title)

        return fig
    
    def save_fig(self, fig, title):
        os.chdir(self.save_folder)
        fig.savefig('{}.{}'.format(title, self.ext), 
                        format= self.ext, dpi=self.dpi)
        os.chdir(self.root_path)
        
    def clear_plot_mem(self):
        """
        In order to not accumulate memory data on every plot

        Returns
        -------
        None.

        """
        plt.cla()
        plt.clf()
        plt.close()
        
class plot_ankle_DJS(plot_dynamic):
    def __init__(self, SD = False, ext='png', 
                 dpi=500, save=False, plt_style='seaborn', 
                 alpha=1.5, sep=True, fig_size=[5,5], params=None):
        #Setting plot parameters
        # Cleaning the plots
        super().__init__(SD, ext, dpi, save, plt_style, alpha)
        self.sd = SD
        self.ext = ext
        self.dpi = dpi
        self.save = save
        self.alpha = alpha
        self.sep = sep
        self.idx = pd.IndexSlice
        self.fig_size = fig_size
        self.params ={'sharex':True, 'sharey':True, 'arr_size': int(self.alpha*3), 
                      'color_DJS': self.colors, 'color_symbols': self.colors,
                      'color_reg': self.colors, 'left_margin': 0.15,
                      'yticks': None,
                      'xticks': None,
                      'hide_labels':(False, False),
                      'alpha_prod': 0.3,
                      'alpha_absorb': 0.1,
                      'DJS_linewidth': 0.4,
                      'reg_linewidth': 0.8,
                      'sd_linewidth': 0.3*self.alpha/self.fig_size[0],
                      'grid':True,
                      'text': False,
                      'tp_labels' : {'I.C.':(3,3),'ERP':(2.5,2.5),
                     'LRP':(1.2,1.0),'DP':(1,1.1),'S':(1.2,1.1), 
                     'TS':(1,1)},
                      'instances': ['CP','ERP', 'LRP', 'DP', 'S']}
        
        if params is not None:
            self.params.update(params)
             
        plt.style.use(plt_style)
    
    def deg2rad(self, row_name):
    
        self.df_.loc[self.idx[row_name,:],:] = self.df_.loc[self.idx[row_name,:],
                                                    :].apply(np.deg2rad, axis=0)
    
    def rm_static_pos(self, row_name):
    
        self.df_.loc[self.idx[row_name,:],:] = self.df_.loc[self.idx[row_name,:],
                                                    :].apply(lambda x: x - x[0])
    def separared(self, rows):
        areas_prod = []  
        areas_abs = []
        direction = []
        for _ , self.ax in np.ndenumerate(self.axs):
            try:
                #This exception is for unpaired plots in order to get rid of empty axes
                if self.sd:
                    self.ang_mean = self.extract_data([rows[0], self.count, 1]).squeeze()
                    self.mom_mean = self.extract_data([rows[1], self.count, 1]).squeeze()
                    label = self.columns_first[self.count]
                else:
                    self.ang_mean = self.extract_data([rows[0], 0, self.count]).squeeze()
                    self.mom_mean = self.extract_data([rows[1], 0, self.count]).squeeze()
                    label = self.columns_second[self.count]
                if self.sd:
                    self.sd_plot(rows)
                if not self.params['grid']:
                    self.ax.grid(False)
                    self.ax.spines['right'].set_visible(False)
                    self.ax.spines['top'].set_visible(False)
                line_plot = self.ax.plot(self.ang_mean, self.mom_mean, 
                             color= self.params['color_DJS'][self.count],
                             label= label,
                             linewidth=self.params['DJS_linewidth'])
                if self.integrate:
                    _prod, _abs, _dir = self.areas_fun()
                    areas_abs.append(_abs)
                    areas_prod.append(_prod)
                    direction.append(_dir)
                if isinstance(self.TP, pd.DataFrame):
                    self.reg_lines()
                if self.ang_mean.shape[0] <= 300:
                    arr_space = 2
                else:
                    arr_space = 20
                self.add_arrow(line_plot, step=arr_space)
                if self.params['text']:
                    self.labels_inside()
                if not self.params['hide_labels'][1]:
                    #showing only 1 column y labels and last row x labels
                    if self.count % self.sep[1] == 0:
                        self.ax.set_ylabel(self.y_label)
                if not self.params['hide_labels'][0]:
                    if self.count >= (self.sep[0]-1)*self.sep[1]:
                        self.ax.set_xlabel(self.x_label)
                if self.legend:
                    self.ax.legend(ncol=int(len(self.columns_first)/2), fancybox=True,
                                   loc = 'upper left')                       
                self.count +=1
            except ValueError:
                #Because plots does not match
                continue
        if self.integrate:
            if self.integrate: self.df_areas(areas_abs, areas_prod, direction)

    def together(self, rows):
        areas_prod = []  
        areas_abs = []
        direction = []
        for _ in enumerate(self.columns_first):
            if self.sd:
                self.ang_mean = self.extract_data([rows[0], self.count, 1]).squeeze()
                self.mom_mean = self.extract_data([rows[1], self.count, 1]).squeeze()
            else:
                self.ang_mean = self.extract_data([rows[0], self.count, self.count]).squeeze()
                self.mom_mean = self.extract_data([rows[1], self.count, self.count]).squeeze()
            if self.sd:
                self.sd_plot(rows)
            if not self.params['grid']:
                self.ax.grid(False)
                self.ax.spines['right'].set_visible(False)
                self.ax.spines['top'].set_visible(False)
            line_plot = self.ax.plot(self.ang_mean, self.mom_mean, 
                         color= self.params['color_DJS'][self.count],
                         label= self.columns_first[self.count],
                         linewidth=self.params['DJS_linewidth'])
            if self.integrate:
                _prod, _abs, _dir = self.areas_fun()
                areas_abs.append(_abs)
                areas_prod.append(_prod)
                direction.append(_dir)
            if isinstance(self.TP, pd.DataFrame):
                self.reg_lines()
            if self.ang_mean.shape[0] <= 400:
                arr_space = 2
            else:
                arr_space = 20
            self.add_arrow(line_plot, step=arr_space)
            if self.params['text'] and self.TP is not None:
                self.labels_inside()
            if not self.params['hide_labels'][0]:
                self.ax.set_xlabel(self.x_label)
            if not self.params['hide_labels'][1]:
                self.ax.set_ylabel(self.y_label)
            if self.legend == True:
                try:
                    self.ax.legend(ncol=1, fancybox=True, #int(len(self.columns_first)/2)
                               loc = 'upper left')
                except ZeroDivisionError:
                    self.ax.legend(ncol=1, fancybox=True, 
                               loc = 'upper left', fontsize=self.alpha*3)
                        
            elif self.legend == 'sep':
                figLegend = pylab.figure(figsize = self.fig_size)
                pylab.figlegend(*self.ax.get_legend_handles_labels(), 
                                loc = 'upper left')
                
                self.save_fig(figLegend, "legend_{}".format(self.title))
            else:
                pass

            self.count +=1
        if self.integrate: self.df_areas(areas_abs, areas_prod, direction)

    def labels_inside(self):
        #Printing point labels

        for n_, (tp_lab, val) in enumerate(self.params['tp_labels'].items()):
            self.ax.text(self.ang_mean[self.TP.iloc[self.count, n_]]*val[0],
                         self.mom_mean[self.TP.iloc[self.count, n_]]*val[1],
                         tp_lab,
                         color=self.params['color_reg'][self.count])
            self.ax.text(0.5, 0.3, r'$W_{net}$', horizontalalignment='center',
                         verticalalignment='center', 
                         color=self.params['color_reg'][self.count],
                         transform=self.ax.transAxes)   
            self.ax.text(0.85, 0.2, r'$W_{abs}$', horizontalalignment='center',
                         verticalalignment='center', transform=self.ax.transAxes,
                         color=self.params['color_reg'][self.count]) 


    def df_areas(self, areas_abs, areas_prod, direction):
        
        """
        Returns areas in a df way

        Returns
        -------
        None.

        """
        try:
            if self.sd:
                ind_ = self.columns_first
            else:
                ind_ = self.columns_second
            self.areas = pd.DataFrame(np.array([areas_abs, areas_prod, direction]).T, 
                                      columns = ['work abs', 'work prod', 'direction'], index=ind_)
        except ValueError:
            self.areas = pd.DataFrame(np.array([areas_abs, areas_prod, direction]).T, 
                                      columns = ['work abs', 'work prod', 'direction'], 
                                      index=self.reg_info_df.index.get_level_values(1).unique())
    def is_positive(self):
        signedarea = 0
        for len_arr in range(self.ang_mean.shape[0]-1):
           signedarea += self.ang_mean[len_arr]*self.mom_mean[len_arr+1] - \
               self.ang_mean[len_arr+1]*self.mom_mean[len_arr]
        if signedarea < 0:
            return 'cw'
        else:
            return 'ccw'
                
    def areas_fun(self):
        #We are making the integration of the closed loop
        try: 
            prod = self.integration(self.ang_mean, self.mom_mean, 
                             self.params['color_DJS'][self.count],
                             alpha = self.params['alpha_prod'])
            #To discover which direction the DJS is turning
            # If angle at 40% of the gait is bigger than angle in 65% of the gait
            # and if the moment at 55 % is bigger than 40%, so It is counter clockwise
            # otherwise clockwise
            len_vars = self.ang_mean.shape[0]
            try:
                #Taking the 10 highest values
                max_vals = self.ang_mean.argsort()[-15:].values
                # Proving that higher moments are generated in max values >0.25 Nm/kg
                max_ang = max_vals[self.mom_mean[max_vals].values > 0.25][-1]
            except IndexError:
                #In few cases no max ang is found, setting and average
                max_ang = int(len_vars*0.5)
            
            direction = self.is_positive()
            #We pretend to integer the area under the loop
            if direction == 'ccw':
                X = self.ang_mean[0:max_ang].values
                Y = self.mom_mean[0:max_ang].values
                pos_init = [-1,0,0,0]
            else:
                X = self.ang_mean[max_ang:].values
                Y = self.mom_mean[max_ang:].values
                pos_init = [0,-1,0,0]
            #closing the loop with another point in zero
            # going to 0 in Y axis and Adding another point in (0,0)
            X = np.append(X,[X[pos_init[0]],X[pos_init[2]]])
            Y = np.append(Y,[Y[pos_init[1]],Y[pos_init[3]]])
            absorb = self.integration(X,Y, 
                                      self.params['color_DJS'][self.count], 
                                      alpha=self.params['alpha_absorb'])
        except ValueError:
            #Integration cannot be done
            prod = 0
            absorb = 0
            direction = 0
            
                
        return prod, absorb, direction
        
    
    def reg_lines(self):
        # Plotting TP
        self.ax.scatter(self.ang_mean[self.TP.iloc[self.count]],
                        self.mom_mean[self.TP.iloc[self.count]],
                        color=self.params['color_symbols'][self.count],
                        linewidth = self.alpha/3)

        for i in range(self.TP.shape[1]-2):
            ang_data = self.ang_mean[self.TP.iloc[self.count][i]: \
                          self.TP.iloc[self.count][i+1]].values.reshape(-1,1)
            mom_data = self.mom_mean[self.TP.iloc[self.count][i]: \
                          self.TP.iloc[self.count][i+1]].values.reshape(-1,1)
            # print(ang_data, mom_data, self.TP.iloc[self.count])
            info = self.ridge(ang_data, mom_data)
            
            if i == 0:
                info2 = info
            else:
                for key, item in info.items():
                    info2[key].extend(item)
        reg_info = info2
        self.reg_data = reg_info.pop('pred_data')
        #We are placing the second level, so if there is only one item we need to take the 0 item 
        #otherwise the first
        if self.columns_second.shape[0] == 3:
            num = 1
        else:
            num = 0

        reg_info_df = pd.DataFrame(reg_info)
        instance = self.params['instances']
        instance = instance[:reg_info_df.shape[0]]
        if self.sd:
            reg_idx= pd.MultiIndex.from_product([[self.columns_first[self.count]], [self.columns_second[num]],
                             instance], names=['Speed', 'instance','QS phase'])
        else:
            reg_idx= pd.MultiIndex.from_product([self.columns_first, [self.columns_second[self.count]],
                             instance], names=['Speed', 'instance','QS phase'])
        reg_info_df.index = reg_idx

        if hasattr(self, 'reg_info_df'):
            self.reg_info_df = pd.concat([self.reg_info_df, reg_info_df])
        else:
            self.reg_info_df = reg_info_df
        style= ['--', '-.', ':', '--', '-.', ':']
        for i, reg in enumerate(self.reg_data):
            if self.params['text']:
                self.ax.plot(reg[:,0], reg[:,1], 
                                 color = self.params['color_reg'][self.count], 
                                 linestyle = style[i], zorder=15, 
                                 linewidth = self.params['reg_linewidth'],
                                 label=instance[i])
            else:
                self.ax.plot(reg[:,0], reg[:,1], 
                                color = self.params['color_reg'][self.count], 
                                linestyle = style[i], zorder=15, #style[i] -> for plot in paper
                                linewidth = self.params['reg_linewidth'])

    # Ridge regression
    def ridge(self, var1, var2, alpha = 0.001):
        """Function to do Ridge regression
        
        when the slope is so high, it is better to do regression in the opposite 
        side, it means, normally we predict moment through angles, when the bounds 
        have 0.03 radians of range we are predicting the angles through moments.
        
        """
        if var1[-1]-var1[0] < 0.03 and var1[-1]-var1[0] > -0.03:
            X = var2
            Y = var1
            inverted = True
        else:
            X = var1
            Y= var2
            inverted = False
            
        y_linear_lr = linear_model.Ridge(alpha= alpha)
        y_linear_lr.fit(X, Y)
        pred = y_linear_lr.predict(X)
        # R2 = y_linear_lr.score(X, Y)
        SS_Residual = np.sum((Y-pred)**2)       
        SS_Total = np.sum((Y-np.mean(Y))**2)     
        R2 = 1 - (float(SS_Residual))/SS_Total
        
        if inverted == False:
            pred_mod = (X, pred)
        else:
            pred_mod = (pred, X)
        meanSquare = mean_squared_error(Y, pred)
        return {'intercept': [y_linear_lr.intercept_.item(0)], 
                'stiffness': [y_linear_lr.coef_.item(0)], 
                'MSE':[meanSquare], 
                'R2':[R2],
                'pred_data': [np.hstack(pred_mod)],
                'inverted': [inverted]}
    
    def linear_fun(self,a,b,x):
        return a*x+b
    
    def add_reg_lines(self, pred_df, label='Predicted'):
        pred_df_ind = pred_df.index.get_level_values(0)
        for i, phase in enumerate(self.params['instances'][:-1]):
            stiffness = pred_df.loc[self.idx[pred_df_ind[self.count], phase]][1]
            intercept = pred_df.loc[self.idx[pred_df_ind[self.count], phase]][0]
            ang_data = self.ang_mean[self.TP.iloc[self.count][i]: \
                          self.TP.iloc[self.count][i+1]].values.reshape(-1,1)
            pred_data = self.linear_fun(stiffness, 
                                        intercept, 
                                        ang_data)
            self.ax.plot(ang_data, pred_data, 
                             color= self.params['color_reg'][self.count+1], 
                             linestyle = 'dashdot', label=label)
        return pred_data
        
    
            
    def plot_DJS(self, df_, cols=None, rows= [0,2],
                 title='No name given', legend=True, reg=None,
                 integration= True, rad= True, sup_static= True, header=None):
        self.clear_plot_mem()
        # Suitable distribution for plotting
        self.TP = reg
        self.legend = legend
        self.header = header
        self.integrate = integration
        self.index_first = df_.index.get_level_values(0).unique()
        self.index_second = df_.index.get_level_values(1).unique()
        self.columns_first = df_.columns.get_level_values(0).unique()
        self.columns_second = df_.columns.get_level_values(1).unique()
        self.y_label = 'Moment '+ r'$[\frac{Nm}{kg}]$'
        self.x_label = 'Angle [deg]'
        self.title = title
        if cols is None:
            cols = self.columns_first
            self.df_ = df_.loc[:,self.idx[self.columns_first,:]]
        else:
            self.df_ = df_.loc[:,self.idx[self.columns_first[cols],:]]
            #To keep the index column order
            self.df_ = self.df_.reindex(self.columns_first[cols], level=0, axis=1)
            #To keep the order of the TP
            if self.TP is not None:
                #check if always you will have at least two levels
                #If so this is ok
                self.TP = self.TP.reindex(self.columns_first[cols], axis=0, level=-2)
        if rad:
            self.deg2rad(self.index_first[rows[0]])
            self.x_label = 'Angle [rad]'
        if sup_static:
            self.rm_static_pos(self.index_first[rows[0]])
                        
        if rows is None:
            rows = self.index_first
        
        self.columns_first = self.df_.columns.get_level_values(0).unique()
        if self.sep == True:
            nrows, ncols = self.get_mult_num(len(cols))[-1]            
        elif self.sep == False: 
            nrows = 1
            ncols = 1
        elif isinstance(self.sep , list):
            nrows, ncols = self.sep
            
        # Adjusting the plot settings
        self.rc_params(self.alpha/nrows, self.fig_size[0], self.fig_size[1])
        self.count = 0
        if self.sep:
            self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, 
                                          sharex=False, squeeze=False)
            self.fig.tight_layout()
            self.separared(rows) 
        else:
            self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=self.params['sharey'], 
                                          sharex=self.params['sharex'], squeeze=True)
            self.together(rows)
        if self.header is not None:
            self.fig.suptitle(self.header)
        #In case y label is not displyed vary this parameter
        plt.subplots_adjust(left=self.params['left_margin'])
        #Ticks
        if self.params['yticks'] is not None:
            plt.yticks(self.params['yticks'])
            plt.ylim((self.params['yticks'][0], self.params['yticks'][-1]))
        if self.params['xticks'] is not None:   
            plt.xticks(self.params['xticks'])
            plt.xlim((self.params['xticks'][0], self.params['xticks'][-1]))
        if self.save:
            self.save_fig(self.fig, self.title)
        
        #Setting margins of figure
        return self.fig

    
    def sd_plot(self, rows):
        """
        Generates the plot for SD either for rows and columns

        Parameters
        ----------
        rows : List with rows indexes for angles and moments

        Returns
        -------
        None.

        """
        self.ang_sd1 = self.extract_data([rows[0], self.count, 0])
        self.ang_sd2 = self.extract_data([rows[0], self.count, 2])
        self.mom_sd1 = self.extract_data([rows[1], self.count, 0])
        self.mom_sd2 = self.extract_data([rows[1], self.count, 2])
        self.err_ang = np.abs([self.ang_mean - self.ang_sd1, 
                           self.ang_sd2 - self.ang_mean])
        self.err_mom = np.abs([self.mom_mean - self.mom_sd1, 
                           self.mom_sd2 - self.mom_mean])
        # print(f"Printing Error angle: {self.err_ang}")
        self.ax.errorbar(self.ang_mean, self.mom_mean, xerr=self.err_ang,
                         color= self.params['color_DJS'][self.count],
                         elinewidth = self.params['sd_linewidth'])
        self.ax.errorbar(self.ang_mean, self.mom_mean, yerr=self.err_mom, 
                         color= self.params['color_DJS'][self.count],
                         elinewidth = self.params['sd_linewidth'])
        
        return
        
    def extract_data(self, idx_):
        
        """
        Extract the specific feature information
        Parameters
        ----------
        idx : list with three items
            The first specifies the first index position.
            The second specifies the first column position.
            The third specifies the second column position
        Returns
        -------
        data : TYPE
            DESCRIPTION.
        """
        
        data = self.df_.loc[self.idx[self.index_first[idx_[0]] , :], 
                                    self.idx[self.columns_first[idx_[1]], 
                                        self.columns_second[idx_[2]]]]
        #print(data.name)
        return data
    
    def add_arrow(self, axs, direction='right', step=2):
        
        """
        
        Add an arrow to a line.
    
        line:       Line2D object
        position:   x-position of the arrow. If None, mean of xdata is taken
        direction:  'left' or 'right'
        size:       size of the arrow in fontsize points
        """
        #How often will be repeated
        arr_num = self.params['arr_size']*step
        axs = axs[0]
        color = self.params['color_symbols'][self.count]
    
        xdata = axs.get_xdata()
        ydata = axs.get_ydata()
        
        #Selecting the positions
        position = xdata[0:-20:arr_num]
        # find closest index
        start_ind = self.ang_mean[5:-1:arr_num].index.get_level_values(1)
        if direction == 'right':
            # Be careful when there is no index with decimals
            end_ind = self.ang_mean[6::arr_num].index.get_level_values(1)
        else:
            end_ind = self.ang_mean[0:-1:arr_num].index.get_level_values(1)
        for n, ind in enumerate(start_ind):
            axs.axes.annotate('',
                xytext=(self.ang_mean.loc[self.idx[:,ind]], 
                        self.mom_mean.loc[self.idx[:,ind]]),
                xy=(self.ang_mean.loc[self.idx[:,end_ind[n]]], 
                    self.mom_mean.loc[self.idx[:,end_ind[n]]]),
                arrowprops=dict(arrowstyle="-|>", color=color),
                size=self.params['arr_size'])
    
    def integration(self, var1, var2, color, alpha=0.3): #, dx= 0.5, Min = 0, Max = None):
        
        """
        integration based on two variables with shapely

        Parameters
        ----------
        var1 (angle data) : DataFrame, Array containing the values for angles.
        var2 (moment data) : DataFrame, Array containing the values for Moment.

        Returns
        -------
        FLOAT
            Integral under the curve

        """
        # Making pairs
        list_area = list(zip(var1, var2))
        multi_point = MultiPoint(list_area)
        # print(f"list area: {multi_point}")
        poly2 = Polygon([[p.x, p.y] for p in multi_point])
        x,y = poly2.exterior.xy
        poly1patch = PolygonPatch(poly2, fc= color, ec=color, 
                                  alpha=alpha, zorder=2)
        self.ax.add_patch(poly1patch)
        return poly2.area
    

                       



        
        
        
    
    