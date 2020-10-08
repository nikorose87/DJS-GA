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

class plot_dynamic:
    def __init__(self, SD = False, ext='png', 
                 dpi=500, save=False, plt_style='seaborn', alpha=1.5,
                 folder='Figures'):
        
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
        plt.style.use(plt_style)
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*100
        self.root_path = PurePath(os.getcwd())
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.save_folder =  self.root_path / folder
        
    
    def rc_params(self, proportion = 1):
        # Adjusting plot parameters
        mpl.rcParams['axes.titlesize'] = 5*proportion
        mpl.rcParams['axes.labelsize'] = 4*proportion
        mpl.rcParams['lines.linewidth'] = proportion
        mpl.rcParams['lines.markersize'] = 4*proportion
        mpl.rcParams['xtick.labelsize'] = 3*proportion
        mpl.rcParams['ytick.labelsize'] = 3*proportion
        mpl.rcParams['legend.fontsize'] = 3*proportion
        mpl.rcParams['legend.handlelength'] = proportion
        
 
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

        # Suitable distribution for plotting
        nrows, ncols = self.get_mult_num(len(rows_0))[-1]
        # Adjusting the plot settings
        self.rc_params(self.alpha/nrows)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, 
                                squeeze=False) 
        fig.tight_layout(pad=3*self.alpha/ncols)
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
                 alpha=1.5, sep=True):
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
        plt.style.use(plt_style)
    
    def deg2rad(self, row_name):
    
        self.df_.loc[self.idx[row_name,:],:] = self.df_.loc[self.idx[row_name,:],
                                                    :].apply(np.deg2rad, axis=0)
    
    def rm_static_pos(self, row_name):
    
        self.df_.loc[self.idx[row_name,:],:] = self.df_.loc[self.idx[row_name,:],
                                                    :].apply(lambda x: x - x[0])
    def separared(self, rows):
        areas = []
        for _ , self.ax in np.ndenumerate(self.axs):
            self.ang_mean = self.extract_data([rows[0], self.count, self.sd])
            self.mom_mean = self.extract_data([rows[1], self.count, self.sd])
            if self.sd:
                self.sd_plot(rows)
            line_plot = self.ax.plot(self.ang_mean, self.mom_mean, 
                         color= self.colors[self.count],
                         label= self.columns_first[self.count])
            if self.integrate:
                areas.append(self.integration(self.ang_mean, self.mom_mean, 
                                 self.colors[self.count]))
            if isinstance(self.TP, pd.DataFrame):
                self.reg_lines()
                
            self.add_arrow(line_plot)
            self.ax.set_xlabel(self.x_label)
            self.ax.set_ylabel(self.y_label)
            if self.legend:
                self.ax.legend(ncol=int(len(self.columns_first)/2), fancybox=True,
                               loc = 'upper left')
            self.count +=1
        self.areas = pd.DataFrame(areas, columns = ['work'], index=self.columns_first)
            
    def together(self, rows):
        areas = []
        for _ in enumerate(self.columns_first):
            self.ang_mean = self.extract_data([rows[0], self.count, int(self.sd)])
            self.mom_mean = self.extract_data([rows[1], self.count, int(self.sd)])
            if self.sd:
                self.sd_plot(rows)
            line_plot = self.ax.plot(self.ang_mean, self.mom_mean, 
                         color= self.colors[self.count],
                         label= self.columns_first[self.count])
            if self.integrate:
                areas.append(self.integration(self.ang_mean, self.mom_mean, 
                                 self.colors[self.count]))
            if isinstance(self.TP, pd.DataFrame):
                self.reg_lines()
            self.add_arrow(line_plot)
            self.ax.set_xlabel(self.x_label)
            self.ax.set_ylabel(self.y_label)
            if self.legend:
                self.ax.legend(ncol=int(len(self.columns_first)/2), fancybox=True,
                           loc = 'upper left')
        
            self.count +=1
        self.areas = pd.DataFrame(areas, columns = ['work'], index=self.columns_first)
    
    def reg_lines(self):
        self.ax.scatter(self.ang_mean[self.TP.iloc[self.count][:-1]],
                        self.mom_mean[self.TP.iloc[self.count][:-1]],
                        color=self.colors[self.count])

        for i in range(self.TP.shape[1]-2):
            ang_data = self.ang_mean[self.TP.iloc[self.count][i]: \
                          self.TP.iloc[self.count][i+1]].values.reshape(-1,1)
            mom_data = self.mom_mean[self.TP.iloc[self.count][i]: \
                          self.TP.iloc[self.count][i+1]].values.reshape(-1,1)
            info = self.ridge(ang_data, mom_data)
            if i == 0:
                info2 = info
            else:
                for key, item in info.items():
                    info2[key].extend(item)
        reg_info = info2
        self.reg_data = reg_info.pop('pred_data')
        reg_idx= pd.MultiIndex.from_product([[self.columns_first[self.count]], 
                                             ['ERP', 'LRP', 'DP']], 
                                                  names=['Speed', 'QS phase'])
        reg_info_df = pd.DataFrame(reg_info, index = reg_idx)
        if hasattr(self, 'reg_info_df'):
            self.reg_info_df = pd.concat([self.reg_info_df, reg_info_df])
        else:
            self.reg_info_df = reg_info_df
            
                                            
        for reg in self.reg_data:
            self.ax.plot(reg[:,0], reg[:,1], 
                             color= self.colors[self.count], 
                             linestyle = 'dashed')
                             # label= '{} QS'.format(self.columns_first[self.count]))
            
        
    
    # Ridge regression
    def ridge(self, var1, var2, alpha = 0.001):
        """Function to do Ridge regression"""
        y_linear_lr = linear_model.Ridge(alpha= alpha)
        y_linear_lr.fit(var1, var2)
        R2 = y_linear_lr.score(var1, var2)
        pred = y_linear_lr.predict(var1)
        meanSquare = mean_squared_error(var2, pred)
        return {'intercept': [y_linear_lr.intercept_.item(0)], 
                'stiffness': [y_linear_lr.coef_.item(0)], 
                'MSE':[meanSquare], 
                'R2':[R2],
                'pred_data': [np.hstack((var1, pred))]}
    
    
            
    def plot_DJS(self, df_, cols=None, rows= [0,2],
                 title=False, legend=True, reg=False,
                 integration= True, rad= True, sup_static= True):
        self.clear_plot_mem()
        # Suitable distribution for plotting
        self.TP = reg
        self.legend = legend
        self.integrate = integration
        self.index_first = df_.index.get_level_values(0).unique()
        self.index_second = df_.index.get_level_values(1).unique()
        self.columns_first = df_.columns.get_level_values(0).unique()
        self.columns_second = df_.columns.get_level_values(1).unique()
        self.y_label = 'Moment '+r'$[\frac{Nm}{kg}]$'
        self.x_label = 'Angle [deg]'
        if cols is None:
            cols = self.columns_first
            self.df_ =  df_.loc[self.idx[:,:],self.idx[self.columns_first,:]]
        else:
            self.df_ =  df_.loc[self.idx[:,:],self.idx[self.columns_first[cols],:]]
        if rad:
            self.deg2rad(self.index_first[rows[0]])
            self.x_label = 'Angle [rad]'
        if sup_static:
            self.rm_static_pos(self.index_first[rows[0]])
                        
        if rows is None:
            rows = self.index_first
        
        self.columns_first = self.df_.columns.get_level_values(0).unique()
        if self.sep:
            nrows, ncols = self.get_mult_num(len(cols))[-1]
        else: 
            nrows = 1
            ncols = 1
            
        # Adjusting the plot settings
        self.rc_params(self.alpha/nrows)
        
        # Adjusting the plot settings
        self.rc_params(self.alpha/nrows)
        self.count = 0
        if self.sep:
            self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, 
                                          sharex=True, squeeze=False)
            self.fig.tight_layout(pad=3*self.alpha/ncols)
            self.separared(rows) 
        else:
            self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, 
                                          sharex=True, squeeze=True)
            self.together(rows)
        if title: 
            self.fig.suptitle(title)
        if self.save:
            self.save_fig(self.fig, title)
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
        self.err_ang = [self.ang_mean - self.ang_sd1, 
                           self.ang_sd2 - self.ang_mean]
        self.err_mom = [self.mom_mean - self.mom_sd1, 
                           self.mom_sd2 - self.mom_mean]
        self.ax.errorbar(self.ang_mean, self.mom_mean, xerr=self.err_ang,
                         color= self.colors[self.count],
                         elinewidth = 0.1*self.alpha)
        self.ax.errorbar(self.ang_mean, self.mom_mean, yerr=self.err_mom, 
                         color= self.colors[self.count],
                         elinewidth = 0.1*self.alpha)
        
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
        return data
    
    def add_arrow(self, axs, arr_num=8, direction='right'):
        """
        Add an arrow to a line.
    
        line:       Line2D object
        position:   x-position of the arrow. If None, mean of xdata is taken
        direction:  'left' or 'right'
        size:       size of the arrow in fontsize points
        color:      if None, line color is taken.
        """
        axs = axs[0]
        color = axs.get_color()
    
        xdata = axs.get_xdata()
        ydata = axs.get_ydata()

        #Selecting the positions
        position = xdata[0:-1:arr_num]
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
                size=self.alpha*8)
    
    def integration(self, var1, var2, color): #, dx= 0.5, Min = 0, Max = None):
        
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
        poly2 = Polygon([[p.x, p.y] for p in multi_point])
        x,y = poly2.exterior.xy
        poly1patch = PolygonPatch(poly2, fc= color, ec=color, 
                                  alpha=0.3, zorder=2)
        self.ax.add_patch(poly1patch)
        return poly2.area
    

                       



        
        
        
    
    