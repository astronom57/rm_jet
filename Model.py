#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:13:23 2021

@author: mlisakov
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0,'..')
import mylogger


class Model():
    """Handles modelfit models. 
    
    clean models are not handled for now but could be added later. 
    """
    
    def __init__(self):
        """Initialize a model
        
        Attributes:
            data: 
                pandas DataFrame containing the model itself. Columns= 
                [flux, radius, theta, major, axratio, phi, freq, spix]
                All columns ending with v are boolean flags if the parameter is variable
                (True) or not (False).
                (x,y) - raduis and theta converted to ra and dec offsets from the image center.
                Index = comp number.
            
            origin ([float]):
                coordinates of the origin of the model [mas, mas]. If the model is shifted,
                the origin is changed accordingly.
        """
        
        
        self.data = pd.DataFrame(columns=['flux', 'fluxv', 
                                          'radius', 'radiusv',
                                          'theta', 'thetav',
                                          'x', 'y',
                                          'major', 'majorv',
                                          'axratio', 'axratiov',
                                          'phi', 'phiv',
                                          'T', 'freq', 'spix'])
        
        self.origin = np.array([0.0, 0.0])


    def read_model(self, file):
        """Read model from file
        
        Args:
            file:
                file with a model
        
        Returns:
            self.data:
                Dataframe with the model
        """
        
        self.data = pd.read_csv(file, names=['flux', 'radius', 'theta', 'major',
                                             'axratio', 'phi', 'T', 'freq', 'spix'],
                                index_col=False, sep='\s+',
                                comment='!')
        
        # set flags
        for col in ['flux', 'radius', 'theta', 'major', 'axratio', 'phi']:
            self.data.loc[:, col] = self.data.loc[:, col].astype(str)
            self.data.loc[:, '{}v'.format(col)] = self.data.loc[:, col].str.endswith('v') 
            self.data.loc[:, col] = self.data.loc[:, col].str.replace('v','').astype(np.float)
            
            
        # calculate x,y
        self.data.loc[:, 'x'] = self.data.loc[:, 'radius'] * np.sin(np.deg2rad(self.data.loc[:, 'theta']))
        self.data.loc[:, 'y'] = self.data.loc[:, 'radius'] * np.cos(np.deg2rad(self.data.loc[:, 'theta']))
        
        return self.data
        
    def shift_model(self, dx,dy):
        """Shift all model component positions by dx [mas] in ra and dy [mas] in dec.
        
        Args:
            dx (float):
                shift in the direction of RA
            dy (float):
                shift in the direction of Dec
            
        """
        
        logger.warning('Before shifting model')
        logger.warning('origin = {}'.format(self.origin))
        logger.warning('x = \n{}'.format(self.data.x))
        logger.warning('y = \n{}'.format(self.data.y))
        
        self.origin += [dx, dy]
        
        self.data.loc[:, 'x'] = self.data.loc[:, 'x'] + dx 
        self.data.loc[:, 'y'] = self.data.loc[:, 'y'] + dy 
        
        self.data.loc[:, 'radius'] = np.sqrt(self.data.loc[:, 'x']**2 + self.data.loc[:, 'y']**2)
        self.data.loc[:, 'theta'] = np.rad2deg(np.arctan2(self.data.loc[:, 'x'], self.data.loc[:, 'y']))
        
        logger.warning('AFTER shifting model')
        logger.warning('origin = {}'.format(self.origin))
        logger.warning('x = \n{}'.format(self.data.x))
        logger.warning('y = \n{}'.format(self.data.y))
        
        
        
        
    def core_origin(self):
        """Switch origin to the core. Core should be the first listed component in the model
        
        """
        dx = -self.data.loc[0, 'x']
        dy = -self.data.loc[0, 'y']
        self.shift_model(dx, dy)
