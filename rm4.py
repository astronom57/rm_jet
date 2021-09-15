#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-08-25

DISCLAIMER: This is a newer, more general, and more usable version of the rm3.py 
script that was used for the "Overzised sheath..." paper. 
All specific slices, their coordinates, etc. can be found in the rm3.py script

Script to analyze Rotation Measure maps. The most common use case is to make slices 
across or along the jet direction. It is possible both interactively (fast) and
by providing coordinates of the slice (repeatable).


@author: mikhail
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) #
    # FutureWarning: Support for multi-dimensional indexing (e.g. `obj[:, None]`) 
    # is deprecated and will be removed in a future version.  Convert to a numpy 
    # array before indexing instead. x = x[:, np.newaxis]


warnings.filterwarnings("ignore", category=RuntimeWarning) 
    # RuntimeWarning: invalid value encountered in less
    # condition = (xf < v1) | (xf > v2)

import os
#print(os.environ['HOME'])
import logging

import platform  # to get computer name in order to avoid selecting computer name and paths by hand
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from astropy import wcs
from astropy.io import fits
import os.path
import os
import re
import astropy
from astropy.wcs import WCS
from astropy.wcs import utils as UTILS
from astropy.visualization.wcsaxes import WCSAxes
from matplotlib import collections as mc
import matplotlib.patheffects as path_effects
import astropy.units as u
from astropy.units import cds as C
from matplotlib.ticker import LinearLocator, NullFormatter, ScalarFormatter, FormatStrFormatter, MultipleLocator, AutoMinorLocator
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

import datetime as dt

#from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import label, find_objects, gaussian_filter, mean
from scipy import ndimage

#from rm2lib import *
from rm2lib import loglevs, limits_mas2pix, make_transform, corner_std
from deconvolve import make_beam


class Slice():
    """Handles slices. 
    """
    
    def __init__(self):
        """Initialize a slice.
        
        Attributes:
            start ([float]):
                slice start coordinates in mas
            stop ([float]):
                slice stop coordinates in mas
            data (pandas dataframe):
                measurements along the slice. (x,y) - coordinates in mas,
                (v,verr) - value and error, (dd,d) - piecewise distance between points and
                distance along the slice in mas
            npoints (int):
                number of point in case of resampling
            
        """
        
        self.logger = create_logger('slice', dest=['STDERR'], levels=['INFO'])
        self.start = np.array([0,0])
        self.stop = np.array([0,0])
        self.data = pd.DataFrame(columns=['x', 'y', 'v', 'verr'])
        self.npoints = 100 # default
        
    def make_slice(self, fits, start, stop, npoints=100):
        """make a slice across the data
        
        Args:
            fits:
                HDUList fits object with a map
            start ([float]):
                slice start coordinates in mas
            stop ([float]):
                slice stop coordinates in mas
            npoints (int):
                number of points along the slice
        
        Returns:
            slice: 
                array with (x,y,v,verr) data. Also, self.data is modified in-place.
        """
        
        map2d = fits[0].data.squeeze()
        header = fits[0].header
        param = get_parameters(header)
        
        self.start = start # (x,y) in mas
        self.stop = stop   # (x,y) in mas
        
        # convert start and stop to pixels
        start_pix =( (self.start[0] / param.masperpix + param.racenpix.values[0]).astype(int),
                    (self.start[1] / param.masperpix + param.deccenpix.values[0]).astype(int))

        stop_pix =( (self.stop[0] / param.masperpix + param.racenpix.values[0]).astype(int),
                    (self.stop[1] / param.masperpix + param.deccenpix.values[0]).astype(int))
        
        x_pix = np.linspace(start_pix[0], stop_pix[0], npoints) # check if need to swap coordinates
        y_pix = np.linspace(start_pix[1], stop_pix[1], npoints) # check if need to swap coordinates
        
        self.data.x = np.linspace(start[0], stop[0], npoints) # same points in mas
        self.data.y = np.linspace(start[1], stop[1], npoints) # same points in mas
        self.data.v = slice_values = ndimage.map_coordinates(map2d, np.hstack((x_pix,y_pix)).T)        
        
        
        # distance along the slice
        self.data.loc[:, 'dx'] = self.data.x.diff()
        self.data.loc[:, 'dy'] = self.data.y.diff()
        self.data.loc[self.data.dx.isna(), 'dx'] = 0.0
        self.data.loc[self.data.dy.isna(), 'dy'] = 0.0
        self.data.loc[:, 'dd'] = np.sqrt((self.data.dx)**2 + (self.data.dy)**2)
        self.data.loc[:, 'd'] = self.data.dd.cumsum()
        
        return slice_values
        
    def plot_slice(self):
        """Make a separate plot with slice values wrt to the distance along the slice"""
        
        fig, ax = plt.subplots(1,1)
        ax.plot(self.data.d, self.data.v, 'o')
        ax.set_xlabel('Distance along the slice [mas]')
        ax.set_ylabel('Slice value')



class Ridgeline():
    """Handles ridge lines.
    Allows to find ridgeline coordinates from a map and apply these coordinates 
    to get values from another map. Also allows different manipulations with the 
    redgeline.
    
    Args:
        ridgeline_fits:
            astropy HDUList object with a map to build the ridgeline. Ridgeline is 
            made in pixels and then converted to [mas] using FITS header info.
        apply_to_fits:
            astropy HDUList object to apply ridgeline to and get values. If nothing 
            is supplied, ridgeline_fits is used. 
        
    Attributes:
        rr:
            ridgeline as a pandas dataframe. Important columns are: (x,y) - ridgeline
            coordinates [mas], distance - distance along the ridgeline [mas],
            (v, verr) - value and error, quality - boolean flag to use 
            this particular point or not. 
            
    Methods:
        TBA
        
    """
    
    
    def __init__(self, ridgeline_fits=None, apply_to_fits=None):
        """Initialize a ridgeline object. 
        
        Args:
            ridgeline_fits:
                astropy HDUList object with a map to build the ridgeline. Ridgeline is 
                made in pixels and then converted to [mas] using FITS header info.
            apply_to_fits:
                astropy HDUList object to apply ridgeline to and get values. If nothing 
                is supplied, ridgeline_fits is used. 
        
        """
        self.logger = create_logger('ridgeline', dest=['STDERR'], levels=['INFO'])

        # self.rr = pd.DataFrame(columns=['x', 'y', 'v', 'verr',  # ridgeline data + error
        #                                'xminus', 'yminus',  # coordinates of the previous point. Can be rewritten in order not to use this
        #                                'xplus', 'yplus',    # coordinates of the next point. Can be rewritten in order not to use this
        #                                'a', 'b',            # fit line coefficients: y = ax + b
        #                                'dir', 'perp',       # angle of the fit line and perpendicular one
        #                                'xslice1', 'yslice1',# terminal point 1 of the cross slice
        #                                'xslice2', 'yslice2',# terminal point 2 of the cross slice
        #                                'distance',          # distance along the ridgeline
        #                                'quality'])          # whether use this slice or not. Default True
        self.rr = pd.DataFrame(columns=['x', 'y', 'v', 'verr',  # ridgeline data + error
                                       'distance',          # distance along the ridgeline
                                       'quality'])          # whether use this slice or not. Default True


        



    def make_ridgeline(self, ridgeline_fits, steps=50, 
                       smooth=True, smooth_factor=1.5, 
                       method='center_of_mass',
                       snr_cutoff=20.0):
        """Get ridgeline coordinates given a map. Before making a ridgeline, the map
        is smoothed with a Gaussian, see smooth_map for details. 
        
        Args:
            ridgeline_fits:
                astropy HDUList object with a map to build the ridgeline. Ridgeline is 
                made in pixels and then converted to [mas] using FITS header info.
            method (str):
                center_of_mass - return the position of the center of mass within the label
                max - return position of the maximum within the label
            snr_cutoff (float):
                turn a point quality to False if it is derived with snr < snr_cutoff

        Returns:
            radial_max_pos ([[float]]):
                array with coordinates of the ridgeline in [pixels] ? (TODO: maybe change to mas? )
                Also, self.rr.[x,y] are filled with coordinates in [mas]
        """
        # save unsmoothed HDUList state
        map4d_init = ridgeline_fits[0].data
        header_init = ridgeline_fits[0].header
        param = get_parameters(header_init)
        MASperPIX = np.abs(param.rapixsize.values[0]*3.6e6)

        #smooth map
        if smooth is True:
            ridgeline_fits = smooth_map(ridgeline_fits, factor=smooth_factor, logger=self.logger)
                
        map2d = ridgeline_fits[0].data.squeeze()
        header = ridgeline_fits[0].header
        param = get_parameters(header)

               
    
        #based on http://scipy-lectures.org/advanced/image_processing/auto_examples/plot_radial_mean.html#sphx-glr-advanced-image-processing-auto-examples-plot-radial-mean-py
        
        sx, sy = map2d.shape
        X, Y = np.ogrid[0:sx, 0:sy]
        r = np.hypot(X - sx/2, Y - sy/2) # hypotenuse, i.e. simply radius
        rbin = (steps* r/r.max()).astype(np.int)
        # radial_max = np.array(ndimage.maximum(map2d, labels=rbin, index=np.arange(1, rbin.max() +1)))
        
        if method == 'center_of_mass':
            # center of mass
            radial_max_pos = np.array(ndimage.center_of_mass(map2d, labels=rbin, index=np.arange(1, rbin.max() )))
        else:
            # just find a maximum
            radial_max_pos = np.array(ndimage.maximum_position(map2d, labels=rbin, index=np.arange(1, rbin.max() )))

        # get SNR of each max
        
        # noise = np.array(ndimage.standard_deviation(map2d, labels=rbin, index=np.arange(1, rbin.max() )))
        noise = corner_std(map2d)
        radial_max_value = np.array(ndimage.maximum(map2d, labels=rbin, index=np.arange(1, rbin.max() )))
        
        self.logger.debug('RADIAL max values')
        self.logger.debug(radial_max_value)
        self.logger.debug('RADIAL noise')
        self.logger.debug(noise)
        
        
        self.rr.loc[:, 'snr'] = radial_max_value / noise
        # filter out low snr points
        self.rr.loc[self.rr.snr < snr_cutoff , 'quality'] = False
        
        
        
        # self.logger.debug('radial_max_pos = \n{}'.format(radial_max_pos))
        
        # radial_max_pos = radial_max_pos[radial_max_pos[:, 0] > 0]
        # radial_max_pos = radial_max_pos[radial_max_pos[:, 1] > 0]
        # radial_max_pos = radial_max_pos[radial_max_pos[:, 0] < param.decmapsize.values[0]]
        # radial_max_pos = radial_max_pos[radial_max_pos[:, 1] < param.ramapsize.values[0]]
        # self.logger.debug('radial_max_pos = \n{}'.format(radial_max_pos))

        
        self.logger.debug('MAKE: First 3 point of the ridgeline in pix: \n{}'.format(radial_max_pos[0:3, :]))
        
        self.rr.x = np.array( radial_max_pos[:, 1] - param.racenpix.values[0]) * MASperPIX
        self.rr.y = np.array( radial_max_pos[:, 0] - param.deccenpix.values[0]) * MASperPIX
        
        
        # drop everything with False quality
        self.rr.drop(self.rr.loc[self.rr.quality == False].index, axis=0, inplace=True)
        
        # return fits object to its initial state before smoothing
        ridgeline_fits[0].data = map4d_init
        ridgeline_fits[0].header = header_init
        
        return radial_max_pos
    
    def proceed_ridgeline(self, remove_jumps=True, factor=5.0):
        """Calculate distance along the ridgeline.
        Modifies self.rr inplace
        
        Args:
            factor (float):
                if any distance is higher than factor*median(), result is True.
        Returns:
            None
        """
        
        self.rr.loc[:, 'xminus'] = np.roll(self.rr.x, 1)
        self.rr.loc[:, 'yminus'] = np.roll(self.rr.y, 1)
        self.rr.loc[0, 'xminus'] = self.rr.loc[0, 'x']
        self.rr.loc[0, 'yminus'] = self.rr.loc[0, 'y']
        self.rr.loc[:, 'distance'] = np.sqrt((self.rr.x - self.rr.xminus)**2 + (self.rr.y - self.rr.yminus)**2) # distance to previous point 
        self.rr.loc[:, 'distance_piecewise'] = np.sqrt((self.rr.x - self.rr.xminus)**2 + (self.rr.y - self.rr.yminus)**2) # distance to previous point 

        self.logger.debug(self.rr.x)
        # self.logger.debug(self.rr.xminus)
        
        
        # remove points which are too far from the previous one
        median_distance = self.rr.distance.median()
        self.logger.info('Median distance between ridgeline points is {:.3f} mas'.format(median_distance))
        
        # self.logger.debug('rr.distance = {}'.format(self.rr.distance))
        
        
        # dealing with jumps.
        if remove_jumps:
            while self.has_jumps():
                self.logger.debug('before removing jumps')
                self.logger.debug(self.rr.distance_piecewise)
                self.rr.drop(self.rr.loc[self.rr.distance_piecewise.abs() > factor * median_distance].index, axis=0, inplace=True)
                self.rr.loc[:, 'xminus'] = np.roll(self.rr.x, 1)
                self.rr.loc[:, 'yminus'] = np.roll(self.rr.y, 1)
                self.rr.loc[0, 'xminus'] = self.rr.loc[0, 'x']
                self.rr.loc[0, 'yminus'] = self.rr.loc[0, 'y']
                self.rr.loc[:, 'distance_piecewise'] = np.sqrt((self.rr.x - self.rr.xminus)**2 + (self.rr.y - self.rr.yminus)**2) # distance to previous point 
                self.logger.debug('AFTER removing jumps')
                self.logger.debug(self.rr.distance_piecewise)

        
        
        
        
        self.rr.loc[:, 'distance'] = np.cumsum(self.rr.distance_piecewise.values) # distance from the 0-th point along the ridgeline 
        
        logger.debug('AFTER remowing all jumps, cumsum is')
        logger.debug(self.rr.distance)

        
        # self.logger.debug('AFTER cumsum: rr.distance.index.size = {}'.format(self.rr.index.size))
    
        # self.rr.loc[np.max(self.rr.index), 'quality'] = False
        
        
        
    
    def apply_ridgeline(self, apply_to_fits):
        """Take ridgeline coordinates and get map values at these points. 
        Useful when e.g. making the ridgeline on a Stokes I map and then getting 
        values from the RM map. Modifies self.rr[v, verr] inplace
    
        Args:
            fits:
                astropy HDUList object with the map
            error:
                astropy HDUList object with the error map
            
        Returns:
            ridgeline as a dataframe
        """

        data_map = apply_to_fits[0].data.squeeze()
        # get pixel2mas convertion
        param = get_parameters(apply_to_fits[0].header)        
        MASperPIX = np.abs(param.rapixsize.values[0]*3.6e6)
        y = (self.rr.x.values / MASperPIX + param.racenpix.values[0]).astype(int)
        x = (self.rr.y.values / MASperPIX + param.deccenpix.values[0]).astype(int)
        
        self.logger.debug('APPLY: First 3 point of the ridgeline in pix: \n{}'.format(np.array([x[0:3],y[0:3]]).T))

        self.rr.v = data_map[x, y]
        
        # if error is not None:
        #     error_map = error[0].data.squeeze()
        #     rr.verr = error_map[rr.x.values.astype(int), rr.y.values.astype(int)]
    
        return self.rr.loc[:, ['x', 'y', 'v']]
    

    def has_jumps(self, factor=5):
        """Look whether the ridgeline has any distance jumps exceeding factor*median(). 
        
        Agrs:
            factor (float):
                if any distance is higher than factor*median(), result is True.
        Returns:
            (boolean): True if the ridgeline has jumps. False otherwise
        """
        
        median_distance = self.rr.distance_piecewise.median()
        if any(self.rr.distance_piecewise.abs() > factor * median_distance):
            return True
        else:
            return False
        
        





def create_logger(obj=None, dest=['stderr'], levels=['INFO']):
    """Creates a logger instance to write log information to STDERR.
    
    Args:
        obj: caller object or function with a __name__ parameter
        
        dest (list, default is ['stderr']): a destination where to write 
        logging information. Possible values: 'stderr', 'filename'. Every 
        non-'stderr' string is treated as a filename
            
        levels (list, default is ['INFO']): logging levels. One value per dest 
            is allowed. A single value will be applied to all dest. 
    
    Returns:
        logging logger object
    
    Examples: 
        a logger for a class: 
            
        logger = create_logger(obj=self) # create a logger 
        logger.debug('debug message')
        logger.info('info message')
        logger.warning('warn message')
        logger.error('error message')
        logger.critical('critical message')
        
    """
    
    # initialize a named logger    
    try:
        logger = logging.getLogger(obj.__name__) # does not work yet. Do we need it at all? 
    except:
        logger = logging.getLogger('tsyslogger')
    
    # set the minimum logging level
    logger.setLevel('DEBUG')
    
    
    
    # solve the issue of multiple handlers appearing unexpectedly
    if (logger.hasHandlers()):
        logger.handlers.clear()
        logger.parent.handlers.clear()
        
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # match dimensions of dest and level
    if isinstance(dest, list):
        num_dest = len(dest)
    else:
        num_dest = 1
        dest = [dest]
    
    if isinstance(levels, list):
        num_levels = len(levels)
    else:
        num_levels = 1
        levels = [levels]
    
    if num_dest > num_levels:
        for i in np.arange(0, num_dest - num_levels):
            levels.append(levels[-1])
            
    if num_dest < num_levels:
        levels = levels[:len(num_dest)]

    # add all desired destinations with proper levels
    for i, d in enumerate(dest):
        
        if d.upper() in ['STDERR', 'ERR']:
            handler = logging.StreamHandler()   # stderr
        else:
            handler = logging.FileHandler(d, mode='w') # file. w-write, a-append
            
        level = levels[i]
        # set logging leveridgelinel
        if level.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            original_level = level
            level = 'INFO'
            handler.setLevel(level)
            handler.error('Logging level was not properly set ({}). Fall back to INFO.'.format(original_level))
        else:
            handler.setLevel(level.upper())
        
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def average_rm_map(rms, rmes, logger=None):
    """
    Make average RM and R   
        # self.rr.loc[]
    
    
        # logger.debug('radial_max = {}'.format(radial_max))
        # self.logger.debug('radial_max_positions[0].T - param.deccenpix.values[0] = {}'.format(radial_max_pos[1].T -param.racenpix.values[0] ))
    
            
    
    
    # ridgeline = np.concatenate([radial_max_pos, [radial_max]], axis=0).T
        
        # logger.debug('radial_max shape is {}'.format(radial_max.shape))
        # logger.debug('radial_max_pos.T shape is {}'.format(radial_max_pos.shape))
        
        # logger.info('\n{}'.format(ridgeline[:3]))
        M_ERROR maps out of several RM maps and RM_ERROR maps.

    Args:
        rms ([[]]): 
            list of RM maps as 2D numpy arrays
        rmes ([[]]): 
            list of RM_ERROR maps as 2D numpy arrays
    
    Returns:
        rm:
            average RM map as 2D numpy array
        rme:
            average RM_ERROR map as 2D numpy array

    """
    if logger:
        logger.debug('Stacking RM maps:\n{}'.format(rms))
        logger.debug('Stacking RM_ERROR maps:\n{}'.format(rmes))
    
    rm_stack = np.array(rms)
    rme_stack = np.array(rmes)
    
    rm_stack[rm_stack < -4999.0] = np.nan
    rme_stack[rme_stack < -4999.0] = np.nan
    
    if logger:
        logger.debug('rm[0] dimensions are: {}'.format(rms[0].shape))
        logger.debug('rm_stack dimensions are: {}'.format(rm_stack.shape))
    
    rm = np.nanmean(rm_stack, axis = 0)
    rme = np.nanmean(rme_stack, axis = 0)
    
    np.nan_to_num(rm, copy=False, nan=-5000)
    np.nan_to_num(rme, copy=False, nan=-5000)
    

    return rm, rme


def average_contour_map(conts, logger=None):
    """
    Make average CONTOUR map out oridgelinef several Stokes I

    Args:
        rms ([[]]): 
            list of RM maps as 2D numpy arrays
        rmes ([[]]): 
            list of RM_ERROR maps as 2D numpy arrays
    
    Returns:
        rm:
            average RM map as 2D numpy array
        rme:
            average RM_ERROR map as 2D numpy array

    """
    if logger:
        logger.debug('Stacking RM maps:\n{}'.format(conts))
    
    cont_stack = np.array(conts)
    
    if logger:
        logger.debug('conts[0] dimensions are: {}'.format(conts[0].shape))
        logger.debug('cont_stack dimensions are: {}'.format(cont_stack.shape))
    
    cont = np.nanmean(cont_stack, axis = 0)
    
    return cont


def smooth_map(fits, factor=2.0, logger=None):
    """Smooth a map with a Gaussian kernel. 
    The size of the kernel is defined as a factor to the BMAJ, BMIN which are
    already written in the FITS object.
    
    MAP = REAL_MAP * BEAM(BMAJ, BMIN)
    SMOOTH_MAP = REAL_MAP * BEAM(BMAJ*factor, BMIN*factor)
    
    Given, that for convolution of two Gaussians f() and g() with FWHMs w_f and w_g 
    respectively: FWHM(f*g) = sqrt(w_f**2 + w_g**2) . If f() - initial map and g() - convolving kernel,
    then w_g = w_f * sqrt(factor**2 - 1)
    
    Args:
        fits:
            FITS object (data+header)
        factor (float):
            increase beam by this factor
        logger:
            logger to use
            
    
    Returns:
        smoothed fits map
    """
    header = fits[0].header
    map2d = fits[0].data.squeeze()
    param = get_parameters(header)
    
    # in pixels
    BMAJ = np.abs(param.bmaj.values[0] / param.rapixsize.values[0])
    BMIN = np.abs(param.bmin.values[0] / param.rapixsize.values[0])
    
    
    # for simplicity, use round Gaussian with area = area_initial * (factor**2 - 1)
    # => radius = sqrt(bmaj**2 + bmin**2) * sqrt(factor**2 - 1) 
    kernel_r = np.sqrt(BMAJ**2 + BMIN**2) * np.sqrt(factor**2 - 1) 
    if logger:
        logger.info('Initial map BMAJ, BMIN = {:.2f}, {:.2f} pix'.format(BMAJ, BMIN))
        logger.info('Smoothing with a kernel. FWHM = {:.2f} pix'.format(kernel_r))
    
    smoothed_map = gaussian_filter(map2d, kernel_r)
    
    # fits[0].data = smoothed_map[np.newaxis, np.newaxis, :]
    fits[0].data = np.expand_dims(smoothed_map, axis=(0,1))

    # if logger:
        # logger.info(fits[0].header['BMAJ'])
    
    return fits
    


def plot_rm_diff(rm1file,rm2file, contourfile = None, drm = 500, at = (0,0)):
    '''plot difference of 2 rm maps.
    In both maps, pixels are blanked out if they are NaN in any of the maps.'''
    
#    base = 
    
    # 3C 273
    if 0:
        rm1file = base+'/3c273/frm.1226+023.c1-x2.2009_08_28.fits'
        rm2file = base+'/3c273/frm.1226+023.c1-x2.2009_10_25.fits'
    if 0:
        rm1file = base+'/3c273/frm.1226+023.c1-x2.2009_10_25.fits'
        rm2file = base+'/3c273/frm.1226+023.c1-x2.2009_12_05.fits'
    if 0:
        rm1file = base+'/3c273/frm.1226+023.c1-x2.2009_12_05.fits'
        rm2file = base+'/3c273/frm.1226+023.c1-x2.2010_01_26.fits'
    
    
    
    if 0:
        rm1file = base+'/3c273/frm.1226+023.x1-u1.2009_08_28.fits'
        rm2file = base+'/3c273/frm.1226+023.x1-u1.2009_10_25.fits'
    if 0:
        rm1file = base+'/3c273/frm.1226+023.x1-u1.2009_10_25.fits'
        rm2file = base+'/3c273/frm.1226+023.x1-u1.2009_12_05.fits'
    if 0:
        rm1file = base+'/3c273/frm.1226+023.x1-u1.2009_12_05.fits'
        rm2file = base+'/3c273/frm.1226+023.x1-u1.2010_01_26.fits'
    
    
    if 0:
        rm1file = base+'/3c273/frm.1226+023.u1-q1.2009_08_28.fits'
        rm2file = base+'/3c273/frm.1226+023.u1-q1.2009_10_25.fits'
    if 0:
        rm1file = base+'/3c273/frm.1226+023.u1-q1.2009_10_25.fits'
        rm2file = base+'/3c273/frm.1226+023.u1-q1.2009_12_05.fits'
    if 0:
        rm1file = base+'/3c273/frm.1226+023.u1-q1.2009_12_05.fits'
        rm2file = base+'/3c273/frm.1226+023.u1-q1.2010_01_26.fits'
    
    
    # 3C 279
    if 0:
        rm1file = base+'/3c279/frm.1253-055.c1-x2.2009_08_28.fits'
        rm2file = base+'/3c279/frm.1253-055.c1-x2.2009_12_05.fits'
    if 0:
        rm1file = base+'/3c279/frm.1253-055.c1-x2.2009_12_05.fits'
        rm2file = base+'/3c279/frm.1253-055.c1-x2.2010_01_26.fits'
    
    
    if 0:
        rm1file = base+'/3c279/frm.1253-055.x1-u1.2009_08_28.fits'
        rm2file = base+'/3c279/frm.1253-055.x1-u1.2009_12_05.fits'
    if 0:
        rm1file = base+'/3c279/frm.1253-055.x1-u1.2009_12_05.fits'
        rm2file = base+'/3c279/frm.1253-055.x1-u1.2010_01_26.fits'
    
    
    
    rm1d = read_fits(rm1file)
    rm2d = read_fits(rm2file)
    
    rm1 = rm1d[0].data.squeeze()
    rm2 = rm2d[0].data.squeeze()
    
    
    
#    rm1[((rm1 is np.NaN) & (rm1< -4000)) | ((rm2 is np.NaN) &(rm2< -4000))] = np.NaN
    
    rm1[rm1 < -4000 ] = np.nan
    rm1[rm2 < -4000 ] = np.nan
    
    rm2[rm1 < -4000 ] = np.nan
    rm2[rm2 < -4000 ] = np.nan
    
    
    
    RM = rm1d
    
    rm = rm1-rm2
    rm[rm < -drm] = np.nan
    rm[rm >  drm] = np.nan

    
    RM[0].data = rm
    
    
    
    
    
    rmheader = RM[0].header
    rmdf = get_parameters(rmheader)
#    wr = wcs.WCS(naxis=2)
#    wr.wcs.crpix = [rmdf['racenpix'].values[0],rmdf['deccenpix'].values[0]]
#    PIXEL_PER_MAS = 3.6*10**6
##    wr.wcs.cdelt = np.array([rmdf['rapixsize'].values[0] * PIXEL_PER_MAS, -rmdf['decpixsize'].values[0] * PIXEL_PER_MAS])  # note "-" in front of Y axis . Am I mixing axes???
#    wr.wcs.crval = [0, 0]
##    wr.wcs.ctype = ["RA", "DEC"]
#    wr.wcs.ctype = [ 'XOFFSET' , 'YOFFSET' ]
#    wr.cunit = ['mas', 'mas']
#    wr.wcs.cdelt = (wr.wcs.cdelt * u.deg).to(u.mas)

    
    wr = make_transform(rm1d)
    


    fig, ax = start_plot(RM, w=wr)
    
    if 1: # ok for low and mid
        ax.set_xlim([400,800])
        ax.set_ylim([200,600])

#    if 1: # ok for hig
#        ax.set_xlim([450,650])
#        ax.set_ylim([350,550])

    if contourfile:
        cont = read_fits(contourfile)
        cdata = cont[0].data.squeeze()
        
#        print('RM min = {}, RM max = {}'.format(RM[0].data.squeeze().nanmin(), RM[0].data.squeeze().nanmax()) )
        
        fig, ax = plot_image(RM,  fig=fig, ax=ax, vlim = [np.nanmin(RM[0].data.squeeze()), np.nanmax(RM[0].data.squeeze())] ,  colorbar = True)
        fig, ax = plot_contours(cdata, fig=fig, ax=ax, colors = 'black')
    else:
        fig, ax = plot_image(RM,  fig=fig, ax=ax, vlim = [-drm, drm] ,  colorbar = True)
    
    
    
#    print('RM value at ({},{}) = {}'.format(at[0], at[1], rm[at]))
    
    return RM
    


def start_plot(i,df=None, w=None, xlim = [None, None], ylim=[None, None]):
    '''starts a plot and returns fig,ax .
    xlim, ylim - axes limits in mas
    '''
       
    # make a transformation
    # Using a dataframe
    if df is not None:
        w = make_transform_df(df)	     
    # using a header    
    if w is not None:
        pass
    # not making but using one from the arg list
    else:
        w = make_transform(i)

#    print('In start_plot using the following transformation:\n {}'.format(w))


    fig = plt.figure()
    
    if w.naxis == 4:
        ax = plt.subplot(projection = w, slices = ('x', 'y', 0 ,0 ))
    elif w.naxis == 2:
        ax = plt.subplot(projection = w)
        
    
    # convert xlim, ylim to coordinates of BLC and TRC, perform transformation, then return back to xlim, ylim in pixels
    if any(xlim) and any(ylim):
        xlim_pix, ylim_pix = limits_mas2pix(xlim, ylim, w)
        ax.set_xlim(xlim_pix)
        ax.set_ylim(ylim_pix)
        
    
    fig.add_axes(ax)  # note that the axes have to be explicitly added to the figure
    return fig, ax 


# p is squeezed 
# fig, ax - required
def plot_image(p, fig , ax, cutoff = None, cmap = 'rainbow', origin = 'lower',  colorbar = False, xlim = None, ylim = None, vlim=None,  transform=None,
               title = None, xlabel = None, ylabel = None, use_cutoff = False):
    '''plot an image using imshow.
    cutoff here is in physical units    
    '''
    
#    # open a new plotting window if required
#    if fig is None and ax is None:
#        fig, ax = start_plot(p)
#    
#    if fig is not None and ax is None:
#        ax = fig.axes
        
    try:
        data = p[0].data.squeeze()
    except:
        data = p
    
    
    
#    print(data)
    
    if cutoff is None and use_cutoff is True:  # default is to guess cutoff from he data
        cutoff = corner_std(data)
        print('Calculated cutoff is {:.3f}'.format(cutoff))
        
    
    # mask values below the cutoff        
    if cutoff is not None and use_cutoff is True:
        mdata = ma.masked_less(data, cutoff)
    elif vlim is not None:
        mdata = ma.masked_outside(data, vlim[0], vlim[1])
    else:
        mdata = ma.masked_less(data, np.inf)
    
    
    if transform is not None: 
        if vlim is not None:
            im = ax.imshow(mdata, cmap = cmap, origin = origin, transform = transform, vmin = vlim[0], vmax = vlim[1])
        else:
            im = ax.imshow(mdata, cmap = cmap, origin = origin, transform = transform)
        
    else:
        if vlim is not None:
            im = ax.imshow(mdata, cmap = cmap, origin = origin, vmin = vlim[0], vmax = vlim[1])
        else:
            im = ax.imshow(mdata, cmap = cmap, origin = origin)
        

    
    if xlim is not None:
        try:
            ax.set_xlim(xlim)
        except:
            pass

    if ylim is not None:
        try:
            ax.set_ylim(ylim)
        except:
            pass
    
    
    if fig is None and colorbar is True:
        print('Can not plot a colorbar with no figure specified')
    elif fig is not None and colorbar is True:
        fig.colorbar(mappable = im, ax = ax)
    
    
    if title is not None:
        fig.suptitle(title)
    
    
    return fig, ax


    

# unlike in rm.py, in this version plot_contours receives a squeezed data 
def plot_contours(p, fig, ax, cutoff=1, colors = 'white', alpha = 0.5,  transform = None):
    '''plot contours on an image. cutoff is in %.'''
#    data = p[0].data.squeeze()
    if transform is not None:
        plt.contour(p, levels=loglevs(cutoff,p), colors=colors, alpha=alpha, transform = transform)
    else:
        plt.contour(p, levels=loglevs(cutoff,p), colors=colors, alpha=alpha)
    
    return fig, ax


# p is squeezed
def plot_ticks(p, chi,  fig , ax , every =1, pcutoff = 0.001, rotate =0, scale=2000, tick_length = 10, colors = 'white'):
    '''plot polarisation ticks.
    p defines length of a tick. If p==1, all have the same length.
    chi defines rotation.
    Every 'every's tick is plotted'''
    
    
    # load data and reduce by taking every 'every' point
    P = p[::every,::every]
    Pm = ma.masked_less(P, pcutoff) 
    
    CHI = chi[0].data.squeeze()[::every,::every]
    CHIm = ma.masked_where(P < pcutoff, CHI)
    
    xx,yy = np.meshgrid(np.arange(0,P.shape[0]), np.arange(0,P.shape[1]))       # P is already reduced in size
    
    angle = CHIm*np.pi/180
    
    if tick_length:
        dx = tick_length*np.sin(angle) / 2.0
        dy = tick_length*np.cos(angle) / 2.0
    else:
        dx = Pm*scale*np.sin(angle) / 2.0
        dy = Pm*scale*np.cos(angle) / 2.0
        
    xstarts = every * xx + dx           # expand grid by a factor of 'every', e.g. if every==4, (1,2,3) --> (4,8,12)
    ystarts = every * yy - dy
    xstops = every * xx - dx
    ystops = every * yy + dy
    
    # make segments
    seg = np.dstack([xstarts, ystarts, xstops, ystops])     # seg shape is data.shape, 4 i.e. every pixel of p has 4 values attached: xstart, ystart, xstop, ystop
    # mask seg according to the mask of P
    MASK = np.ma.dstack((Pm.mask,Pm.mask,Pm.mask,Pm.mask)).squeeze()
    segm = ma.masked_where(MASK, seg)
    
    segr = np.reshape(segm, (xstarts.size, 4), 'F')              # convert an underlying 2D map shape into a 1D array. Every element of this array still have 4 values attached. 'F' makes first axes to be reshaped. E.g. (10,10,4) array is converted to a (100, 4) array
    
#    # mask 
#    Pline = np.reshape(P, (P.size,1))
#    Pline4 = np.ma.dstack((Pline,Pline,Pline,Pline)).squeeze()
##    Pline4 = Pline4.squeeze()
#    segrm = ma.masked_where(Pline4.mask, segr)
    
    seg = np.reshape(segr, (xstarts.size,2,2))      # group 4 values into [2],[2] arrays. First is [xstart, ystart]. Second is [xstop, ystop].
    lw= 0.5
    lines = mc.LineCollection(seg[~seg.mask].reshape((-1,2,2)), colors = 'white', linewidth = lw)
#    seg2 = seg+lw    # add 1 pixel to all points
#    lines2 = mc.LineCollection(seg2[~seg2.mask].reshape((-1,2,2)), colors = 'black', linewidth = lw)
    # open a new plotting window if required
    if fig is None and ax is None:
        fig, ax = start_plot(p)
    
    if fig is not None and ax is None:
        ax = fig.axes
    
    t = ax.add_collection(lines)
    t.set_path_effects([path_effects.PathPatchEffect(offset=(1,-1), facecolor='gray'), path_effects.PathPatchEffect(edgecolor='white', linewidth=1.1, facecolor='black')])
#    ax.add_collection(lines2)
    
    
    return fig, ax
    


#start_plot

def plot_pol_map(idata, pdata, chidata, df, 
                 xlim=[None, None], ylim=[None, None], zoom = 'auto', 
                 icutoff_units = None, icutoff = None, icutoff_snr = 5,
                 pcutoff_units = None, pcutoff = None, pcutoff_snr = 6,
                 every = 5, scale = 2000,
                 outfile = None,
                 addtext=[None]):
    ''' plot a polarization map for a single frequency.
    INPUTS: Stokes I data, P data, CHI data, df with parameters.
    xlim and ylim - limit axes , [mas]
    zoom = None - default
         = auto : calculate xlim, ylim, Not implemented yet
    icutoff- cutoff level for contours in %
    icutoff_units - cutoff level for contours in physical units. Can not be set together with icutoff
    icutoff_snr - cutoff  = imap_rms *  icutoff_snr for imap
    pcutoff - cutoff for image in %
    pcutoff_units - cutoff level for image in physical units. Can not be set together with pcutoff
    pcutoff_snr - cutoff  = pmap_rms *  pcutoff_snr for pmap
    every - plot every's pol tick
    outfile - plot figure to outfile. Then close interactive window. 
    addtext - array of lines to add to the suptitle. 
    
    1. start plot.
    2. plot image (p)
    3. plot contors (i)
    4. plot ticks (P, CHI)
    5. plot beam, labels etc.
    '''
    

    # calculate cutoff values based on parameters
    # default is to take icutoff_snr times i_rms for imap and pcutoff_snr times p_rms for pmap 
    if icutoff_units is not None:    # cutoff in physical units first
        ICUTOFF = icutoff_units
    elif icutoff is not None:    # then cutoff in per cent
        ICUTOFF = icutoff * 0.01 * idata[0].data.squeeze().max()
    else:    # else calculate cutoff from map rms
        ICUTOFF = corner_std(idata[0].data) * icutoff_snr
        print('corner std = {}'.format(corner_std(idata[0].data)))
   
    if pcutoff_units is not None:    # cutoff in physical units first
        PCUTOFF = pcutoff_units
    elif pcutoff is not None:    # then cutoff in per cent
        PCUTOFF = pcutoff * 0.01 * pdata[0].data.squeeze().max()
    else:    # else calculate cutoff from map rms
        PCUTOFF = corner_std(pdata[0].data) * pcutoff_snr
        
    # map extent in pixels
    if all(xlim) and all(ylim):
        map_extent = 0.5 * (np.abs(xlim[0] - xlim[1]) + np.abs(ylim[0] - ylim[1]))
    else:
        print(idata[0].data.squeeze().shape)
        map_extent = np.average(idata[0].data.squeeze().shape)
    
    
#    print('$$'*40)
#    print('map extent = {}'.format(map_extent))
#    print('$$'*40)
        
        
    # with given or calculated ICUTOFF, define image portion to be displayed based on idata
    if zoom == 'auto':
        w = make_transform_df(df)
        xlim_auto, ylim_auto = calc_zoom(idata[0].data.squeeze().byteswap().newbyteorder(), ICUTOFF, margins = 0.05)
        # convert from array-view of indexing to FITS(imshow) view
        xlim_auto[0] = df.ramapsize - xlim_auto[0]
        xlim_auto[1] = df.ramapsize - xlim_auto[1]
        ylim_auto[0] = df.decmapsize - ylim_auto[0]
        ylim_auto[1] = df.decmapsize - ylim_auto[1]
        
        map_extent = 0.5 * (np.abs(xlim_auto[0] - xlim_auto[1]) + np.abs(ylim_auto[0] - ylim_auto[1]))
        
        xlim, ylim = limits_pix2mas(xlim_auto, ylim_auto, w)
        
        print('xlim = {} - {} \nylim = {} - {}'.format(xlim[0], xlim[1], ylim[0], ylim[1]) )
    
    
    print('ICUTOFF = {:.4f} Jy = {:.4f}% , PCUTOFF = {:.4f} Jy = {:.4f}%'.format(ICUTOFF, ICUTOFF * 100 / idata[0].data.squeeze().max(), PCUTOFF,  PCUTOFF * 100 / pdata[0].data.squeeze().max()))
    
    
#    fig,ax = start_plot(idata, df=df, xlim = xlim, ylim = ylim)
    fig,ax = start_plot(idata, df=df, xlim = ylim, ylim = xlim)   # note swapped limits. Need to figure out why it is correct
    fig,ax = plot_image(pdata, fig=fig, ax=ax, cutoff = PCUTOFF, colorbar = True)    # cutoff here in physical units
    fig,ax = plot_contours(idata, fig=fig, ax=ax, cutoff = ICUTOFF * 100 / idata[0].data.squeeze().max(), colors = 'grey', alpha=0.7) # cutoff here in per cent. TODO rewrite plot_contours later to accept physical units for cutoff
    
    # calculate best every value
    SCALE = scale
    EVERY = every
    
    if every ==0:
        # calculate scale. Let average tick length be 10 pixels
        SCALE =  10 / np.average(pdata[0].data.squeeze())
        # 
        EVERY = np.int(map_extent / 25)  # fundamentally wrong. Need to think how to make this properly
        print('map extent = {}'.format(map_extent))
        print('every = {}, scale = {}'.format(EVERY, SCALE))
    
    fig,ax = plot_ticks(pdata, chidata, fig=fig, ax=ax, every = EVERY, scale = SCALE, pcutoff = PCUTOFF)
    
    
    title = '{} at {:.1f} GHz on {}'.format(df.source, df.frequency/10**9 , df.dateobs)
    
    if any(addtext) is not None:
        for i,t in enumerate(addtext):
            title = title + '\n' + t
    
    
    
    fig.suptitle(title)
    
    
    if outfile: 
        fig.savefig(outfile)
        plt.close()
    
    

#def calculate_cutoff(imap, cutoff):
#    ''' calculate cutoff to 




def calc_zoom(imap, cutoff, margins = 0.1):
    '''find contigous region on a map. Return back an index to crop all maps to that'''

    
    # clip image
    imap[imap<cutoff] = 0
    # label continuous features  
    labeled_array, num_features = label(imap)
    # locate objects -> return slices for every object
    loc = find_objects(labeled_array)  
    
    # check label at ~ the middle of the array (center of the image). Might be unsafe in general case
    LABEL = labeled_array[int(labeled_array.shape[0]/2), int(labeled_array.shape[1]/2)]
    
#    print('loc of the LABEL {} is {} '.format(LABEL, loc[LABEL-1]))
#    print('array values are {}'.format(imap[loc[LABEL-1]]))
    
    # x range
    xlim = [ loc[LABEL-1][0].start * (1 - margins), loc[LABEL-1][0].stop * (1 + margins) ]
    # y range 
    ylim = [ loc[LABEL-1][1].start * (1 - margins), loc[LABEL-1][1].stop * (1 + margins)]
    

    return xlim, ylim




# FITS handling functions



def deepcopy(HDULIST):
    '''deep copy HDU object'''
    return fits.open(HDULIST.filename())



def read_fits(file):
    '''read fits file into object'''
    try:
        res =  fits.open(file)
        return res
    except:
        return None


def make_filelist(files):
    ''' Make a filelist given file basenames. 
    1. Basenames are expanded. 
    *. names are expected to have .ifits extension
    2. add qfits and ufits '''
    
    filelist = np.empty((len(files),3), dtype='object')
    
    for i, f in enumerate(files):
        path = os.path.realpath(f)
        tmp = path.split('.')
        basename = '.'.join(tmp[:-1])
        ifile = basename+'.ifits'
        qfile = basename+'.qfits'
        ufile = basename+'.ufits'
        filelist[i,:] = (ifile, qfile, ufile)
        
    return filelist




def read_files(filelist):
    '''read in all FITS files. Should be 3 times N where N is the number of frequncies. '''
    
    files = np.empty_like(filelist)
    
    for (freq_idx, stokes_idx), file in np.ndenumerate(filelist):
        files[freq_idx, stokes_idx] = read_fits(file)
        
    return files # array Nx3


def get_maps(files):
    '''get FITS maps from primaryHDUs of files'''
    maps = np.empty_like(files)
    
    for (freq_idx, stokes_idx), file in np.ndenumerate(files):
        maps[freq_idx, stokes_idx] =  file[0].data.squeeze()
    
    return maps


def get_header(file):
    '''get header for a file.'''
    return file[0].header



def get_headers(files):
    '''get headers for all files. Or may be only for Stokes I.'''
    headers = np.empty_like(files)
    
    for (freq_idx, stokes_idx), file in np.ndenumerate(files):
        headers[freq_idx, stokes_idx] =  file[0].header
    
    return headers    



def get_parameters(header):
    ''' get some parameters from a header: CRVAL, CRPIX, FREQ, SOURCE, DATE-OBS'''
    
    param = pd.DataFrame(columns = [ 'racenpix', 'deccenpix',
                                    'rapixsize', 'decpixsize', 
                                 'ramapsize', 'decmapsize',
                                 'bmaj', 'bmin', 'bpa',
                                 'source', 'dateobs',
                                 'frequency',
                                 'masperpix'],
                        data = [[header['CRPIX1'],header['CRPIX1'],
                                header['CDELT1'],header['CDELT1'], 
                                header['NAXIS1'],header['NAXIS2'],
                                header['BMAJ'],header['BMIN'],header['BPA'],
                                header['OBJECT'],header['DATE-OBS'],
                                header['CRVAL3'], 
                                header['CDELT1']*3.6e6]])
     # MASperPIX = np.abs(self.rm[0].header['CDELT1']*3.6e6)
    
    return param  # a Pandas DataFrame


# image manipulation. 









    

def plot_pmap():
    '''like a main func. Instead of using polplot.py, plot maps with auto zoomed ranges from here.'''

    '''should be one source one date??? '''
#    dates = ['2009_02_02', '2009_02_05', '2009_08_28' , '2009_10_25', '2009_12_05', '2010_01_26']  # first two epoch are really poor in polarisation. Just discard them
    dates = ['2009_08_28' , '2009_10_25', '2009_12_05', '2010_01_26']
    sources = ['0851+202', '1226+023', '1253-055' , '1308+326']
    ranges = ['low', 'mid' , 'hig']
    
    # test case    
    dates = [ '2009_10_25']
    sources = [ '1308+326']
    ranges = [ 'low']
    
    for s in sources:
        for d in dates:
            for r in ranges:
                fff = glob.glob("/aux/vcompute2a/mlisakov/S2087A/polar/{}/maps4rm/{}*{}*{}.ifits".format(s,s,d,r))
                if len(fff) == 0:   # if no files found
                    continue
                
                print('--'*30)
                print('doing {} on {}, {} range'.format(s, d ,r))
                print('--'*30)

                filelist = make_filelist(fff)
                files = read_files(filelist)    # astropy FITS objects
                maps = get_maps(files)          # maps 
                headers = get_headers(files)    # headers
                bands = get_bands(filelist)
                pchis = make_pchis(files)
                # pack all the data into a dataframe
                df = pd.DataFrame(columns = [ 'frequency', 
                                             'ifile', 'qfile' , 'ufile' , 
                                             'idata', 'qdata', 'udata', 
                                             'imap', 'qmap', 'umap', 
                                             'pdata', 'chidata',
                                             'pmap', 'chimap',
                                             'iheader', 'qheader', 'uheader',
                                             'rashift', 'decshift' , 
                                             'racenpix', 'deccenpix' ,
                                             'rapixsize', 'decpixsize', 
                                             'ramapsize', 'decmapsize',
                                             'bmaj', 'bmin', 'bpa',
                                             'source', 'dateobs',
                                             'pimage'], 
                                    index = bands    )
                
                df.loc[bands, ['ifile', 'qfile' , 'ufile']] = filelist      # file names
                df.loc[bands, ['idata', 'qdata', 'udata']] = files          # astropy FITS objects
                df.loc[bands, [ 'imap', 'qmap', 'umap']] = maps             # maps
                df.loc[bands, [ 'iheader', 'qheader', 'uheader']] = headers # headers
                
                # map shifts
                df.loc[bands, ['rashift', 'decshift']] = np.zeros((len(bands), 2))      # map shifts
                
                   # map parameters
                for freq_idx, h in enumerate(df.iheader):
                    param = get_parameters(h)
                    df.loc[bands[freq_idx], ['racenpix', 'deccenpix' , 'rapixsize', 'decpixsize', 'ramapsize', 'decmapsize','bmaj', 'bmin', 'bpa', 'source', 'dateobs', 'frequency']] = param.loc[0,:]
                 
                # p, chi maps
                df.loc[bands, [ 'pdata', 'chidata']] = pchis
                
                # construct filename for a pol image
                df.loc[bands, 'pimage'] = df.loc[bands, 'ifile'].str.replace('maps4rm', 'images4rm').str.replace('ifits', 'pdf')
                
                for b in df.index:
                    print()
                    print('>>>> band = {}'.format(b))
                    print()
                    print(df.ifile[b])
                    print()
#                    plot_pol_map(df.idata[b], df.pdata[b], df.chidata[b], df.loc[b], every =0, icutoff_snr = 3,  pcutoff_snr = 10, outfile = df.pimage[b], addtext = [r])   # to file
                    plot_pol_map(df.idata[b], df.pdata[b], df.chidata[b], df.loc[b], every =0, icutoff_snr = 1,  pcutoff_snr = 5,  addtext = [r])    # interactive






    


def plot_rm(rmfile, contourfile, rmefile = None,  vlim = [-5000,5000], xlim = [4,-15], ylim = [4,-15], at = [0,0], interactive = False):    
    '''function to plot RM from a fits file onto a contour map.
    Should have an interactive regime to plot RM values along a slice or at a given point
    '''
    
    s= re.search( 'c1-x2|x1-u1|u1-q1', rmfile)
    suffix=s.group(0)
    rm = read_fits(rmfile)
    cont = read_fits(contourfile)
    
    try:
        rme = read_fits(rmefile)[0].data.squeeze()
    except:
        rme = np.zeros(rm[0].data.squeeze().shape)

    rmheader = rm[0].header
    contheader = cont[0].header
    rmdf = get_parameters(rmheader)
    contdf = get_parameters(contheader)


    # RM  
#    PIXEL_PER_MAS = 3.6*10**6 
#    
#    wr = wcs.WCS( start:naxis=2)
#    wr.wcs.crpix = [rmdf['racenpix'].values[0],rmdf['deccenpix'].values[0]+1]
#    wr.wcs.cdelt = np.array([rmdf['rapixsize'].values[0] * PIXEL_PER_MAS, -rmdf['decpixsize'].values[0] * PIXEL_PER_MAS])  # note "-" in front of Y axis . Am I mixing axes???
#    wr.wcs.crval = [0, 0]
#    wr.wcs.ctype = ['RA', 'DEC']
##    
#    
##    print('rm dataframe is:\n{}'.format(rmdf.iloc[0]))
#    print('conr dataframe is:\n{}'.format(contdf.iloc[0]))

#    wr = wcs.WCS(rmheader)
#    wr = wr.celestial
    wr = make_transform(rm)
                 
    
    wc2 = wcs.WCS(contheader)
    wc2 = wc2.celestial
    
    fig, ax = start_plot(cont, w=wr, xlim = xlim,  ylim = ylim)
#    fig, ax = start_plot(cont, w=wr, xlim = [2,-7], ylim = [2,-7])
#    fig, ax = start_plot(cont, w=wr)

    plot_image(rm[0].data.squeeze(), cutoff = -4000,  vlim = vlim,  fig=fig, ax=ax, colorbar = True,
               title = '{} on {} at {}'.format(rmdf.source.values[0], rmdf.dateobs.values[0], suffix.upper()))
    plot_contours(cont[0].data.squeeze(), cutoff = 0.1, fig=fig, ax=ax, colors = 'grey')
    
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RM value at ({},{}) = {}'.format(at[0], at[1], rm[0].data.squeeze()[-at[1], -at[0]]))
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RMerr value at ({},{}) = {}'.format(at[0], at[1], rme[-at[1], -at[0]]))

    ax.plot(-at[0], -at[1], 'or')
    
    ax.coords[0].set_major_formatter('%.1f')
    ax.coords[1].set_major_formatter('%.1f')
    
    
    
    return fig, ax
    
def plot_rm_fits(rm, cont, rme = None,  vlim = [-5000,5000], xlim = [4,-15], ylim = [4,-15], 
                 at = None, interactive = False,
                 ridgeline=None,
                 ridgeline_new=None,
                 slice_coord=None,
                 logger=None,
                 suffix=None):    
    '''Plot RM from a FITS OBJECT onto a contour map.
    
    Args:
        rm: FITS object with RM data
        rme: FITS object with RM_ERROR data
        cont: FITS object with Stokes I data
        vlim ([float]): RM range in rad/m2
        xlim ([float]): x range in mas
        ylim ([float]): y range in mas
        at ([float]): report RM values at this point  (mas)
        interactive (bool): if True, draw slices with mouse. 
        
    Returns:
         None   
    
    '''
    if suffix is None:
        suffix = ''
    
    # if no logger is given, throw everything in trash
    if logger is None:
        logger = create_logger('null', dest=[], levels=[])

    if rme == None:
        rme = np.zeros(rm[0].data.squeeze().shape)

    rmheader = rm[0].header
    contheader = cont[0].header
    rmdf = get_parameters(rmheader)
    contdf = get_parameters(contheader)

    wr = make_transform(rm)
    wc2 = wcs.WCS(contheader)
    wc2 = wc2.celestial
    
    fig, ax = start_plot(cont, w=wr, xlim = xlim,  ylim = ylim)

    plot_image(rm[0].data.squeeze(), cutoff = -4000,  vlim = vlim,  fig=fig, ax=ax, colorbar = True,
               title = '{} on {} at {}'.format(rmdf.source.values[0], rmdf.dateobs.values[0], suffix.upper()))
    plot_contours(cont[0].data.squeeze(), cutoff = 0.1, fig=fig, ax=ax, colors = 'grey')
    
    
    if ridgeline is not None:
        # plot ridgeline (x, y, value)
        ax.plot(ridgeline[:,1] , ridgeline[:,0] , 'o') # note reversed order
        
    
    if ridgeline_new is not None:
        
        ridgeline_new_pix = ridgeline_new.T
        # first need to convert mas -> pix
        param = get_parameters(rm[0].header)        
        MASperPIX = np.abs(param.rapixsize.values[0]*3.6e6)
        ridgeline_new_pix[0] = (ridgeline_new.T[0] / MASperPIX + param.racenpix.values[0]).astype(int)
        ridgeline_new_pix[1] = (ridgeline_new.T[1] / MASperPIX + param.deccenpix.values[0]).astype(int)
        ridgeline_new_pix = ridgeline_new_pix.T
        
        logger.debug('RIDGELINE_NEW_PIX')
        logger.debug(ridgeline_new_pix[:,[1,0]])
        ax.plot(ridgeline_new_pix[:,0] , ridgeline_new_pix[:,1] , 'o') # note reversed order

    
    if slice_coord is not None:
        """NEED TO RENAME. """
        param = get_parameters(rm[0].header)        
        MASperPIX = np.abs(param.rapixsize.values[0]*3.6e6)
        slice_pix=[0,0]
        slice_pix[0] = (slice_coord.data.x / MASperPIX + param.racenpix.values[0]).astype(int)
        slice_pix[1] = (slice_coord.data.y / MASperPIX + param.deccenpix.values[0]).astype(int)

        ax.plot(slice_pix[0], slice_pix[1], '-')
    
    
    
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RM value at ({},{}) = {}'.format(at[0], at[1], rm[0].data.squeeze()[-at[1], -at[0]]))
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RMerr value at ({},{}) = {}'.format(at[0], at[1], rme[-at[1], -at[0]]))

    if at is not None:
        ax.plot(-at[0], -at[1], 'or')
    
    ax.coords[0].set_major_formatter('%.1f')
    ax.coords[1].set_major_formatter('%.1f')
    
    
    
    return fig, ax
    


class ArrowBuilder:
    
    now_drawing = False
    xystart = []
    laststart = []
    laststop = []
    source = ''
    epoch = ''

    
    # def __init__(self, line, ax, outslice = '/home/mikhail/tmp/slice.pkl', outbeam = '/home/mikhail/tmp/beam.pkl'):
    def __init__(self, line, ax, outslice = '/homes/mlisakov/tmp/slice.pkl', 
                 outbeam = '/homes/mlisakov/tmp/beam.pkl',
                 rm=None, rme=None, cont=None):
        
        self.line = line
        self.ax = ax            # try to get axes inside the object
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.outslice = outslice
        self.outbeam  = outbeam
        #provide data directly here as FITS objects
        self.rm = rm
        self.rme = rme
        self.cont = cont
        
        
        

    def from_coord(self, x1,y1, x2,y2 ):
        self.laststart = [x1,y1]
        self.laststop  = [x2,y2]
        
        print('laststart = {}'.format(self.laststart))
        
        if self.rm is not None:
            rmdata = self.rm[0].data.squeeze()
            rmheader = self.rm[0].header
            rmedata = self.rme[0].data.squeeze()
            contdata = self.cont[0].data.squeeze()
        else:
            rmdata = read_fits(rmfile)[0].data.squeeze()
            rmheader = read_fits(rmfile)[0].header
            rmedata = read_fits(rmefile)[0].data.squeeze()
            contdata = read_fits(contourfile)[0].data.squeeze()    
        
        self.source = rmheader['OBJECT']
        self.epoch = rmheader['DATE-OBS']
        
        
        
        
        rmdata2slice = np.copy(rmdata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
        rmedata2slice = np.copy(rmedata) # errors in RM
        self.proceed_slice(rmdata2slice, rmedata2slice = rmedata2slice)
        
        arrow = mpatches.Arrow(self.laststart[0], self.laststart[1], self.laststop[0] - self.laststart[0], self.laststop[1] - self.laststart[1], width=10)
        self.line.axes.add_patch(arrow)
        
        return         
        
        
        
    def __call__(self, event):
        
#        print(event.button)
        
        if event.inaxes!=self.line.axes: return
        if event.key == 'X' or event.key == 'x': return
        if event.key == 'j' :
            print('GOVNO!!!!!!!')
            self.parabola()
        
#        if event.button != 1: return
        
        self.now_drawing = not self.now_drawing
        print('Now drawing = {}'.format(self.now_drawing))
        print('click', event)
        
        if(self.now_drawing is True): # started drawing a new arrow
            circle = mpatches.Circle([event.xdata, event.ydata],2)
            self.xystart = [event.xdata, event.ydata]
            self.laststart = [event.xdata, event.ydata]
            self.line.axes.add_patch(circle)

        if(self.now_drawing is False): # stopped drawing a new arrow
            arrow = mpatches.Arrow(self.xystart[0], self.xystart[1], event.xdata - self.xystart[0], event.ydata - self.xystart[1], width=10)
            self.line.axes.add_patch(arrow)
            self.xystart = []
            self.laststop = [event.xdata, event.ydata]
            
            self.coord = np.array([self.laststart, self.laststop]).astype(np.int).flatten()
            
            # get a rect region from an image 
            if self.rm is not None:
                rmdata = self.rm[0].data.squeeze()
                rmedata = self.rme[0].data.squeeze()
                contdata = self.cont[0].data.squeeze()
                rmdata2slice = np.copy(rmdata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
                rmedata2slice = np.copy(rmedata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
            else:
                rmdata = read_fits(rmfile)[0].data.squeeze()
                rmedata = read_fits(rmefile)[0].data.squeeze()
                contdata = read_fits(contourfile)[0].data.squeeze()
                rmdata2slice = np.copy(rmdata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
                rmedata2slice = np.copy(rmedata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
            
            self.proceed_slice(rmdata2slice, rmedata2slice = rmedata2slice)
          
            
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
#        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
    
    def proceed_slice(self, rmdata2slice, rmedata2slice = None):
        '''having data arraya nd coordinates of the slice beginning and end, do stuff: diagonal, save sliace, save beam etc. '''
        
        
        print('in proceed slice, laststart = {}'.format(self.laststart))
        
        if self.laststart[0] == self.laststop[0]:
            self.laststop[0] = self.laststop[0] +1 
        if self.laststart[1] == self.laststop[1]:
            self.laststop[1] = self.laststop[1] +1 
        
        
        rmdata2slice = rmdata2slice[:, :]
        rmdata2slice = np.transpose(rmdata2slice)
        
        rmedata2slice = rmedata2slice[:, :]
        rmedata2slice = np.transpose(rmedata2slice) # errors

        
        slice_rect = rmdata2slice[int(self.laststart[0]) : int(self.laststop[0]):  -1 if self.laststart[0] > self.laststop[0] else 1,  int(self.laststart[1]):  int(self.laststop[1]) : -1 if  self.laststart[1] > self.laststop[1] else 1]
        slice_rect_e = rmedata2slice[int(self.laststart[0]) : int(self.laststop[0]):  -1 if self.laststart[0] > self.laststop[0] else 1,  int(self.laststart[1]):  int(self.laststop[1]) : -1 if  self.laststart[1] > self.laststop[1] else 1]

        self.slice_direction = np.arctan(slice_rect.shape[1] / slice_rect.shape[0])
#        self.ax.text(self.laststop[0], self.laststop[1], '{:0.0f} deg'.format(180 / np.pi* self.slice_direction))
        from skimage.transform import resize
        res = resize(slice_rect, (max(slice_rect.shape[0], slice_rect.shape[1]), max(slice_rect.shape[0], slice_rect.shape[1])))
        res_e = resize(slice_rect_e, (max(slice_rect_e.shape[0], slice_rect_e.shape[1]), max(slice_rect_e.shape[0], slice_rect_e.shape[1])))

        print(res.shape)
        diag = np.diag(res.squeeze())
        diag_e = np.diag(res_e.squeeze())
        
        
        if self.rm is not None:
            MASperPIX = np.abs(self.rm[0].header['CDELT1']*3.6e6)
        else:
            MASperPIX = np.abs(read_fits(rmfile)[0].header['CDELT1']*3.6e6)
        print('IN the original image, there are {} mas per 1 pixel'.format(MASperPIX))
        L = np.sqrt(slice_rect.shape[0]**2 + slice_rect.shape[1]**2) * MASperPIX
        
        print('The slice length in pixels is {} , in mas is {}'.format(np.sqrt(slice_rect.shape[0]**2 + slice_rect.shape[1]**2), L))
        F = L / diag.shape[0]
        
        print ('IN the slice there are {} mas per 1 pixel'.format(F))
        
#        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(x*F))
#        nax.xaxis.set_major_formatter(ticks_x)
#        nax.coord[0].set_major_formatter('%.1f')
#        nax.coords[0].set_major_formatter('%.1f')

        # mask out values that are out of the specified range in RM 
        #            self.diag_m = ma.masked_outside(diag, -1500 , 1500)
        self.diag_m = ma.masked_outside(diag, vlim[0] , vlim[1])
        self.diag_e = diag_e
        
        # add mas scale to the diagonal
        diag_mas = np.indices(diag.shape).squeeze() * F
        self.d = pd.DataFrame(index =np.indices(diag.shape).squeeze(),  columns = ['length', 'value'] )
        self.d.length = diag_mas
        self.d.value = diag
        # self.d is a dataframe that contains both pixel-scale (indices) and mas-scale (length) positions of measured data points with values (value)
        # save slice to file. By default, tmp.pkl
        save_slice(self.d, filename=self.outslice)
        print('Saved slice to {}'.format(self.outslice))
        
        # self.bbb = save_beam(get_parameters(get_header(read_fits(contourfile))), self.slice_direction*180/np.pi, pixel_size = F , minimize_N = True, filename = self.outbeam)
        # print('Saved beam to {}'.format(self.outbeam))
        
        
        # PLOT SLICE in a separate window
        # searate plot window for a slice
        nfig, nax = plt.subplots(1,1)
        #       
        # pixel-based plot
#        nax.errorbar(np.arange(self.diag_m.size), self.diag_m, yerr = diag_e)
#        nax.set_xlim(0, self.diag_m.size)
#        nax.set_ylim(vlim[0], vlim[1])
        
        # mas-based plot
        nax.errorbar(diag_mas, self.diag_m, yerr = diag_e)
#        nax.set_xlim(0, self.diag_m.size)
        nax.set_ylim(vlim[0], vlim[1])
        nax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#        nax.xaxis.set_minor_locator(MultipleLocator(5))
        nax.xaxis.set_minor_locator(AutoMinorLocator(5))

#        nax.set_xticks()
    
    
    
    
    
    def parabola(self):
        self.ax.text('Event button is pressed, p')
        

def get_slice_values(image):
    '''get a 2D image, resample it to a square grid, return a diagonal '''
#    import scipy.interpolate
#    squared_image = scipy.interpolate.Rbf(lon, lat, z, function='linear')
    
    import cv2
    res = cv2.resize(image, dsize=(max(image.shape[0], image.shape[1]), max(image.shape[0], image.shape[1])), interpolation=cv2.INTER_LINEAR)

    return res



def save_slice(df, filename='/home/mikhail/tmp/slice.pkl'):
    '''save slice dataframe into a file'''
    df.to_pickle(filename)
    return 


def read_slice(filename = '/home/mikhail/tmp/slice.pkl'):
    '''read slice back from file'''
    df = pd.read_pickle(filename)
    return df



def save_beam(df,  slice_direction, pixel_size = 0.1 , N=200, filename = '/home/mikhail/tmp/beam.pkl', minimize_N = False):
    '''save beam as an array.
    Input:  df :    FITS header as a result of get_parameters
            slice_direction : slice direction in degrees
            pixel_size : pixel size in [mas]. Better if match that os the slice
            N : number of points in desired output
            minimize_N : array length should not be loo big. With this flag set to True, output array is truncated so the minimum (border) values are 0.001 of the beam amplitude.
                
    opt. input: slice - to make resolutuion the same for the beam and for the slice itself'''
    
    bmaj = df.bmaj.values[0] * 3.6e6    # [mas]
    bmin = df.bmin.values[0] * 3.6e6    # [mas]
    bpa = df.bpa.values[0]              # [degrees]
    
    print(df)
    
    print('\n\nSAVING THE BEAM PROJECTION\n')
    print('beam bmaj = {:.2f}, bmin = {:.2f} , bpa = {:.2f}'.format(bmaj, bmin, bpa))
    print('slice direction = {:.1f} deg'.format(slice_direction))
    proj = ellipse_projection(bmaj, bmin, bpa, slice_direction,phi_coordinate_system='w2n')
#    print(proj)
    print('Beam projection length = {:.2f}'.format(proj))
    
    
    
    # proj - size of the beam FWHM [mas] projected onto a certain direction (direction of the slice)
    # proj/pixel_size- beam FWHM in pixels. 
    # pixel_size - size of one pixel of the beam
    # total_length  = N * pixel_size
    
    
    beam_data = make_beam(n=N, amplitude = 1, fwhm = proj/pixel_size)

    if minimize_N is True: 
        beam_data = beam_data[beam_data >= 0.001 * beam_data.max()]
        N = beam_data.size
    
    beam = pd.DataFrame(index = np.arange(0, N), columns = ['length' , 'value'])
    
    beam.loc[:,'length'] = np.arange(0,N) * pixel_size
    beam.loc[:, 'value'] = beam_data
    
    beam.to_pickle(filename)
    
    
    return beam
    
def ellipse_projection(bmaj, bmin, bpa, phi, phi_coordinate_system='n2e'):
    '''calculate a length of an ellipse projection onto a specified direction. 
    bmaj - ellipce major axis [mas]
    bmin - ellipse minor axis [mas]
    bpa  - ellipse major axis position angle, measures North-to-East [degrees]
    phi  - direction [degrees]
    phi_coordinate_system - could be n2e == North to East (default, as that for bpa)
                                     w2n == West to North (as per normal line definition)
    
    
    l - length of intersection of an ellipse with major axis = 2a, minor = 2b 
    with a straight line iclined by beta degrees to the major axis.
    Beta is calculates as alpha - phi (alpha - NorthToEast inclination angle of the line,
    phi is ellipse major axis NorthToEast inclination)
    l here is beam/core/comp FWHM in specified direction

    Equation: 
        l_beam = 1 / sqrt(  (cos(deg2rad(phi - bpa)) / bmaj)**2  + (sin(deg2rad(phi - bpa)) / bmin)**2  );
    '''
    
    if phi_coordinate_system == 'w2n':
        phi = 90 - phi   # convert phi to n2e system
    
    proj = 1 / np.sqrt(  (np.cos(np.deg2rad(phi - bpa)) / bmaj)**2 + (np.sin(np.deg2rad(phi - bpa)) / bmin)**2    )
    
    return proj
    
    
    
def make_filelist_etc(base=None, basecont=None, source='3C273', frange='mid', logger=None):
    """
    Generate lists of RM map files, RM_error map files, Stokes I map files. 
    All these files are used to plot a neat RM map.
    
    Args:
        base (str): base path for RM map files
        basecont (str): base path for Stokes I map files
        source (str): source name, default '3C273'. Can be '3C279' also. Other to be added later.
        frequency_range (str): frequency range to consider: low, mid, or hig. 
    Returns:
        rmfiles ([str]): list of RM map files
        rmefiles ([str]): list of RM_error map files
        contourfiles ([str]): list of Stokes I maps
        vlim ([float]): RM values range in rad/m2
        xlim ([float]): x range in mas
        ylim ([float]): y range in mas
    """
    
    rmfiles = rmefiles = contourfiles = vlim = xlim = ylim = []
    
    if source == '3c273':
        if frange == 'low':
            vlim = [-1000,1000]
            xlim = [4,-20]
            ylim = [4,-20]
            rmfiles= [
                    base+'/3c273/frm.1226+023.c1-x2.2009_08_28.fits', 
                    base+'/3c273/frm.1226+023.c1-x2.2009_10_25.fits',
                    base+'/3c273/frm.1226+023.c1-x2.2009_12_05.fits',
                    base+'/3c273/frm.1226+023.c1-x2.2010_01_26.fits'
                    ]
            rmefiles= [
                    base+'/3c273/frme.1226+023.c1-x2.2009_08_28.fits', 
                    base+'/3c273/frme.1226+023.c1-x2.2009_10_25.fits',
                    base+'/3c273/frme.1226+023.c1-x2.2009_12_05.fits',
                    base+'/3c273/frme.1226+023.c1-x2.2010_01_26.fits'
                    ]
            contourfiles=[
                    basecont+'/1226+023/maps4rm/1226+023.C1.2009_08_28.low.ifits',
                    basecont+'/1226+023/maps4rm/1226+023.C1.2009_10_25.low.ifits',
                    basecont+'/1226+023/maps4rm/1226+023.C1.2009_12_05.low.ifits',
                    basecont+'/1226+023/maps4rm/1226+023.C1.2010_01_26.low.ifits'
                    ]

        if frange == 'mid':
            vlim = [-2000,1500]
            xlim = [4,-20]
            ylim = [4,-20]
            rmfiles= [
                    base+'/3c273/frm.1226+023.x1-u1.2009_08_28.fits', 
                    base+'/3c273/frm.1226+023.x1-u1.2009_10_25.fits',
                    base+'/3c273/frm.1226+023.x1-u1.2009_12_05.fits',
                    base+'/3c273/frm.1226+023.x1-u1.2010_01_26.fits'
            ]
            rmefiles= [
                    base+'/3c273/frme.1226+023.x1-u1.2009_08_28.fits', 
                    base+'/3c273/frme.1226+023.x1-u1.2009_10_25.fits',
                    base+'/3c273/frme.1226+023.x1-u1.2009_12_05.fits',
                    base+'/3c273/frme.1226+023.x1-u1.2010_01_26.fits'
            ]
            contourfiles=[
                    basecont+'/1226+023/maps4rm/1226+023.X1.2009_08_28.mid.ifits',
                    basecont+'/1226+023/maps4rm/1226+023.X1.2009_10_25.mid.ifits',
                    basecont+'/1226+023/maps4rm/1226+023.X1.2009_12_05.mid.ifits',
                    basecont+'/1226+023/maps4rm/1226+023.X1.2010_01_26.mid.ifits'
            ]

        if frange == 'hig':
            vlim = [-4500,4500]
            xlim = [4,-10]
            ylim = [4,-10]
            rmfiles= [
                    base+'/3c273/frm.1226+023.u1-q1.2009_08_28.fits', 
                    base+'/3c273/frm.1226+023.u1-q1.2009_10_25.fits',
                    base+'/3c273/frm.1226+023.u1-q1.2009_12_05.fits',
                    base+'/3c273/frm.1226+023.u1-q1.2010_01_26.fits'
                    ]
            rmefiles= [
                    base+'/3c273/frme.1226+023.u1-q1.2009_08_28.fits', 
                    base+'/3c273/frme.1226+023.u1-q1.2009_10_25.fits',
                    base+'/3c273/frme.1226+023.u1-q1.2009_12_05.fits',
                    base+'/3c273/frme.1226+023.u1-q1.2010_01_26.fits'
                    ]
            contourfiles=[
                    basecont+'/1226+023/maps4rm/1226+023.U1.2009_08_28.hig.ifits',
                    basecont+'/1226+023/maps4rm/1226+023.U1.2009_10_25.hig.ifits',
                    basecont+'/1226+023/maps4rm/1226+023.U1.2009_12_05.hig.ifits',
                    basecont+'/1226+023/maps4rm/1226+023.U1.2010_01_26.hig.ifits'
                    ]
        
    if source == '3c279':
        if frange == 'low':
            vlim = [-250,250]
            xlim = [4,-10]
            ylim = [4,-10]
            rmfiles= [
                    base+'/3c279/frm.1253-055.c1-x2.2009_08_28.fits', 
                    base+'/3c279/frm.1253-055.c1-x2.2009_12_05.fits',
                    base+'/3c279/frm.1253-055.c1-x2.2010_01_26.fits'
                    ]
            rmefiles= [
                    base+'/3c279/frme.1253-055.c1-x2.2009_08_28.fits', 
                    base+'/3c279/frme.1253-055.c1-x2.2009_12_05.fits',
                    base+'/3c279/frme.1253-055.c1-x2.2010_01_26.fits'
                    ]
            contourfiles=[
                    basecont+'/1253-055/maps4rm/1253-055.C1.2009_08_28.low.ifits',
                    basecont+'/1253-055/maps4rm/1253-055.C1.2009_12_05.low.ifits',
                    basecont+'/1253-055/maps4rm/1253-055.C1.2010_01_26.low.ifits'
                    ]

        if frange == 'mid':
            vlim = [-800,800]
            xlim = [3,-7]
            ylim = [3,-7]
            rmfiles= [
                    base+'/3c279/frm.1253-055.x1-u1.2009_08_28.fits', 
                    base+'/3c279/frm.1253-055.x1-u1.2009_12_05.fits',
                    base+'/3c279/frm.1253-055.x1-u1.2010_01_26.fits'
                    ]
            rmefiles= [
                    base+'/3c279/frme.1253-055.x1-u1.2009_08_28.fits', 
                    base+'/3c279/frme.1253-055.x1-u1.2009_12_05.fits',
                    base+'/3c279/frme.1253-055.x1-u1.2010_01_26.fits'
                    ]
            contourfiles=[
                    basecont+'/1253-055/maps4rm/1253-055.X1.2009_08_28.mid.ifits',
                    basecont+'/1253-055/maps4rm/1253-055.X1.2009_12_05.mid.ifits',
                    basecont+'/1253-055/maps4rm/1253-055.X1.2010_01_26.mid.ifits'
                    ]

        if frange == 'hig':
            vlim = [-4000,5000]
            xlim = [3,-7]
            ylim = [3,-7]

            rmfiles= [
                    base+'/3c279/frm.1253-055.u1-q1.2009_08_28.fits', 
                    base+'/3c279/frm.1253-055.u1-q1.2009_12_05.fits',
                    base+'/3c279/frm.1253-055.u1-q1.2010_01_26.fits'
                    ]
#            
#            rmefiles= [
#                    base+'/3c279/frme.1253-055.u1-q1.2009_08_28.fits', 
#                    base+'/3c279/frme.1253-055.u1-q1.2009_12_05.fits',
#                    base+'/3c279/frme.1253-055.u1-q1.2010_01_26.fits'
#                    ]
#            rmfiles= [
#                    base+'/3c279/frm.3c279.u1-q1-hig.2009_08_28.fits', 
#                    base+'/3c279/frm.3c279.u1-q1-hig.2009_12_05.fits',
#                    base+'/3c279/frm.3c279.u1-q1-hig.2010_01_26.fits'
#                    ]
            
            rmefiles= [
#                    base+'/3c279/frme.3c279.u1-q1.2009_08_28.fits', 
#                    base+'/3c279/frme.3c279.u1-q1.2009_12_05.fits',
#                    base+'/3c279/frme.3c279.u1-q1.2010_01_26.fits'
            # tmp filler
                    base+'/3c279/frme.3c279.x1-u1.2009_08_28.fits', 
                    base+'/3c279/frme.3c279.x1-u1.2009_12_05.fits',
                    base+'/3c279/frme.3c279.x1-u1.2010_01_26.fits'
                    ]
            contourfiles=[
                    basecont+'/1253-055/maps4rm/1253-055.U1.2009_08_28.hig.ifits',
                    basecont+'/1253-055/maps4rm/1253-055.U1.2009_12_05.hig.ifits',
                    basecont+'/1253-055/maps4rm/1253-055.U1.2010_01_26.hig.ifits'
                    ]

    return rmfiles, rmefiles, contourfiles, vlim, xlim, ylim



def rm_along_ridgeline(fits1, fits2, vlim, xlim, ylim, fits_err=None, logger=None, frange=None):
    '''Plots RM map with ridgeline. Plots map values along the ridgeline.
    The ridgeline is calculated based on map in fits1. 
    Values are taken from the map in fits2.
    
    Args:
        fits1:
            HDUList object for contours
        fits2: 
            HDUList object for values
        vlim ([float]):
            limits for the values to plot
        xlim ([float]):
            x limits   
        ylim ([float]):
            y limits   
        
    
    '''
    if logger is None:
        logger = create_logger('null', dest=[], levels=[])

    rr = Ridgeline()
    rad_max_pos = rr.make_ridgeline(fits1, steps = 100, smooth_factor=1.3, method='max')
    rr.proceed_ridgeline(remove_jumps=False, factor=6.)
    rm_ridgeline = rr.apply_ridgeline(fits2).values

    fig, ax = plot_rm_fits(fits2, fits1, fits_err, 
                  vlim=vlim, xlim=xlim, ylim=ylim, 
                  interactive=False,
                  ridgeline_new=rm_ridgeline,
                  logger=logger,
                  suffix=frange)

    figr, axr = plt.subplots(1,1)
    axr.plot(rr.rr.distance, rr.rr.v, '-o') # should be distance along the ridge line, not radius
    # axr.set_yscale('log')
    axr.set_xlabel('Distance along the ridge line [pix]')
    axr.set_ylabel('Pixel intensity [Jy/beam?]')
    
    return rr
    
    
def doit(source='3c273', frange='mid', epoch_num=None):
    """Perform map averaging across epochs and then analyzing ridgeline RM values.
    
    Args:
        source (str):
            source name
        frange (str):
            frequency range. One of (low, mid, hig)
        epoch_num (int):
            epoch number. [0,1,2,3] for 3C273. Select only this epoch.
            TODO: make it more general
    """
    
    logger.info('Doing it for source = {}, frange = {}'.format(source, frange))
    
    rmfiles, rmefiles, contourfiles, vlim, xlim, ylim = make_filelist_etc(base, basecont, source, frange)
    logger.info('Made filelists for the source {} at {} frequency range'.format(source, frange))
    # logger.info(contourfiles)
    
    if epoch_num is not None:
        rmfiles = [rmfiles[epoch_num]]
        rmefiles = [rmefiles[epoch_num]]
        contourfiles = [contourfiles[epoch_num]]
        
    logger.info('RM file[0]: {}'.format(rmfiles[0]))
    logger.info('CONT file[0]: {}'.format(contourfiles[0]))
        
    rm_maps = list(map(lambda x: read_fits(x)[0].data.squeeze() ,rmfiles))
    rme_maps = list(map(lambda x: read_fits(x)[0].data.squeeze() ,rmefiles))
    cont_maps = list(map(lambda x: read_fits(x)[0].data.squeeze() ,contourfiles))

    logger.debug('rm_maps size is {} '.format(len(rm_maps)))
    
    # average RM
    average_rm_fits = read_fits(rmfiles[0]) # contains metadata. The data will be replaced with average
    average_rme_fits = read_fits(rmfiles[0]) # contains metadata. The data will be replaced with average
    average_cont_fits = read_fits(contourfiles[0]) # contains metadata. The data will be replaced with average
    smooth_cont_fits = read_fits(contourfiles[0]) # contains metadata. The data will be replaced with average
    
    
    # TODO: do it properly with astropy FITS objects
    average_rm, average_rme = average_rm_map(rm_maps, rme_maps, logger=logger)
    average_rm[average_rm < -4999.0]=np.nan
    average_cont = average_contour_map(cont_maps, logger=logger)
    average_rm_4axes = np.expand_dims(average_rm, axis=(0,1))
    average_rme_4axes = np.expand_dims(average_rme, axis=(0,1))
    average_cont_4axes = np.expand_dims(average_cont, axis=(0,1))
    average_rm_fits[0].data = average_rm_4axes
    average_rme_fits[0].data = average_rme_4axes
    average_cont_fits[0].data = average_cont_4axes
    smooth_cont_fits[0].data = average_cont_4axes

    rr = rm_along_ridgeline(smooth_cont_fits, average_rm_fits, vlim, xlim, ylim, frange=frange)
    
    return rr 





# ============================================================================================================================

if __name__== "__main__":
    
    logger = create_logger('rm', dest=['rm4.log', 'STDERR'], levels=['DEBUG', 'INFO'])
    logger.info('Start working. Task 3C273 slices along the jet'.format(dt.datetime.now()))
    
    computer_name = platform.node()
    logger.debug('Computer name is {}'.format(computer_name))
    if computer_name == 'vlb098': # desktop 
        base='/homes/mlisakov/data/S2087A/polar/final_effort'
        basecont = '/homes/mlisakov/data/S2087A/polar'
        tmp =  '/homes/mlisakov/tmp'
        plots = '/homes/mlisakov/data/S2087A/polar/plots'
    else: # laptop    
        base = '/home/mikhail/sci/pol/final_effort'
        basecont = '/home/mikhail/sci/pol'
        tmp =  '/home/mikhail/tmp'
        plots = '/home/mikhail/sci/pol/plots'
    
    source = '3c273'
    frange = 'mid'
    


    # DISCLAIMER: 
        # Since we know that the sheath in 3C 273 is in reality wider than the jet
        # at a single epoch, RM along a ridgeline will change just because the ridgeline 
        # will go through different portions on the wide channel at different 
        # separation from the apparent core. This is dur to the jet being not a
        # rigid body. 
        # So, in order to trace evolution of the RM with separation from the core, one should 
        # use straight radial slices instead of a ridgeline. Counter-intuitive, but true. 
        
    #     
    
    s = Slice()
    rmfiles, rmefiles, contourfiles, vlim, xlim, ylim = make_filelist_etc(base, basecont, source, frange)
    cc = read_fits(contourfiles[0])
    rm = read_fits(rmfiles[0])
    # v = s.make_slice(cc, (0,0), (16,-10), npoints=200)
    v = s.make_slice(rm, (0,0), (7,-13), npoints=200)   # CHECK COORDINATES 
    # BOTH ON THE MAP AND IN GETTING VALEUS!!!!
    #
    #
    #
    #
    s.plot_slice() 
    fig, ax = plot_rm_fits(read_fits(rmfiles[0]), read_fits(contourfiles[0]), read_fits(rmefiles[0]), 
              vlim=vlim, xlim=xlim, ylim=ylim, 
              interactive=False,
              logger=logger,
              slice_coord=s,
              suffix=frange)


    # proceed three freq ranges separately: 
    # average all dates together, make a ridgeline, plot everything
    #     
    # rr1 = doit(source=source, frange='low')
    # rr2 = doit(source=source, frange='mid')
    # rr3 = doit(source=source, frange='hig')

    
    #proceed epochs idividually
    # frange = 'mid'
    # logger.info('\n' + '='*50 + '\nsource = {} frange = {} epoch_num = {}\n'.format(source, frange, 0) + '='*50)
    # rr1 = doit(source=source, frange=frange, epoch_num=0)
    # logger.info('\n' + '='*50 + '\nsource = {} frange = {} epoch_num = {}\n'.format(source, frange, 1) + '='*50)
    # rr2 = doit(source=source, frange=frange, epoch_num=1)
    # logger.info('\n' + '='*50 + '\nsource = {} frange = {} epoch_num = {}\n'.format(source, frange, 2) + '='*50)
    # rr3 = doit(source=source, frange=frange, epoch_num=2)
    # logger.info('\n' + '='*50 + '\nsource = {} frange = {} epoch_num = {}\n'.format(source, frange, 3) + '='*50)
    # rr4 = doit(source=source, frange=frange, epoch_num=3)

    


    

    
