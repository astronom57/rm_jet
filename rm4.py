#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:49:22 2020

@author: mlisakov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019-08-28 

script to make faraday rotation maps

@author: mikhail
"""

import os
#print(os.environ['HOME'])


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

#from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import label, find_objects

#from rm2lib import *
from rm2lib import loglevs, limits_mas2pix, make_transform
from deconvolve import make_beam




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
                                 'frequency'],
                        data = [[header['CRPIX1'],header['CRPIX1'],
                                header['CDELT1'],header['CDELT1'], 
                                header['NAXIS1'],header['NAXIS2'],
                                header['BMAJ'],header['BMIN'],header['BPA'],
                                header['OBJECT'],header['DATE-OBS'],
                                header['CRVAL3']]])
    
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






    


def plot_rm(rmfile, contourfile, rmefile = None,  vlim = [-5000,5000], interactive = False, xlim = [4,-15], ylim = [4,-15], at = [0,0]):    
    '''function to plot RM from a fits file onto a contour map.
    Should have an interactive regime to plot RM values along a slice or at a given point
    '''
    
    print('IN PLOT_RM: at = {} {}'.format(at[0], at[1]))
    
    
    # test
    if 0:
        # laptop
        base = '/home/mikhail/sci/pol/final_effort'
        basecont = '/home/mikhail/sci/pol'
        
#        # work PC
#        base='/homes/mlisakov/data/S2087A/polar/final_effort'
#        basecont = '/homes/mlisakov/data/S2087A/polar'
        
        rmfiles =       [base+'/3c273/frm.1226+023.x1-u1.2009_08_28.fits']
        contourfiles =  [basecont+'/1226+023/maps/1226+023.U1.2009_08_28.ifits2']
        contourfiles =  [basecont+'/1226+023/maps4rm/1226+023.X1.2009_08_28.mid.ifits']
        
        rmfile = rmfiles[0]
        contourfile = contourfiles[0]
        
    
    s= re.search( 'c1-x2|x1-u1|u1-q1', rmfile)
    suffix=s.group(0)
    
#    print('\n'*30)
#    print('suffix = {}'.format(suffix))    
    
    
    rm = read_fits(rmfile)
    cont = read_fits(contourfile)
    
    print('RM file = {}\nCONT file= {}'.format(rmfile, contourfile))
    
    
    rmheader = rm[0].header
    
    try:
        rme = read_fits(rmefile)[0].data.squeeze()
    except:
        rme = np.zeros(rm[0].data.squeeze())

    contheader = cont[0].header
    
    rmdf = get_parameters(rmheader)
    contdf = get_parameters(contheader)


    # RM  
#    PIXEL_PER_MAS = 3.6*10**6 
#    
#    wr = wcs.WCS(naxis=2)
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
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RM value at ({},{}) = {}'.format(at[0], at[1], rm[0].data.squeeze()[-at[1], -at[0]]))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RMerr value at ({},{}) = {}'.format(at[0], at[1], rme[-at[1], -at[0]]))

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

    
    def __init__(self, line, ax, outslice = '/home/mikhail/tmp/slice.pkl', outbeam = '/home/mikhail/tmp/beam.pkl'):
#    def __init__(self, line, ax, outslice = '/homes/mlisakov/tmp/slice.pkl', outbeam = '/homes/mlisakov/tmp/beam.pkl'):
        
        self.line = line
        self.ax = ax            # try to get axes inside the object
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.outslice = outslice
        self.outbeam  = outbeam

    def from_coord(self, x1,y1, x2,y2 ):
        self.laststart = [x1,y1]
        self.laststop  = [x2,y2]
        
        print('laststart = {}'.format(self.laststart))
        
        rmdata = read_fits(rmfile)[0].data.squeeze()
        
        rmheader = read_fits(rmfile)[0].header
        self.source = rmheader['OBJECT']
        self.epoch = rmheader['DATE-OBS']
        
        
        
        rmedata = read_fits(rmefile)[0].data.squeeze()
        contdata = read_fits(contourfile)[0].data.squeeze()
        rmdata2slice = np.copy(rmdata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
        # test
#            rmdata2slice = np.copy(contdata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
        # test^^^
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
            rmdata = read_fits(rmfile)[0].data.squeeze()
            rmedata = read_fits(rmefile)[0].data.squeeze()
            contdata = read_fits(contourfile)[0].data.squeeze()
            rmdata2slice = np.copy(rmdata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
            rmedata2slice = np.copy(rmedata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
            
            # test
#            rmdata2slice = np.copy(contdata)      # same array as rmdata but with reverces indexes. Useful for taking proper slices since in imshow the image is reverted (i.e. origin = lower)
            # test^^^
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
        
        self.bbb = save_beam(get_parameters(get_header(read_fits(contourfile))), self.slice_direction*180/np.pi, pixel_size = F , minimize_N = True, filename = self.outbeam)
        print('Saved beam to {}'.format(self.outbeam))
        
        
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
    
    
    
def doit(rmfile,contourfile,vlim,xlim, ylim):
    '''plot RM figure to file'''
    fig,ax = plot_rm(rmfile=rmfile, contourfile=contourfile, vlim =vlim, xlim=xlim, ylim=ylim)
    filename = rmfile.replace('fits', 'pdf')
    plt.savefig(filename)
    print('saved RM figure to {}'.format(filename))
    return filename

# ============================================================================================================================

if __name__== "__main__":
    
    base= os.getenv('RMDATA')
    
    # laptop
    base = '/home/mikhail/sci/pol/final_effort'
    basecont = '/home/mikhail/sci/pol'
    tmp =  '/home/mikhail/tmp'
    plots = '/home/mikhail/sci/pol/plots'
    
    # work PC
#    base='/homes/mlisakov/data/S2087A/polar/final_effort'
#    basecont = '/homes/mlisakov/data/S2087A/polar'
#    tmp =  '/homes/mlisakov/tmp'
#    plots = '/homes/mlisakov/data/S2087A/polar/plots'

    print(base)    





    source = '3c273'
    source = '3c279'
    frange = 'mid'
    rmfiles = rmefiles = contourfiles = []

    # 3C 273 
    # low freq range
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
        #    # 3C 273 
        #    # mid freq range
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
                    
        ##    # 3C 273 
        ##    # hig freq range
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








    # run one by one
#    doit(rmfile,contourfile,vlim,xlim, ylim)
    
#    doit(base+'/3c273/frm.1226+023.c1-x2.2009_08_28.fits', basecont+'/1226+023/maps4rm/1226+023.C1.2009_08_28.low.ifits',[-800,800],  [4, -20],  [4, -20] )
#    doit(base+'/3c273/frm.1226+023.c1-x2.2009_10_25.fits', basecont+'/1226+023/maps4rm/1226+023.C1.2009_10_25.low.ifits',[-800,800],  [4, -20],  [4, -20] )
#    doit(base+'/3c273/frm.1226+023.c1-x2.2009_12_05.fits', basecont+'/1226+023/maps4rm/1226+023.C1.2009_12_05.low.ifits',[-800,800],  [4, -20],  [4, -20] )
#    doit(base+'/3c273/frm.1226+023.c1-x2.2010_01_26.fits', basecont+'/1226+023/maps4rm/1226+023.C1.2010_01_26.low.ifits',[-800,800],  [4, -20],  [4, -20] )
#    import sys
#    sys.exit()


    ab = ['a']*len(rmfiles)
    fig = ['a']*len(rmfiles)
    ax = ['a']*len(rmfiles)
    line = ['a']*len(rmfiles)


    xlims = [[],[]]
    ylims = [[],[]]
    




    
    
    for i,rmfile in enumerate(rmfiles): 
        #vlim = [-1000,3500]
#        vlim = [-600,600]
        
#        xlim = [2, -8]
#        ylim = [2, -8]
        
#        xlim = [4,-20]
#        ylim = [4,-20]
        
        contourfile = contourfiles[i]
        rmefile = rmefiles[i]
        print(rmfile)
        print(contourfiles[i])
        if source == '3c273':
            fig[i],ax[i] = plot_rm(rmfile=rmfile, contourfile=contourfiles[i], rmefile = rmefiles[i], vlim =vlim, xlim=xlim, ylim=ylim, at = [-530,-483])
        if source == '3c279':
            if frange == 'low':
                fig[i],ax[i] = plot_rm(rmfile=rmfile, contourfile=contourfiles[i], rmefile = rmefiles[i], vlim =vlim, xlim=xlim, ylim=ylim, at = [ -541, -490])
            if frange == 'mid':
                fig[i],ax[i] = plot_rm(rmfile=rmfile, contourfile=contourfiles[i], rmefile = rmefiles[i], vlim =vlim, xlim=xlim, ylim=ylim, at = [ -565, -481])

        line[i], = ax[i].plot([0], [0])  # empty line
#        ab = ArrowBuilder(line,ax, outslice = tmp+'/slice_B_hig.pkl', outbeam = tmp+'/beam_B_hig.pkl')
#        ab = ArrowBuilder(line,ax, outslice = tmp+'/slice_B_mid.pkl', outbeam = tmp+'/beam_B_mid.pkl')
#        ab = ArrowBuilder(line,ax, outslice = tmp+'/slice_B_low.pkl', outbeam = tmp+'/beam_B_low.pkl')
        
        ab[i] = ArrowBuilder(line[i],ax[i], outslice = tmp+'/slice_{}_mid.pkl'.format(i))
#        print('slice coordinates:\n{:.0f}, {:.0f}, {:.0f}, {:.0f}'.format(ab[i].laststart[0], ab[i].laststart[1], ab[i].laststop[0], ab[i].laststop[1] ) )


        
        if source == '3c273':
            # Manually edited below. The essential slices to show.
            # 3C 273. Slices across the jet at C1-X2 that shows no significant evolution of RM values 
    #        ab[i].from_coord(595, 416, 639, 471) # [6] <- even farther from the core
    #        ab[i].from_coord(574, 424, 622, 488) # [5] <- farther from the core
    #        ab[i].from_coord(561, 435, 606, 501) # [4] REFERENCE SLICE
    #        ab[i].from_coord(540, 449, 582, 517) # [3] <- closer to the core
    #        ab[i].from_coord(526, 460, 565, 523) # [2] <- even closer
    #        ab[i].from_coord(511, 470, 547, 529) # [1] <-even closer  : slight difference between A and B at the southern part of the jet, while E has these values blanked (not a good lambda^2 fit? )
    
            # 3C 273. A slice across the jet at X1-U1 that shows no significant evolution of RM values 
    #        ab[i].from_coord(595, 416, 639, 471) # [6] <- even farther from the core
    #        ab[i].from_coord(574, 424, 622, 488) # [5] <- farther from the core
    #        ab[i].from_coord(561, 435, 606, 501) # [4] REFERENCE SLICE
    #        ab[i].from_coord(540, 449, 582, 517) # [3] <- closer to the core
    #        ab[i].from_coord(520, 453, 564, 520) # [2] <- even closer
    #        ab[i].from_coord(511, 470, 547, 529) # [1] <-even closer
            
            # 3C 273. Slices across the jet at U1-Q1  (slices are different from those at low and mid). 
    #        ab[i].from_coord(506, 448, 570, 519 )  # [1] <- closer to the core
    #        ab[i].from_coord(513, 440, 576, 501 ) # [2] <- farther downstream
    #        ab[i].from_coord(521, 432, 580, 490 ) # [3] <- even farther downstream
    #        ab[i].from_coord(530, 424, 586, 484 ) # [4] <- even farther downstream
    #        ab[i].from_coord(536, 415, 591, 480 ) # [5] <- even farther downstream
    #        ab[i].from_coord(545, 408, 601, 470 ) # [6] <- even farther downstream
    #        ab[i].from_coord(550, 400, 603, 459 ) # [7] <- even farther downstream
    #        ab[i].from_coord(556, 392, 605, 450 ) # [8] <- even farther downstream
             
    #        ab[i].from_coord(515, 466, 546, 499)  # [9] <- a slice crossing the point (-530,-483) with high RM variations
    
        
        
        
            # 3C 273. Slices ALONG the jet at C1-X2
    #        ab[i].from_coord(510, 511, 594, 406) # [1] <- same as [1] at hig
    #        ab[i].from_coord(503, 515, 705, 370) # [2] <- a long slice along the jet
        
        
            # 3C 273. Slices ALONG the jet at X1-U1
    #        ab[i].from_coord(510, 511, 594, 406) # [1] <- same as [1] at hig
#            ab[i].from_coord(503, 515, 705, 370) # [2] <- a long slice along the jet
    
    
            # 3C 273. Slices ALONG the jet at U1-Q1
    #        ab[i].from_coord(510, 511, 594, 406) # [1] <- reference slice, approx in the middle of RM structure
    #        ab[i].from_coord(506, 488, 585, 389) # [2] <- to the south from the ref slice
    #        ab[i].from_coord(509, 475, 566, 470) # [3] <- inclined slice starting at a negative feature that is visible in epoch B
        
            pass
    
        if source == '3c279':
            if frange == 'low':
                # 3C 279. Slices across the jet at C1-X2
#                ab[i].from_coord( 500, 483, 531, 526 ) # [1] <- ref slice
    #            ab[i].from_coord( 506, 474, 541, 521 ) # [2] <- further downstream
    #            ab[i].from_coord( 516, 472, 553, 518 ) # [3]  <- further downstream
    #            ab[i].from_coord( 523, 463, 569, 510 ) # [4]  <- further downstream
    #            ab[i].from_coord( 530, 452, 578, 501 ) # [5]  <- further downstream
#                ab[i].from_coord( 541, 440, 589, 488 ) # [6]  <- further downstream
    
    
                # 3C 279. Slices ALONG the jet at C1-X2
                ab[i].from_coord(497, 510, 525, 508)    # [1] <- perpendicular to the dip line
#                ab[i].from_coord(494, 524, 575, 465)    # [2] <- long cut, middle of the jet
#                ab[i].from_coord(490, 516, 574, 452)    # [3] <- long cut, to the south of [2]
#                ab[i].from_coord(499, 534, 580, 475)    # [4] <- long cut, to the north of [2]
                pass
    
        
            if frange == 'mid':
                # 3C 279. Slices across the jet at X1-U1
#                ab[i].from_coord( 490, 489, 522, 542 ) # [1] <- ref slice
#                ab[i].from_coord( 500, 479, 532, 527 ) # [2] <- further downstream
#                ab[i].from_coord( 509, 473, 541, 514 ) # [3]  <- further downstream
#                ab[i].from_coord( 528, 464, 553, 508 ) # [4]  <- further downstream
#                ab[i].from_coord( 545, 449, 572, 504 ) # [5]  <- further downstream
#                ab[i].from_coord( 561, 427, 590, 490 ) # [6]  <- further downstream
#                ab[i].from_coord( 574, 412, 608, 475 ) # [7]  <- further downstream
#                ab[i].from_coord( 588, 401, 622, 460 ) # [8]  <- further downstream
                
                
                # 3C 279. Slices ALONG the jet at X1-U1
                ab[i].from_coord(493, 523, 613, 429)  # [1] <- a slice through the whole jet length
#                ab[i].from_coord(491, 508, 548, 467) # [2] <- a short slice of the southern edge wrt [3]
#                ab[i].from_coord(492, 522, 548, 482) # [3] <- a refernce short slice in the midlle line of the jet
#                ab[i].from_coord(497, 533, 547, 496) # [4] <- a short slice of the northern edge wrt [3]
                
                pass
            
            
            
            
            pass
    
    
    
#    line, = ax.plot([0], [0])  # empty line
#    ab = ArrowBuilder(line)





    # plot all slices alltogether in one figure
    figs, axs = plt.subplots(1,1)
    figs.suptitle(ab[0].source)
    markers = ['o', 'v', 's', '*']
    for i,rmfile in enumerate(rmfiles): 
#        axs.errorbar(np.arange(ab[i].diag_m.size), ab[i].diag_m, yerr = ab[i].diag_e, label = '{}'.format(ab[i].epoch), marker = markers[i])
        
        # mas-based plot
        axs.errorbar(ab[i].d.length, ab[i].diag_m, yerr = ab[i].diag_e)
        axs.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        axs.xaxis.set_minor_locator(AutoMinorLocator(5))
        
        
        
#        axs.set_xlim(0, self.diag_m.size)
#        axs.set_ylim(vlim[0], vlim[1])

    figs.legend()


    # RM difference 
    # 3C 273    
    if source == '3c273':
        r1= plot_rm_diff(rmfiles[1], rmfiles[0], contourfile = contourfiles[0], drm = 1000 )
        r2= plot_rm_diff(rmfiles[2], rmfiles[1], contourfile = contourfiles[0], drm = 1000 )
        r3= plot_rm_diff(rmfiles[3], rmfiles[2], contourfile = contourfiles[0], drm = 1000 )
        
    #    r= plot_rm_diff(rmfiles[3], rmfiles[1], contourfile = contourfiles[0], drm = 700 )
    
    #    r= plot_rm_diff(rmfiles[3], rmfiles[0], contourfile = contourfiles[0], drm = 10000 )
    #    r= plot_rm_diff(rmfiles[2], rmfiles[0], contourfile = contourfiles[0], drm = 10000 )
        print('The median difference between B and A is: {:.0f}'.format(np.nanmedian(r1[0].data.squeeze())))
        print('The median difference between C and B is: {:.0f}'.format(np.nanmedian(r2[0].data.squeeze())))
        print('The median difference between E and C is: {:.0f}'.format(np.nanmedian(r3[0].data.squeeze())))
    
    
    
        # plot RM changes at at = -530 -483 for hig freq range
        a = [1920.62109375, -2403.939453125, -1419.633544921875,2013.0157470703125]     # RM values
        ae = [ 405.8110046386719, 407.8999938964844, 392.44207763671875,534.901489257812   ] # RM errors
        b = ['2009-08-28', '2009-10-25' , '2009-12-05', '2010-01-26']  # dates
        c=[0]*len(b)
        from datetime import datetime
        for i,bb in enumerate(b):
            t = datetime.strptime(bb,'%Y-%m-%d')
            tt = t.timetuple()
            c[i] = tt.tm_year + tt.tm_yday/365
    
        figr, axr = plt.subplots(1,1)
    #    axr.plot(c, a)
        axr.errorbar(c, a , list(map(lambda x: x*1,ae)), color = 'blue', marker = 'o')




    if source == '3c279':
        drm =1000
        b = ['2009-08-28', '2009-12-05', '2010-01-26']  # dates
        c=[0]*len(b)
        from datetime import datetime
        for i,bb in enumerate(b):
            t = datetime.strptime(bb,'%Y-%m-%d')
            tt = t.timetuple()
            c[i] = tt.tm_year + tt.tm_yday/365
        
        
        if frange == 'low':
            drm = 250
            r1= plot_rm_diff(rmfiles[1], rmfiles[0], contourfile = contourfiles[0], drm = drm )
            r2= plot_rm_diff(rmfiles[2], rmfiles[1], contourfile = contourfiles[0], drm = drm )
            print('The median difference between C and A is: {:.0f}'.format(np.nanmedian(r1[0].data.squeeze())))
            print('The median difference between E and C is: {:.0f}'.format(np.nanmedian(r2[0].data.squeeze())))
            # plot RM changes at at =  -554, -495 for low freq range
            a = [  145.615  ,   16.48  ,   -181.9  ]     # RM values
            ae = [ 33.97   ,  35.1   , 44.6    ] # RM errors
     
        if frange =='mid':
            drm = 1000
            r1= plot_rm_diff(rmfiles[1], rmfiles[0], contourfile = contourfiles[0], drm = drm )
            r2= plot_rm_diff(rmfiles[2], rmfiles[1], contourfile = contourfiles[0], drm = drm )
            print('The median difference between C and A is: {:.0f}'.format(np.nanmedian(r1[0].data.squeeze())))
            print('The median difference between E and C is: {:.0f}'.format(np.nanmedian(r2[0].data.squeeze())))
            # plot RM changes at at =  -565, -481 for low freq range
            a = [  -139  ,  -93  ,  565 ]     # RM values
            ae = [  125 ,  122 ,   186  ] # RM errors

        
        
    
        figr, axr = plt.subplots(1,1)
    #    axr.plot(c, a)
        axr.errorbar(c, a , list(map(lambda x: x*1,ae)), color = 'blue', marker = 'o')




