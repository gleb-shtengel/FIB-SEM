import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

import pandas as pd
import os
from pathlib import Path
import time
import glob
import re

import matplotlib
import matplotlib.image as mpimg
from matplotlib import pylab, mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.core.pylabtools import figsize, getfigs

import psutil
import inspect

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from SIFT_gs.FIBSEM_custom_transforms_gs import *
except:
    raise RuntimeError("Unable to load FIBSEM_custom_transforms_gs")



######################################################
#    General Help Functions
######################################################

def swap_elements(a, i, j):
    '''
    Swaps tw elements of the array.
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com

    Parameters:
    a : array
    i : int
        Index of the first element to swap
    j : int
        Index of the second element to swap
    Returns: 
        array with i-th and j-th elements swapped
    '''
    b=a.copy()
    b[i]=a[j]
    b[j]=a[i]
    return b


def scale_image_gs(image, im_min, im_max, amplitude, **kwargs):
    '''
    Clips the image between the values im_min and im_max and then rescales it to fit the new range: 0 to amplituide.
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com

    Parameters:
    image : array
    im_min : float
        min value. The values below that will be set to im_min
    im_max : float
        max value. The values above that will be set to im_max
    amplitude : float
        the rescaled range for output value

    kwargs:
    invert : boolean
        If True, the data is inverted. Default is False
    Returns:
        data_spread : float

    '''
    invert = kwargs.get('invert', False)
    out = np.clip((image-im_min)*np.float(amplitude)/(im_max - im_min), 0, amplitude)
    if invert:
        out = amplitude-out
    out = out.astype(int)
    return out


def get_spread(data, window=501, porder=3):
    '''
    Calculates spread - standard deviation of the (signal - Sav-Gol smoothed signal).
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com

    Parameters:
    data : 1D array
    window : int
        aperture (number of points) for Sav-Gol filter)
    porder : int
        polynomial order for Sav-Gol filter

    Returns:
        data_spread : float

    '''

    try:
        #sm_data = savgol_filter(data.astype(np.double), window, porder)
        sm_data = savgol_filter(data.astype(np.double), window, porder, mode='mirror')
        data_spread = np.std(data-sm_data)
    except :
        print('spread error')
        data_spread = 0.0
    return data_spread


def get_min_max_thresholds(image, **kwargs):
    '''
    Determines the data range (min and max) with given fractional thresholds for cumulative distribution.
    ©G.Shtengel 11/2022 gleb.shtengel@gmail.com

    Calculates the histogram of pixel intensities of the image with number of bins determined by parameter nbins (default = 256)
    and normalizes it to get the probability distribution function (PDF), from which a cumulative distribution function (CDF) is calculated.
    Then given the thr_min, thr_max parameters, the minimum and maximum values of the data range are found
    by determining the intensities at which CDF= thr_min and (1- thr_max), respectively

    Parameters:
    ----------
    image : 2D array
        Image to be analyzed

    kwargs:
     ----------
    thr_min : float
        lower CDF threshold for determining the minimum data value. Default is 1.0e-3
    thr_max : float
        upper CDF threshold for determining the maximum data value. Default is 1.0e-3
    nbins : int
        number of histogram bins for building the PDF and CDF
    log  : bolean
        If True, the histogram will have log scale. Default is false
    disp_res : bolean
        If True display the results. Default is True.
    save_res : boolean
        If True the image will be saved. Default is False.
    dpi : int
        Default is 300
    save_filename : string
        the name of the image to perform this operations (defaulut is 'min_max_thresholds.png').

    Returns
    (dmin, dmax) : float array
    '''
    thr_min = kwargs.get('thr_min', 1.0e-3)
    thr_max = kwargs.get('thr_max', 1.0e-3)
    nbins = kwargs.get('nbins', 256)
    disp_res = kwargs.get('disp_res', True)
    log = kwargs.get('log', False)
    save_res = kwargs.get('save_res', False)
    dpi = kwargs.get('dpi', 300)
    save_filename = kwargs.get('save_filename', 'min_max_thresholds.png')

    if disp_res:
        fsz=11
        fig, axs = plt.subplots(2,1, figsize = (6,8))
        hist, bins, patches = axs[0].hist(image.ravel(), bins=nbins, log=log)
    else:
        hist, bins = np.histogram(image.ravel(), bins=nbins)
    pdf = hist / np.prod(image.shape)
    cdf = np.cumsum(pdf)
    data_max = bins[np.argmin(np.abs(cdf-(1.0-thr_max)))]
    data_min = bins[np.argmin(np.abs(cdf-thr_min))]
    
    if disp_res:
        xCDF = bins[0:-1]+(bins[1]-bins[0])/2.0
        xthr = [xCDF[0], xCDF[-1]]
        ythr_min = [thr_min, thr_min]
        ythr_max = [1-thr_max, 1-thr_max]
        axs[1].plot(xCDF, cdf, label='CDF')
        axs[1].plot(xthr, ythr_min, 'r', label='thr_min={:.5f}'.format(thr_min))
        axs[1].plot(xthr, ythr_max, 'g', label='1.0 - thr_max = {:.5f}'.format(1-thr_max))
        axs[1].set_xlabel('Intensity Level', fontsize = fsz)
        axs[0].set_ylabel('PDF', fontsize = fsz)
        axs[1].set_ylabel('CDF', fontsize = fsz)
        xi = data_min - (np.abs(data_max-data_min)/2)
        xa = data_max + (np.abs(data_max-data_min)/2)
        rys = [[0, np.max(hist)], [0, 1]]
        for ax, ry in zip(axs, rys):
            ax.plot([data_min, data_min], ry, 'r', linestyle = '--', label = 'data_min={:.1f}'.format(data_min))
            ax.plot([data_max, data_max], ry, 'g', linestyle = '--', label = 'data_max={:.1f}'.format(data_max))
            ax.set_xlim(xi, xa)
            ax.grid(True)
        axs[1].legend(loc='center', fontsize=fsz)
        axs[1].set_title('Data Min and max with thr_min={:.0e},  thr_max={:.0e}'.format(thr_min, thr_max), fontsize = fsz)
        if save_res:
            fig.savefig(save_filename, dpi=dpi)
    return np.array((data_min, data_max))

def argmax2d(X):
    return np.unravel_index(X.argmax(), X.shape)


def find_BW(fr, FSC, **kwargs):
    '''
    Find Cutoff Bandwidth using threshold parameter OSNRt. ©G.Shtengel 03/2024 gleb.shtengel@gmail.com
    Parameters
    ---------
    fr : array of Frequency points
    FSC : array of FSC data points
    
    kwargs
    ---------
    SNRt : float
        SNR threshold for determining the resolution bandwidth
    verbose : boolean
        print the outputs. Default is False
    fit_data : boolean
        If True the BW will be extracted fron inverse power fit.
        If False, the BW will be extracted from the data
    fit_power : int
        parameter for FSC data fitting: FSC_fit  = a/(x**fit_power+a)
    fr_cutoff : float
        The fractional value between 0.0 and 1.0. The data points within the frequency range [0 : max_frequency*cutoff]  will be used.

    Returns
        BW, fr_fit, FSC_fit
    
    '''
    SNRt = kwargs.get('SNRt', 0.143)
    verbose = kwargs.get('verbose', False)
    fit_data = kwargs.get('fit_data', False)
    fit_power = kwargs.get('fit_power', 3)
    fr_cutoff = kwargs.get('fr_cutoff', 0.9)
    def inverse_power(x, a):
        return a / (x**fit_power + a)
    
    ind_cutoff = int(np.shape(FSC)[0] * fr_cutoff)
    inds = np.arange(np.shape(FSC)[0])
    valid_inds = (np.isnan(FSC)==False) * (inds < ind_cutoff)
    
    fr_limited = fr[valid_inds]
    FSC_limited = FSC[valid_inds]
    
    if fit_data:
        if verbose:
            print('Using a/(x**{:d}+a) Fit of the Raw Data to determine BW'.format(fit_power))
        pguess = [1.0]
        # Fit the data
        popt, pcov = curve_fit(inverse_power, fr_limited, FSC_limited, p0 = pguess)
        a = popt[0]
        BW = (a/SNRt - a)**(1.0/fit_power)
        fr_fit = np.linspace(fr_limited[0], BW*1.1, 25)
        FSC_fit = inverse_power(fr_fit, popt)
        fitOK = 0
    else:
        if verbose:
            print('Using Raw Data to determine BW')
        fr_fit = fr_limited
        FSC_fit = FSC_limited
        npts = len(fr_fit)
        j = 1
        while (j<npts-1) and FSC_limited[j]>SNRt:
            j = j+1
        if j >= npts-1:
            if verbose:
                print('Cannot determine BW accurately: not enough points')
            BW = fr_limited[j]
            fitOK = 1
        else:
            BW = fr_limited[j-1] + (fr_limited[j]-fr_limited[j-1])*(SNRt-FSC_limited[j-1])/(FSC_limited[j]-FSC_limited[j-1])
            fitOK = 0
    if verbose:
        print('BW = {:.3f}'.format(BW))
    return BW, fr_fit, FSC_fit, fitOK


def find_histogram_FWHM(cnts, bins, **kwargs):
    '''
    find FWHM of the histogram. g.Shtengel 09.2024
    
    Parameters:
    cnts : array or list
        histogram counts
    bins : array or list
        histogram bins
        
    kwargs:
    start : string
        'edges' (default) or 'center'. start of search.
    estimation : string
        'interval' (default) or 'count'. Returns a width of interval determied using search direction from above or total number of bins above half max
    max_aver_aperture : int
        Aperture for averaging the Max signal
    verbose : boolean
        display output. Defaults is False.
    Returns:
    FWHM, indi, inda, mx, mx_ind
    '''
    start = kwargs.get('start', 'edges')
    estimation = kwargs.get('estimation', 'interval')
    max_aver_aperture = kwargs.get('max_aver_aperture', 5)
    verbose = kwargs.get('verbose', False)
    
    mx_ind = np.argmax(np.array(cnts))
    ii = np.max((mx_ind-max_aver_aperture//2, 0))
    ia = np.min((ii+max_aver_aperture, len(cnts)))
    ii = ia-max_aver_aperture
    mx = np.mean(cnts[ii:ia])
    cnts_mod = cnts - mx/2.0
    db = bins[1]-bins[0]
    
    if estimation == 'count':
        FWHM = np.sum(i > 0.0 for i in cnts_mod) * db
        indi = np.argmax(cnts_mod > 0.0) + 1
        inda = len(cnts) - np.argmax(np.flip(cnts_mod) > 0.0)
    else:
        if start == 'edges':
            indi = np.argmax(cnts_mod > 0.0) + 1
            inda = len(cnts) - np.argmax(np.flip(cnts_mod) > 0.0) 
        else:
            ln1 = len(cnts_mod[0:mx_ind])
            indi = ln1 - np.argmax(np.flip(cnts_mod[0:mx_ind]) < 0.0) +1
            inda = ln1 + np.argmax(cnts_mod[mx_ind:] < 0.0)
        FWHM = (inda-indi) * db
    if verbose:
        print('FWHM, ' + estimation + ', ' + start+ ' = {:.2f}'.format(FWHM))

    return FWHM, indi, inda, mx, mx_ind


def radial_profile(data, center):
    '''
    Calculates radially average profile of the 2D array (used for FRC and auto-correlation)
    ©G.Shtengel 08/2020 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array

    center : list of two ints
        [xcenter, ycenter]

    Returns
        radialprofile : float array
            limited to x-size//2 of the input data
    '''
    ysz, xsz = data.shape
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = (np.array(tbin) / nr)[0:(xsz//2+1)]
    return radialprofile


def radial_profile_select_angles(data, center, **kwargs):
    '''
    Calculates radially average profile of the 2D array (used for FRC) within a select range of angles.
    ©G.Shtengel 08/2020 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array
    center : list of two ints
        [xcenter, ycenter]

    kwargs
    astart : float
        Start angle for radial averaging. Default is 0
    astop : float
        Stop angle for radial averaging. Default is 90
    rstart : float
        Start radius
    rstop : float
        Stop radius
    symm : int
        Symmetry factor (how many times Start and stop angle intervals are repeated within 360 deg). Default is 4.

    Returns
        radialprofile : float array
            limited to x-size//2 of the input data
    '''
    astart = kwargs.get('astart', -1.0)
    astop = kwargs.get('astop', 1.0)
    rstart = kwargs.get('rstart', 0.0)
    rstop = kwargs.get('rstop', 1.0)
    symm = kwargs.get('symm', 4)
    
    ds = data.shape
    y, x = np.indices(ds)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    rmin = np.max(r)*rstart
    rmax = np.max(r)*rstop
    r_ang = (np.angle(x - center[0]+1j*(y - center[1]), deg=True))
    r = r.astype(int)
    
    cond_tot = np.zeros(ds, dtype=float)
    
    if symm>0:
        for i in np.arange(symm*2):
            ai = max((min(((astart + 360.0/symm*i-360), 180.000001)), - 180.000001))
            aa = max((min(((astop  + 360.0/symm*i-360), 180.000001)), - 180.000001))
            if np.abs(ai-aa)>1e-10:
                #print(ai, aa)
                cond = (r_ang > ai)*(r_ang < aa)
                cond_tot[cond] = 1.0
    else:
        cond = (r_ang > astart)*(r_ang < astop)
        cond_tot[cond] = 1.0
    
    rr = np.ravel(r[cond_tot>0])
    dd = np.ravel(data[cond_tot>0])

    tbin = np.bincount(rr, dd)
    nr = np.bincount(rr)
    radialprofile = tbin / nr
    return radialprofile


def build_kernel_FFT_zero_destreaker_radii_angles(data, **kwargs):
    '''
    Build a Rescaler Kernel for the FFT data within a select range of angles.
    ©G.Shtengel 10/2023 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array

    kwargs
    astart : float
        Start angle for radial segment. Default is -1.0.
    astop : float
        Stop angle for radial segment. Default is 1.0.
    rstart : float
        Low bound for spatial frequencies in FFT space.
    rstop : float
        High bound for spatial frequencies in FFT space.
    symm : int
        Symmetry factor (how many times Start and stop angle intervals are repeated within 360 deg). Default is 2.

    Returns
        rescaler : float array
    '''
    astart = kwargs.get('astart', -1.0)
    astop = kwargs.get('astop', 1.0)
    rstart = kwargs.get('rstart', 0.0)
    rstop = kwargs.get('rstop', 1.0)
    symm = kwargs.get('symm', 2)
    
    ds = data.shape
    y, x = np.indices(ds)
    yc, xc = ds[0]//2, ds[1]//2
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    rmin = np.max(r) * rstart
    rmax = np.max(r) * rstop
    r_ang = (np.angle(x - xc+1j*(y - yc), deg=True))
    
    rescaler = np.ones(ds, dtype=float)
    
    if symm>0:
        for i in np.arange(symm*2):
            ai = max((min(((astart + 360.0/symm*i-360), 180.000001)), - 180.000001))
            aa = max((min(((astop  + 360.0/symm*i-360), 180.000001)), - 180.000001))
            if np.abs(ai-aa)>1e-10:
                #print(ai, aa)
                cond = (r_ang > ai)*(r_ang < aa)*(r >= rmin)*(r <= rmax)
                rescaler[cond] = 0.0
    else:
        cond = (r_ang > astart)*(r_ang < astop)*(r >= rmin)*(r <= rmax)
        rescaler[cond] = 0.0
    return rescaler


def rescale_FFT_select_radii_angles(data, scale, center, **kwargs):
    '''
    Rescales the FFT data within a select range of angles.
    ©G.Shtengel 10/2023 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array
    center : list of two ints
        [xcenter, ycenter]

    kwargs
    astart : float
        Start angle for radial segment. Default is -1.0.
    astop : float
        Stop angle for radial segment. Default is 1.0.
    rstart : float
        Low bound for spatial frequencies in FFT space.
    rstop : float
        High bound for spatial frequencies in FFT space.
    symm : int
        Symmetry factor (how many times Start and stop angle intervals are repeated within 360 deg). Default is 2.

    Returns
        radialprofile : float array
            limited to x-size//2 of the input data
    '''
    astart = kwargs.get('astart', -1.0)
    astop = kwargs.get('astop', 1.0)
    rstart = kwargs.get('rstart', 0.0)
    rstop = kwargs.get('rstop', 1.0)
    symm = kwargs.get('symm', 2)
    
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    rmin = np.max(r)*rstart
    rmax = np.max(r)*rstop
    r_ang = (np.angle(x - center[0]+1j*(y - center[1]), deg=True))
    
    r = r.astype(int)  
    newdata = data.copy()
    if symm>0:
        for i in np.arange(symm*2):
            ai = max((min(((astart + 360.0/symm*i-360), 180.000001)), - 180.000001))
            aa = max((min(((astop  + 360.0/symm*i-360), 180.000001)), - 180.000001))
            if np.abs(ai-aa)>1e-10:
                #print(ai, aa)
                cond = (r_ang > ai)*(r_ang < aa)*(r >= rmin)*(r <= rmax)
                newdata[cond] = data[cond] * scale[cond]
    else:
        cond = (r_ang > astart)*(r_ang < astop)*(r >= rmin)*(r <= rmax)
        newdata[cond] = data[cond] * scale[cond]
    return newdata


def add_scale_bar(ax, **kwargs):
    '''
    Add a scale bar to the existing plot.
    ©G.Shtengel 10/2023 gleb.shtengel@gmail.com

    Parameters:
    ax : axis object

    kwargs:
    bar_length_um : float
        length of the scale bar (um). Default is 1um.
    pixel_size_um : float
        Pixel size in um. Default is 0.004 um.
    loc : (float, float)
        bar location in fractional axis coordinates (0, 0) is left bottom corner. (1, 1) is top right corner.
        Default is (0.07, 0.93)
    bar_width : float
        width of the scale bar. Defalt is 3.0
    bar_color : color
        color of the scale bar. Defalt is 'white'
    label : string
        scale bar label. Default is length in um.
    label_color : color
        color of the scale bar label. Defalt is the same as bar_color.
    label_font_size : int
        Font Size of the scale bar label. Defalt is 12
    label_offset : int
        Additional vertical offset for the label position. Defalt is 0.
    '''
    
    bar_length_um = kwargs.get('bar_length_um', 1.0)
    pixel_size_um = kwargs.get('pixel_size_um', 0.004)
    loc = kwargs.get('loc', (0.07, 0.93))
    bar_width = kwargs.get('bar_width', 3.0)
    bar_color = kwargs.get('bar_color', 'white')
    bar_label = kwargs.get('bar_label', '{:.1f} μm'.format(bar_length_um))
    label_color = kwargs.get('label_color', bar_color)
    label_font_size = kwargs.get('label_font_size', 12)
    label_offset = kwargs.get('label_offset', 0)
    bar_length_image_pixels = bar_length_um / pixel_size_um # length of scale bar in pixels
    
    dpi = ax.get_figure().dpi
    xi_pix, xa_pix = ax.get_xlim()
    dx_pix = (xa_pix - xi_pix)  # size in image pixels
    yi_pix, ya_pix = ax.get_ylim()
    dy_pix = (ya_pix - yi_pix)
    
    y1, y2 = ax.get_window_extent().get_points()[:, 1]
    x1, x2 = ax.get_window_extent().get_points()[:, 0]
    yscale = np.abs(y2-y1) / np.abs(dy_pix)                 # Get unit scale
    #print(np.abs(dy_pix), np.abs(y2-y1))
    #print(yscale, label_font_size/yscale)
  
    dx_um = dx_pix * pixel_size_um                  # X-width of plot im um     
    xi_bar = xi_pix + loc[0] * dx_pix
    xa_bar = xi_bar + bar_length_image_pixels
    yi_bar = yi_pix + loc[1] * dy_pix
    ax.plot([xi_bar, xa_bar], [yi_bar, yi_bar], color = bar_color, linewidth = bar_width)
    label_font_size_scaled = label_font_size
    
    xi_text = xi_bar
    yi_text = yi_bar + bar_width + label_offset + 6.0 * label_font_size
    #print(yi_bar, yi_text, label_font_size)
    ax.text(xi_text, yi_text, bar_label, color = label_color, fontsize = label_font_size)


def clip_pad_image(orig_img, data_min, data_max, **kwargs):
    '''
    Clips the image and adds pads for the clipped margins. ©G.Shtengel 10/2024 gleb.shtengel@gmail.com
    
    Parameters:
    orig_img : 2D array
        original image
    data_min : float
        Low bound for determinung the real data edges
    data_max : float
        High bound for determinung the real data edges
    
    kwargs:
        clip_mask : 2D array
        if supplied, it will be used instead of data_min and data max criteria
    Returns: padded_img, clip_mask
    '''
    clip_mask = kwargs.get('clip_mask', (orig_img>data_min)*(orig_img<data_max))
    ny, nx = np.shape(clip_mask)

    try:
        yi = np.min(np.where(clip_mask[:, nx//2]))
    except:
        yi = 0
    try:
        ya = ny-np.max(np.where(clip_mask[:, nx//2]))
    except:
        ya = 0
    try:
        xi = np.min(np.where(clip_mask[ny//2, :]))
    except:
        xi = 0
    try:
        xa = nx-np.max(np.where(clip_mask[ny//2, :]))
    except:
        xa = 0

    pad_width = np.max((xi, xa, yi, ya))
    if pad_width > 0:
        #padded_img = clip_mask*orig_img + (1-clip_mask)*np.pad(orig_img[pad_width:-pad_width, pad_width:-pad_width], pad_width = pad_width, mode='symmetric')
        if xa>0 and ya>0:
            clip_mask[yi:-ya, xi:-xa] = True  # make sure clip_mask is filled with ones (True)
            padded_img = clip_mask*orig_img + (1-clip_mask)*np.pad(orig_img[yi:-ya, xi:-xa], pad_width = np.array([[yi, ya], [xi, xa]]), mode='symmetric')
        if xa>0 and ya==0:
            clip_mask[yi:, xi:-xa] = True  # make sure clip_mask is filled with ones (True)
            padded_img = clip_mask*orig_img + (1-clip_mask)*np.pad(orig_img[yi:, xi:-xa], pad_width = np.array([[yi, ya], [xi, xa]]), mode='symmetric')
        if ya>0 and xa==0:
            clip_mask[yi:-ya, xi:] = True  # make sure clip_mask is filled with ones (True)
            padded_img = clip_mask*orig_img + (1-clip_mask)*np.pad(orig_img[yi:-ya, xi:], pad_width = np.array([[yi, ya], [xi, xa]]), mode='symmetric')
        if ya==0 and xa==0:
            clip_mask[yi:, xi:] = True  # make sure clip_mask is filled with ones (True)
            padded_img = clip_mask*orig_img + (1-clip_mask)*np.pad(orig_img[yi:, xi:], pad_width = np.array([[yi, ya], [xi, xa]]), mode='symmetric')
    else:
        padded_img = orig_img

    return padded_img, clip_mask


def merge_images_with_transition(img1, img2, **kwargs):
    '''
    Merges two images with smooth transition.
    ©G.Shtengel 09/2024 gleb.shtengel@gmail.com

    Parameters:
    img1 : 2D array
    img2 : 2D array

    kwargs
    transition_direction : str
        'Y' (default), or 'X'.
    flip_transitionY : boolean
        Default is False. flip transition in Y direction.
    flip_transitionX : boolean
        Default is False. flip transition in X direction
    xi : int
        Start index of transion if transition_direction is 'X'. Default is 1/2 of the image.
    xa : int
        Stop index of transion if transition_direction is 'X'. Default is 3/4 of the image.
    yi : int
        Start index of transion if transition_direction is 'Y'. Default is 1/2 of the image.
    ya : int
        Stop index of transion if transition_direction is 'Y'. Default is 3/4 of the image.

    Returns
        composite image : 2D array
    '''
    transition_direction = kwargs.get('transition_direction', 'Y')
    flip_transitionY = kwargs.get('flip_transitionY', False)
    flip_transitionX = kwargs.get('flip_transitionX', False)
    
    ds = img1.shape
    y, x = np.indices(ds)
    
    if transition_direction == 'Y':
        yi = kwargs.get('yi', ds[0]//2)
        ya = kwargs.get('ya', ds[0]//4*3)
        r = (1.0 + np.tanh((y - (ya+yi)/2)/(ya-yi)*2.0))/2.0
    else:
        xi = kwargs.get('xi', ds[1]//2)
        xa = kwargs.get('xa', ds[1]//4*3)
        r = (1.0 + np.tanh((x - (xa+xi)/2)/(xa-xi)*2.0))/2.0
    if flip_transitionY:
        r = np.flip(r, axis=0)
    if flip_transitionX:
        r = np.flip(r, axis=1)
    return img1 * r  + (img2 + np.mean(img1-img2))*(1.0-r)


def build_kernel_FFT_zero_destreaker_radii_angles(data, **kwargs):
    '''
    Builds a de-streaking kernel to zero FFT data within a select range of angles.
    ©G.Shtengel 10/2023 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array

    kwargs
    astart : float
        Start angle for radial segment. Default is -1.0.
    astop : float
        Stop angle for radial segment. Default is 1.0.
    rstart : float
        Low bound for spatial frequencies in FFT space.
    rstop : float
        High bound for spatial frequencies in FFT space.
    symm : int
        Symmetry factor (how many times Start and stop angle intervals are repeated within 360 deg). Default is 2.

    Returns
        rescaler_kernel : float array
    '''
    astart = kwargs.get('astart', -1.0)
    astop = kwargs.get('astop', 1.0)
    rstart = kwargs.get('rstart', 0.0)
    rstop = kwargs.get('rstop', 1.0)
    symm = kwargs.get('symm', 2)
    
    ds = data.shape
    y, x = np.indices(ds)
    yc, xc = ds[0]//2, ds[1]//2
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    rmin = np.max(r) * rstart
    rmax = np.max(r) * rstop
    r_ang = (np.angle(x - xc+1j*(y - yc), deg=True))
    
    rescaler_kernel = np.ones(ds, dtype=float)
    
    if symm>0:
        for i in np.arange(symm*2):
            ai = max((min(((astart + 360.0/symm*i-360), 180.000001)), - 180.000001))
            aa = max((min(((astop  + 360.0/symm*i-360), 180.000001)), - 180.000001))
            if np.abs(ai-aa)>1e-10:
                #print(ai, aa)
                cond = (r_ang > ai)*(r_ang < aa)*(r >= rmin)*(r <= rmax)
                rescaler_kernel[cond] = 0.0
    else:
        cond = (r_ang > astart)*(r_ang < astop)*(r >= rmin)*(r <= rmax)
        rescaler_kernel[cond] = 0.0
    return rescaler_kernel


def build_kernel_FFT_zero_destreaker_XY(data, **kwargs):
    '''
    Builds a de-streaking kernel to zero FFT data within a select ranges of x and y.
    ©G.Shtengel 11/2023 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array

    kwargs
    
    xstart : float
        Low bound on X- spatial frequencies in FFT space
    xstop : float
        High bound on X- spatial frequencies in FFT space
    dy : float
        Width of Y- spatial frequency band in FFT space
    y_offset : float
        vertical offset of the band in Y- spatial frequency FFT space. offset applied in a symmetric fashion - it is of opposite sign in the negative x -space.
    symm : int
        Symmetry factor (how many times Start and stop angle intervals are repeated within 360 deg). Default is 4.

    Returns
        rescaler_kernel : float array
    '''
    xstart = kwargs.get('xstart', 0.1)
    xstop = kwargs.get('xstop', 1.0)
    dy = kwargs.get('dy', 0.01)
    y_offset = kwargs.get('y_offset', 0.00)
    symm = kwargs.get('symm', 4)
    ds = data.shape
    yc, xc = ds[0]//2, ds[1]//2
    
    y, x = np.indices(ds)
    xa = np.abs(x - xc) / np.max(x - xc)
    ya = np.abs((y - yc) / np.max(y - yc) + y_offset*np.sign(x - xc))
    
    rescaler_kernel = np.ones(ds, dtype=float)
    cond = (xa > xstart)*(xa < xstop)*(ya<=dy/2.0)
    rescaler_kernel[cond] = 0.0
    
    return rescaler_kernel


def build_kernel_FFT_destreaker_autodetect(data, **kwargs):
    '''
    Builds a de-streaking kernel to zero FFT data within a select range of angles.
    ©G.Shtengel 10/2023 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array

    kwargs
    astart_reference : float
        Start angle for radial segment for analysis. Default is 2.0.
    astop_reference : float
        Stop angle for radial segment for analysis. Default is 88.0.
    thr_reference: float
        Threshold for referencing. Default is 1.1
    symm_reference : int
        Symmetry factor (how many times Start and stop angle intervals are repeated within 360 deg). Default is 4.
    astart_limit : float
        Start angle for radial segment for analysis. Default is -5.0. Only perfrom autodetection in this range
    astop_limit : float
        Stop angle for radial segment for analysis. Default is 5.0. Only perfrom autodetection in this range
    rstart_limit : float
        Low bound for spatial frequencies in FFT space. Default is 0.01. Only perfrom autodetection in this range.
    rstop_limit : float
        High bound for spatial frequencies in FFT space. Default is 0.50. Only perfrom autodetection in this range.
    symm_limit : int
         Symmetry factor for autodetection limit. Default is 2.
    rescale : boolean
        If False, rescaler is 0 in the "suspect areas". If True, they are scaeled down according to the FFT mag.

    Returns
        rescaler_kernel : float array
    '''
    astart_reference = kwargs.get('astart_reference', 2.0)
    astop_reference = kwargs.get('astop_reference', 88.0)
    astart_limit = kwargs.get('astart_limit', -5.0)
    astop_limit = kwargs.get('astop_limit', 5.0)
    rstart_limit = kwargs.get('rstart_limit', 0.01)
    rstop_limit = kwargs.get('rstop_limit', 0.50)
    symm_reference = kwargs.get('symm_reference', 4)
    thr_reference = kwargs.get('thr_reference', 2.0)
    smooth_sigma = kwargs.get('smooth_sigma', 2.0)
    symm_limit = kwargs.get('symm_limit', 2)
    rescale = kwargs.get('rescale', False)
    
    abs_data = np.abs(data)
    ds = abs_data.shape
    y, x = np.indices(ds)
    yc, xc = ds[0]//2, ds[1]//2

    ref = radial_profile_select_angles(abs_data, [xc, yc], astart=astart_reference, astop = astop_reference, symm=symm_reference)
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    ref_array = np.interp(r, np.arange(len(ref)), ref, left=ref[0], right=ref[-1])

    cr = np.nan_to_num(abs_data/ref_array, nan=1.0)
    cr_smoothed = gaussian_filter(cr, smooth_sigma)
    rescaler_kernel = np.ones(ds, dtype=float)
    rescaler_kernel[cr_smoothed>thr_reference] = 0.0
        
    rmin = np.max(r) * rstart_limit
    rmax = np.max(r) * rstop_limit
    r_ang = (np.angle(x - xc+1j*(y - yc), deg=True))
    
    limit_kernel = np.ones(ds, dtype=float)
    if symm_limit>0:
        for i in np.arange(symm_limit*2):
            ai = max((min(((astart_limit + 360.0/symm_limit*i-360), 180.000001)), - 180.000001))
            aa = max((min(((astop_limit  + 360.0/symm_limit*i-360), 180.000001)), - 180.000001))
            if np.abs(ai-aa)>1e-10:
                #print(ai, aa)
                cond = (r_ang > ai)*(r_ang < aa)*(r >= rmin)*(r <= rmax)
                limit_kernel[cond] = 0.0
    else:
        cond = (r_ang > astart_limit)*(r_ang < astop_limit)*(r >= rmin)*(r <= rmax)
        limit_kernel[cond] = 0.0
    rescaler_kernel = np.clip((rescaler_kernel + limit_kernel), 0.0, 1.0)
    if rescale:
        rescaler_kernel = np.clip((rescaler_kernel +  (1.0-rescaler_kernel)/cr_smoothed), 0.0, 1.0)

    return rescaler_kernel


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2-1):-(window_len//2)]


##########################################
#         helper functions for results presentation
##########################################

def add_hist(dt, **kwargs):
    '''
    add a histogram to an existing ax artist. gleb.shtengel@gmail.com 06/2023.
    
    Parameters:
    dt : data set
    
    kwargs:
    nbins : int
        number of bins for the histogram. Deafult is 64.
    hrange : (float, float)
        histogram range. Default is (0, 10).
    ax : artist
        axis artist to plot the histogram to. Default is 0 (no artist).
        In the case ax==0 no plotting is done and np.histogram is used instead
    col : str
        color of histogram plot. Default is 'red'
    label : str
        label of histogram plot. Default is empty string
        
    Returns:
    x, y, histmax_x, md, mn, std
    
    x : histogram bins
    y : histogram counts
    histmax_x : histogram peak
    md : median of dt
    mn : mean of dt
    std : standard deviation of dt
    '''
    nbins = kwargs.get('nbins', 64)
    hrange = kwargs.get('hrange', (0, 10))
    ax = kwargs.get('ax', 0)
    col = kwargs.get('col', 'red')
    label = kwargs.get('label', '')
    
    mn = np.mean(dt)
    md = np.median(dt)
    std = np.std(dt)
    if ax == 0:
        y, x = np.histogram(dt, bins=nbins, range=hrange)
    else:
        y, x, _ = ax.hist(dt, bins=nbins, range=hrange, color=col, histtype='step', stacked=True, fill=False, linewidth=3, label = label)
    histmax_x = x[np.argmax(y)]+(x[1]-x[0])/2
    histmax_y = np.max(y)
    return x, y, histmax_x, md, mn, std


def read_kwargs_xlsx(file_xlsx, kwargs_sheet_name, **kwargs):
    '''
    Reads (SIFT processing) kwargs from XLSX file and returns them as dictionary. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com
    
    Parameters:
    file_xlsx : str
        Full path to XLSX file containing a worksheet with SIFt parameters saves as two columns: (name, value)
    kwargs_sheet_name : str
        Name of the worksheet containing SIFT parameters
    '''
    disp_res_local = kwargs.get('disp_res', False)

    kwargs_dict_initial = {}
    try:
        stack_info = pd.read_excel(file_xlsx, header=None, sheet_name=kwargs_sheet_name).T                #read from transposed
        if len(stack_info.keys())>0:
            if len(stack_info.keys())>0:
                for key in stack_info.keys():
                    kwargs_dict_initial[stack_info[key][0]] = stack_info[key][1]
            else:
                kwargs_dict_initial['Stack Filename'] = stack_info.index[1]
    except:
        if disp_res_local:
            print('No stack info record present, using defaults')
    kwargs_dict = {}
    for key in kwargs_dict_initial:
        if 'TransformType' in key:
            exec('kwargs_dict["TransformType"] = ' + kwargs_dict_initial[key].split('.')[-1].split("'")[0])
        elif 'targ_vector' in key:
            try:
                exec('kwargs_dict["targ_vector"] = np.array(' + kwargs_dict_initial[key].replace(' ', ',')+ ')')
            except:
                kwargs_dict["targ_vector"] = np.array([1, 0, 0, 0, 1, 0]) 
        elif 'l2_matrix' in key:
            try:
                exec('kwargs_dict["l2_matrix"] = np.array(' + kwargs_dict_initial[key].replace(' ', ',') + ')')
            except:
                kwargs_dict["l2_matrix"] = np.eye(6)
        elif 'fit_params' in key:
            exec('kwargs_dict["fit_params"] = ' + kwargs_dict_initial[key])
        elif 'subtract_linear_fit' in key:
            exec('kwargs_dict["subtract_linear_fit"] = np.array(' + kwargs_dict_initial[key]+')')
        elif 'Stack Filename' in key:
            exec('kwargs_dict["Stack Filename"] = str(kwargs_dict_initial[key])')
        elif 'kernel' in key:
            exec('kwargs_dict["kernel"] = np.array(' + kwargs_dict_initial['kernel'].replace(', ', ',').replace(' ', ',') +')')
        else:
            try:
                exec('kwargs_dict["'+str(key)+'"] = '+ str(kwargs_dict_initial[key]))
            except:
                try:
                    exec('kwargs_dict["'+str(key)+'"] = "' + kwargs_dict_initial[key].replace('\\', '/').replace('\n', ',') + '"')
                except:
                    pass
    if 'dump_filename' in kwargs.keys():
        kwargs_dict['dump_filename'] = kwargs['dump_filename']
    #correct for pandas mixed read failures
    try:
        if kwargs_dict['mrc_mode']:
            kwargs_dict['mrc_mode']=1
    except:
        pass
    try:
        if kwargs_dict['int_order']:
            kwargs_dict['int_order']=1
    except:
        pass
    try:
        if kwargs_dict['flipY'] == 1:
            kwargs_dict['flipY'] = True
        else:
            kwargs_dict['flipY'] = False
    except:
        pass
    try:
        if kwargs_dict['BFMatcher'] == 1:
            kwargs_dict['BFMatcher'] = True
        else:
            kwargs_dict['BFMatcher'] = False
    except:
        pass
    try:
        if kwargs_dict['invert_data'] == 1:
            kwargs_dict['invert_data'] = True
        else:
            kwargs_dict['invert_data'] = False
    except:
        pass
    try:
        if kwargs_dict['sliding_evaluation_box'] == 1:
            kwargs_dict['sliding_evaluation_box'] = True
        else:
            kwargs_dict['sliding_evaluation_box'] = False
    except:
        pass
    
    return kwargs_dict





# Memory profiling functions

def elapsed_since(start):
    #return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    elapsed = time.time() - start
    if elapsed < 1:
        return str(round(elapsed*1000,2)) + "ms"
    if elapsed < 60:
        return str(round(elapsed, 2)) + "s"
    if elapsed < 3600:
        return str(round(elapsed/60, 2)) + "min"
    else:
        return str(round(elapsed / 3600, 2)) + "hrs"


def get_process_memory():
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    try:
        return mi.rss, mi.vms, mi.shared
    except:
        return mi.rss, mi.vms, mi.private


def format_bytes(bytes):
    if np.abs(bytes) < 1000:
        return str(bytes)+"B"
    elif np.abs(bytes) < 1e6:
        return str(round(bytes/1e3,2)) + "kB"
    elif np.abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"


def profile(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        rss_before, vms_before, shared_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: {:>20}  RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format("<" + func.__name__ + ">",
                    format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))
        return result
    if inspect.isfunction(func):
        return wrapper
    elif inspect.ismethod(func):
        return wrapper(*args,**kwargs)
