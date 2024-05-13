import numpy as np
from scipy.signal import savgol_filter

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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from SIFT_gs.FIBSEM_custom_transforms_gs import *
except:
    #from FIBSEM_custom_transforms_gs import *
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
    data_max = bins[np.argmin(abs(cdf-(1.0-thr_max)))]
    data_min = bins[np.argmin(abs(cdf-thr_min))]
    
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
            if abs(ai-aa)>1e-10:
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
            if abs(ai-aa)>1e-10:
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
            if abs(ai-aa)>1e-10:
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
            if abs(ai-aa)>1e-10:
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
    symm : int
        Symmetry factor (how many times Start and stop angle intervals are repeated within 360 deg). Default is 4.

    Returns
        rescaler_kernel : float array
    '''
    xstart = kwargs.get('xstart', 0.1)
    xstop = kwargs.get('xstop', 1.0)
    dy = kwargs.get('dy', 0.01)
    symm = kwargs.get('symm', 4)
    ds = data.shape
    yc, xc = ds[0]//2, ds[1]//2
    
    y, x = np.indices(ds)
    xa = np.abs(x - xc) / np.max(x - xc)
    ya = np.abs(y - yc) / np.max(y - yc)
    
    rescaler_kernel = np.ones(ds, dtype=float)
    cond = (xa > xstart)*(xa < xstop)*(ya<=dy/2.0)
    rescaler_kernel[cond] = 0.0
    
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
                exec('kwargs_dict["'+str(key)+'"] = "' + kwargs_dict_initial[key].replace('\\', '/').replace('\n', ',') + '"')
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
