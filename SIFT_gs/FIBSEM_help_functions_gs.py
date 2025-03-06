import numpy as np
from scipy.signal import savgol_filter, convolve2d
from scipy.ndimage import gaussian_filter

import pandas as pd
import os
import socket
import platform
from pathlib import Path
import time
import glob
import re

import matplotlib
import matplotlib.image as mpimg
from matplotlib import pylab, mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from IPython.core.pylabtools import figsize, getfigs
import cv2

from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve as astro_convolve

import psutil
import inspect

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from SIFT_gs.FIBSEM_custom_transforms_gs import *
except:
    raise RuntimeError("Unable to load FIBSEM_custom_transforms_gs")


######################################################
#  DASK client help functions
######################################################

def check_DASK(DASK_client, **kwargs):
    '''
    Checks DASK client and returns use_DASK and DASK monitor port. ©G.Shtengel 10/2024 gleb.shtengel@gmail.com
    
    Parameters:
    
    
    kwargs:
    disp_res : bolean
        If True (default), intermediate messages and results will be displayed
    
    '''
    disp_res  = kwargs.get("disp_res", True )
    use_DASK = False
    status_update_address = ''
    try:
        client_services = DASK_client.scheduler_info()['services']
        if client_services:
            try:
                dport = client_services['dashboard']
            except:
                dport = client_services['bokeh']
            #if platform.system() == 'Linux':
            hostname = socket.gethostname()
            status_update_address = 'http://' + hostname + ':{:d}/status'.format(dport)
            #if platform.system() == 'Windows':
            #    status_update_address = 'http://localhost:{:d}/status'.format(dport)
            if disp_res:
                print('DASK client exists. Will perform distributed computations')
                print('Use ' + status_update_address +' to monitor DASK progress')
            use_DASK = True
        else:
            if disp_res:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   DASK client does not exist. Will perform local computations')
    except:
        if disp_res:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   DASK client does not exist. Will perform local computations')
            
    return use_DASK, status_update_address


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

def calculate_gradent_map(img, ** kwargs):
    '''
    Computes 2D Gradient of the image. ©G.Shtengel 10/2024 gleb.shtengel@gmail.com

    Parameters:
    img : 2d array

    kwargs:
    perform_smoothing : boolean
        If True, the images is smoothed first before gradient application
    kernel : 2D float array
        a kernel to perfrom 2D smoothing convolution.
    normalize : boolean
        if True, the gradient is normalized by the image. Default is False.
    disp_res : boolean
        (default is False) - to plot/ display the results
    thresholds_disp : list [thr_min_disp, thr_max_disp]
            (default [1e-3, 1e-3]) CDF threshold for determining the min and max data values of abs_grad for display.
    fontsize : int
        Font Size for figure subtitles. Default is 10
    
    Returns
    abs_grad : 2d array
        absolute value of gradient
    '''
    perform_smoothing = kwargs.get('perform_smoothing', True)
    st = 1.0/np.sqrt(2.0)
    def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
    def_kernel = def_kernel/def_kernel.sum()
    kernel = kwargs.get("kernel", def_kernel)
    normalize = kwargs.get('normalize', False)
    disp_res = kwargs.get("disp_res", True)
    thr_min, thr_max  = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    fontsize = kwargs.get('fontsize', 10)

    ysz, xsz = img.shape
    xind, yind = np.meshgrid(np.arange(xsz), np.arange(ysz), sparse=False)

    if perform_smoothing:
        grad = np.gradient(convolve2d(img, kernel, mode='same'))
    else:
        grad = np.gradient(img)
    grad_array = np.array(grad)
    
    if normalize:
        abs_grad = np.sqrt(grad[0]*grad[0]+grad[1]*grad[1])/img
    else:
        abs_grad = np.sqrt(grad[0]*grad[0]+grad[1]*grad[1])
        
    if disp_res:
        fx = 10.0
        fy = fx * ysz / xsz *2.0
        fig, axs = plt.subplots(2,1, figsize=(fx, fy))
        axs[0].imshow(img, cmap='Greys')
        axs[0].set_title('Image', fontsize = fontsize)
        vmin, vmax = get_min_max_thresholds(abs_grad, thr_min=thr_min, thr_max=thr_max, disp_res=False)
        axs[1].imshow(abs_grad, vmin=vmin, vmax=vmax)
        axs[1].set_title('Gradient Map', fontsize = fontsize)
        for ax in axs:
            ax.axis(False)
            ax.grid(True)
    return abs_grad


def convert_tr_matr_into_deformation_field(transformation_matrix, image_shape, **kwargs):
    '''
    Converts transformation_matrix into deformation field on a grid determined by the shape (height, width) parameter.

    Parameters
    ----------
    transformation_matrix : 2D array
        Transformation matrix in a form:
            [[Sxx Sxy Tx]
            [Syx  Syy Ty]
            [0    0   1]]
    image_shape : list of two ints
        shape of the image (height, width)

    kwargs
    ---------
    index_map : str
        'abs' (default) - absolute indices of the pixels.
        'diff' - displacements
    '''
    index_map = kwargs.get('index_map', 'abs')
    # create grid points
    image_height, image_width = image_shape
    grid_x, grid_y = np.meshgrid(np.arange(image_width), np.arange(image_height))
    grid_points = np.stack([grid_x.flatten(), grid_y.flatten()]).T

    # Apply transformation to each point
    #transformed_points = np.dot(transformation_matrix[:2, :2], grid_points) + transformation_matrix[:2, 2]
    transformed_points = grid_points @ transformation_matrix[0:2, 0:2].T + transformation_matrix[0:2, 2]

    # Calculate displacement vectors
    if index_map == 'abs':
        displacement_vectors = transformed_points
    else:
        displacement_vectors = transformed_points - grid_points

    # Reshape into deformation field
    deformation_field = displacement_vectors.reshape((image_height, image_width, 2))

    return deformation_field


def determine_residual_deformation_field(src_pts, dst_pts, transformation_matrix_src, transformation_matrix_dst, image_shape, **kwargs):
    '''
    Calculates residual deformation field from given source points, destination points, and transformation matrices for both

    Parameters
    ----------
    src_pts : list of 2D arrays
        x, y coordinates of source points
    dst_pts : list of 2D arrays
        x, y coordinates of destination points
    transformation_matrix_src : 2D array
        Transformation matrix for src points in a form:
            [[Sxx Sxy Tx]
            [Syx  Syy Ty]
            [0    0   1]]
    transformation_matrix_dst : 2D array
        Transformation matrix for dst points in a form:
            [[Sxx Sxy Tx]
            [Syx  Syy Ty]
            [0    0   1]]
        If dst points are fixed (not moving), use np.eye(3,3) instead
    
    image_shape : list of two ints
        shape of the image (height, width)
    deformation_type : str
        Options are:
            '1DY' - Default. Deformation is performed using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
            '1DX' - Deformation is performed using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
            '2D' - Deformation is performed using 2D deformation field.
    deformation_sigma : list of 1 or two floats.
        Gaussian width of smoothing (units of pixels). Default is 50.
    zero_mean : boolean
        If True (Default), mean value is subtracted at the end.
    verbose : boolean
    
    Returns : deformation_field
    '''
    
    deformation_type = kwargs.get('deformation_type', '1DY')
    deformation_sigma = kwargs.get('deformation_sigma', [50.0, 50.0])
    verbose = kwargs.get('verbose', False)
    zero_mean = kwargs.get('zero_mean', True)
    
    image_height, image_width = image_shape
    # Create a regular grid
    grid_x, grid_y = np.meshgrid(np.linspace(0, image_width), np.linspace(0, image_height))
    
    transformation_matrix_src_inv = np.linalg.inv(transformation_matrix_src)
    transformation_matrix_dst_inv = np.linalg.inv(transformation_matrix_dst)
    
    src_pts_transformed = src_pts @ transformation_matrix_src_inv[0:2, 0:2].T + transformation_matrix_src_inv[0:2, 2]
    dst_pts_transformed = dst_pts @ transformation_matrix_dst_inv[0:2, 0:2].T + transformation_matrix_dst_inv[0:2, 2]
    xshifts, yshifts = (dst_pts_transformed - src_pts_transformed).T
    x, y = dst_pts.T
    if verbose:
        print('determine_residual_deformation_field : will calculate residual deformation field in format: ', deformation_type, ',  df_sigma(s)=', deformation_sigma)

    if deformation_type == '2D':
        # placeholder for now
        deformation_field = np.zeros((image_height, image_width), dtype=float)
    elif deformation_type == '1DX':
        try:
            deformation_sigma = deformation_sigma[0]
        except:
            pass
        x_profile = np.zeros(image_width, dtype=float)
        cnts = np.zeros(image_width, dtype=int)
        xints = x.astype(int)
        for xint, yshift in zip(xints, yshifts):
            x_profile[xint] = x_profile[xint] + yshift
            cnts[xint] = cnts[xint] + 1
        if zero_mean:
            x_profile_weighted = x_profile/cnts
            x_profile = np.nan_to_num(x_profile_weighted - np.nanmean(x_profile_weighted))
        else:
            x_profile = np.nan_to_num(x_profile_weighted)
        x_profile_smoothed = astro_convolve(x_profile, Gaussian1DKernel(stddev=deformation_sigma))
        #deformation_field = np.repeat(x_profile_smoothed[:, np.newaxis], image_height, 1).T
        deformation_field = x_profile_smoothed
    else:
        try:
            deformation_sigma = deformation_sigma[0]
        except:
            pass
        y_profile = np.zeros(image_height, dtype=float)
        cnts = np.zeros(image_height, dtype=int)
        yints = y.astype(int)
        for yint, xshift in zip(yints, xshifts):
            y_profile[yint] = y_profile[yint] + xshift
            cnts[yint] = cnts[yint] + 1
        #y_profile = np.nan_to_num(y_profile/cnts)
        if zero_mean:
            y_profile_weighted = y_profile/cnts
            y_profile = np.nan_to_num(y_profile_weighted - np.nanmean(y_profile_weighted))
        else:
            y_profile = np.nan_to_num(y_profile_weighted)
        y_profile_smoothed = astro_convolve(y_profile, Gaussian1DKernel(stddev=deformation_sigma))
        #deformation_field = np.repeat(y_profile_smoothed[:, np.newaxis], image_width, 1)
        deformation_field = y_profile_smoothed
    if verbose:
        print('determine_residual_deformation_field : Output  deformation_field shape: ', deformation_field.shape)
        print('determine_residual_deformation_field : finished calculation. Average residual deformation = {:.2f} pixels'.format(np.mean(deformation_field)))
    #if zero_mean:
    #    if verbose:
    #        print('Zero_mean = ', zero_mean, ' the mean value will be subtracted')
    #    deformation_field = deformation_field - np.mean(deformation_field)
    return deformation_field

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


def find_FWHM(x, y, **kwargs):
    '''
    find FWHM of the signal. G.Shtengel 09.2024
    
    Parameters:
    x : array or list of x values
    y : array or list of y values
        
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
    
    mx_ind = np.argmax(np.array(y))
    ii = np.max((mx_ind-max_aver_aperture//2, 0))
    ia = np.min((ii+max_aver_aperture, len(y)))
    ii = ia-max_aver_aperture
    mx = np.mean(y[ii:ia])
    y_mod = y - mx/2.0
    dx = x[1]-x[0]
    
    if estimation == 'count':
        FWHM = np.sum(i > 0.0 for i in y_mod) * dx
        indi = np.argmax(y_mod > 0.0) + 1
        inda = len(y) - np.argmax(np.flip(y_mod) > 0.0)
    else:
        if start == 'edges':
            indi = np.argmax(y_mod > 0.0) + 1
            inda = len(y) - np.argmax(np.flip(y_mod) > 0.0) 
        else:
            ln1 = len(y_mod[0:mx_ind])
            indi = ln1 - np.argmax(np.flip(y_mod[0:mx_ind]) < 0.0) +1
            inda = ln1 + np.argmax(y_mod[mx_ind:] < 0.0)
        FWHM = (inda-indi) * dx
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


def determine_pad_offsets(shape, tr_matr):
    ysz, xsz = shape
    corners = np.array([[0.0, 0.0, 1.0], [0.0, ysz, 1.0], [xsz, 0.0, 1.0], [xsz, ysz, 1.0]])
    a = np.array(tr_matr)[:, 0:2, :] @ corners.T
    #a = np.linalg.inv(np.array(tr_matr))[:, 0:2, :] @ corners.T
    xc = a[:, 0, :].ravel()
    yc = a[:, 1, :].ravel()
    xmin = np.round(np.min((np.min(xc), 0.0))).astype(int)
    xmax = np.round(np.max(xc)-xsz).astype(int)
    ymin = np.round(np.min((np.min(yc), 0.0))).astype(int)
    ymax = np.round(np.max(yc)-ysz).astype(int)
    #print(xmin, xmax, ymin, ymax)
    #xi = int(-1 * xmin)
    #yi = int(-1 * ymin)
    xi = int(np.max((xmax, 0)))
    yi = int(np.max((ymax, 0)))
    padx = np.max((int(xmax - xmin), xi))
    pady = np.max((int(ymax - ymin), yi))
    #return xmin, xmax, ymin, ymax
    #print(xi, yi, padx, pady)
    return xi, yi, padx, pady


def set_eval_bounds(shape, evaluation_box, **kwargs):
    '''
    Set up evaluation bounds.

    Parameters:
    shape : list of 2 ints
        [Ysize, Xsize] frame size in pixels
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.


    kwargs:
    perform_transformation : boolean
        If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed.
    pad_edges : boolean
        If True, the frame will be padded to account for frame position and/or size changes.
    tr_matr : list of transformation matrices
    sliding_evaluation_box : boolean
        if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above
    frame_inds : int array
        array of frame indices

    Returnes:
    evaluation_bounds : int array of 4 elements
        evaluation_bounds = (xi_evals, xa_evals, yi_evals, ya_evals)

    '''
    ny, nx = shape

    pad_edges = kwargs.get('pad_edges', False)
    perform_transformation = kwargs.get('perform_transformation', False)
    tr_matr = np.array(kwargs.get('tr_matr', []))
    ntr = len(tr_matr)
    if ntr>0:
        frame_inds = np.array(kwargs.get("frame_inds", np.arange(ntr)))
        nz = len(frame_inds)
    else:
        frame_inds = np.array([])
        nz = 0
    #print('will use frame_inds: ', frame_inds)
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", False)
    stop_evaluation_box = kwargs.get("stop_evaluation_box", False)
    
    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
        top_left_corners = np.vstack((np.array([start_evaluation_box[2] + dx_eval*frame_inds//(ntr-1)]),
                                      np.array([start_evaluation_box[0] + dy_eval*frame_inds//(ntr-1)]))).T
    else:
        top_left_corners = np.array([[evaluation_box[2],evaluation_box[0]]]*nz)
    
    if pad_edges and perform_transformation and nz>0:
        xi, yi, padx, pady = determine_pad_offsets([ny, nx], tr_matr)

        shift_matrix = np.array([[1.0, 0.0, xi],
                         [0.0, 1.0, yi],
                         [0.0, 0.0, 1.0]])
        inv_shift_matrix = np.linalg.inv(shift_matrix)
        top_left_corners_ext = np.vstack(((top_left_corners+ np.array((xi, yi))).T, np.ones(nz)))
        #arr1 = (shift_matrix @ (tr_matr @ inv_shift_matrix))[frame_inds, 0:2, :]
        arr1 = np.linalg.inv(shift_matrix @ (tr_matr @ inv_shift_matrix))[frame_inds, 0:2, :]
        arr2 = top_left_corners_ext.T
        eval_starts = np.round(np.einsum('ijk,ik->ij',arr1, arr2))
    else:
        padx = 0
        pady = 0
        eval_starts = top_left_corners

    if sliding_evaluation_box:
        xa_evals = np.clip(eval_starts[:, 0] + start_evaluation_box[3], 0, (nx+padx))
        ya_evals = np.clip(eval_starts[:, 1] + start_evaluation_box[1], 0, (ny+pady))
        xi_evals = xa_evals - start_evaluation_box[3]
        yi_evals = ya_evals - start_evaluation_box[1]
    else:
        sv_apert = np.min(((nz//8)*2+1, 2001))
        if sv_apert <2:
            # no smoothing if the set is too short
            xa_evals = np.clip(eval_starts[:, 0] + evaluation_box[3], 0, (nx+padx))
            ya_evals = np.clip(eval_starts[:, 1] + evaluation_box[1], 0, (ny+pady))
        else:
            filter_order = np.min([5, sv_apert-1])
            xa_evals = savgol_filter(np.clip(eval_starts[:, 0] + evaluation_box[3], 0, (nx+padx)), sv_apert, filter_order)
            ya_evals = savgol_filter(np.clip(eval_starts[:, 1] + evaluation_box[1], 0, (ny+pady)), sv_apert, filter_order)
        xi_evals = xa_evals - evaluation_box[3]
        yi_evals = ya_evals - evaluation_box[1]

    return np.array(np.vstack((xi_evals, xa_evals, yi_evals, ya_evals))).T.astype(int)


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


def check_registration(img0, img1, **kwargs):
    '''
    Debugging tool. Perform SIFT registration check on two images. Will perform SIFT, and then report Residual Errors and plot residual error histograms
    
    Parameters:
    img0 : 2D array
    img1 : 2D array
    
    kwargs:
    threshold_min : float
        CDF threshold for determining the minimum data value
    threshold_max : float
        CDF threshold for determining the maximum data value
    TransformType : object reference
        Transformation model used by SIFT for determining the transformation matrix from Key-Point pairs.
        Choose from the following options:
            ShiftTransform - only x-shift and y-shift
            XScaleShiftTransform  -  x-scale, x-shift, y-shift
            ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
            AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
            RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
    l2_matrix : 2D float array
        matrix of regularization (shrinkage) parameters
    targ_vector = 1D float array
        target vector for regularization
    solver : str
        Solver used for SIFT ('RANSAC' or 'LinReg')
    RANSAC_initial_fraction : float
        Fraction of data points for initial RANSAC iteration step. Default is 0.005.
    drmax : float
        In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
        In the case of 'LinReg' - outlier threshold for iterative regression
    max_iter : int
        Max number of iterations in the iterative procedure above (RANSAC or LinReg)
    BFMatcher : boolean
        If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches : boolean
        If True, matches will be saved into individual files
    kp_max_num : int
        Max number of key-points to be matched.
        Key-points in every frame are indexed (in descending order) by the strength of the response.
        Only kp_max_num is kept for further processing.
        Set this value to -1 if you want to keep ALL keypoints (may take forever to process!)
    SIFT_nfeatures : int
        SIFT libary default is 0. The number of best features to retain.
        The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
    SIFT_nOctaveLayers : int
        SIFT libary default  is 3. The number of layers in each octave.
        3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
    SIFT_contrastThreshold : double
        SIFT libary default  is 0.04. The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
        The larger the threshold, the less features are produced by the detector.
        The contrast threshold will be divided by nOctaveLayers when the filtering is applied.
        When nOctaveLayers is set to default and if you want to use the value used in
        D. Lowe paper (0.03), set this argument to 0.09.
    SIFT_edgeThreshold : double
        SIFT libary default  is 10. The threshold used to filter out edge-like features.
        Note that the its meaning is different from the contrastThreshold,
        i.e. the larger the edgeThreshold, the less features are filtered out
        (more features are retained).
    SIFT_sigma : double
        SIFT library default is 1.6.  The sigma of the Gaussian applied to the input image at the octave #0.
        If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    save_filename : string
        Filename for saving the images
    dpi : int
        DPI for PNG output. Default is 300
    verbose : boolean
        If True, outputs will be printed
    fontsize : int
        Fontsize
    
    '''
    threshold_min = kwargs.get("threshold_min", 1e-3)
    threshold_max = kwargs.get("threshold_max", 1e-3)
    nbins = kwargs.get('nbins', 64)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", 0.005)  # fraction of data points for initial RANSAC iteration step.
    drmax = kwargs.get("drmax", 1.5)
    max_iter = kwargs.get("max_iter", 2500)
    #kp_max_num = kwargs.get("kp_max_num", -1)
    Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
    save_res_png  = kwargs.get("save_res_png", True)
    save_filename = kwargs.get('save_filename', 'check_registration_output.png')
    dpi = kwargs.get('dpi', 300)
    SIFT_nfeatures = kwargs.get("SIFT_nfeatures", 0)
    SIFT_nOctaveLayers = kwargs.get('SIFT_nOctaveLayers', 3)
    SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", 10.0)
    SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", 0.025)
    SIFT_sigma = kwargs.get('SIFT_sigma', 1.6)
    RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", 0.010)  # fraction of data points for initial RANSAC iteration step.
    Sample_ID = kwargs.get('Sample_ID', '')
    filename = kwargs.get('filename', '') 
    verbose = kwargs.get('verbose', True)
    fontsize = kwargs.get('fontsize', 12) 
    fsize_text = kwargs.get('fsize_text', 6)
    
    YResolution, XResolution = img0.shape
    
    SIFT_kwargs = {
        'threshold_min' : threshold_min, 
        'threshold_max' : threshold_max,
        'nbins' : nbins,
        'TransformType' : TransformType,
        'SIFT_nfeatures' : SIFT_nfeatures,
        'SIFT_nOctaveLayers' : SIFT_nOctaveLayers,
        'SIFT_contrastThreshold' : SIFT_contrastThreshold,
        'SIFT_edgeThreshold' : SIFT_edgeThreshold,
        'SIFT_sigma' : SIFT_sigma,
        'Lowe_Ratio_Threshold' : Lowe_Ratio_Threshold,
        'l2_matrix' : l2_matrix,
        'targ_vector': targ_vector, 
        'solver' : 'RANSAC',
        'RANSAC_initial_fraction' : RANSAC_initial_fraction,
        'drmax' : drmax,
        'max_iter' : max_iter}
    
    
    d0, d1 = get_min_max_thresholds(img0, thr_min=threshold_min, thr_max=threshold_max, disp_res=False)
    img0_uint8 = np.clip(255*(img0-d0)/(d1-d0), 0, 255).astype(np.uint8)
    d0, d1 = get_min_max_thresholds(img1, thr_min=threshold_min, thr_max=threshold_max, disp_res=False)
    img1_uint8 = np.clip(255*(img1-d0)/(d1-d0), 0, 255).astype(np.uint8)
    if verbose:
        print('thr_min={:.0e}, thr_max={:.0e}'.format(threshold_min, threshold_max))
        print(TransformType.__name__)
        print('SIFT_nfeatures={:d}'.format(SIFT_nfeatures))
        print('SIFT_nOctaveLayers={:d},  SIFT_edgeThreshold={:.3f}'.format(SIFT_nOctaveLayers, SIFT_edgeThreshold))
        print('SIFT_contrastThreshold={:.3f},  SIFT_sigma={:.3f}'.format(SIFT_contrastThreshold, SIFT_sigma))
        print('RANSAC_initial_fraction={:.3f}'.format(RANSAC_initial_fraction))
        print('drmax={:.3f}'.format(drmax))
        
        print('')
    if verbose:
        print('Extracting KeyPoints (SIFT) on the first image')
    sift1 = cv2.SIFT_create(nfeatures=SIFT_nfeatures, nOctaveLayers=SIFT_nOctaveLayers, edgeThreshold=SIFT_edgeThreshold, contrastThreshold=SIFT_contrastThreshold, sigma=SIFT_sigma)
    kp1, des1 = sift1.detectAndCompute(img0_uint8, None)
    n_kpts=len(kp1)
    if verbose:
        print('Extracted {:d} keypoints'.format(n_kpts))
    print('Extracting KeyPoints (SIFT) on the second image')
    sift2 = cv2.SIFT_create(nfeatures=SIFT_nfeatures, nOctaveLayers=SIFT_nOctaveLayers, edgeThreshold=SIFT_edgeThreshold, contrastThreshold=SIFT_contrastThreshold, sigma=SIFT_sigma)
    kp2, des2 = sift2.detectAndCompute(img1_uint8, None)
    if verbose:
        print('Extracted {:d} keypoints'.format(len(kp2)))

    if verbose:
        print('Finding Matches')
    if TransformType == RegularizedAffineTransform:
        def estimate(self, src, dst):
            self.params = determine_regularized_affine_transform(src, dst, l2_matrix, targ_vector)
        RegularizedAffineTransform.estimate = estimate
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's Ratio test
    good = []
    for m, n in matches:
        if m.distance < Lowe_Ratio_Threshold * n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

    if verbose:
        print('Finding Transformation Matrix')
    min_samples = np.int32(len(src_pts)*RANSAC_initial_fraction)
    model, inliers = ransac((src_pts, dst_pts),
        TransformType, min_samples = min_samples,
        residual_threshold = drmax, max_trials = max_iter)
    n_inliers = np.sum(inliers)
    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
    src_pts_ransac = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
    dst_pts_ransac = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
    kpts = [src_pts_ransac, dst_pts_ransac]
    transform_matrix = model.params
    if verbose:
        print('Transform_Matrix:')
        print(transform_matrix)

    iteration = len(src_pts)- len(src_pts_ransac)
    reg_errors, xshifts, yshifts = estimate_kpts_transform_error(src_pts_ransac, dst_pts_ransac, transform_matrix)
    error_abs_mean = np.mean(np.abs(reg_errors))
    
    fig_xsize = 10.0
    fig_ysize = fig_xsize * YResolution / XResolution + 5.0
    fig = plt.figure(figsize=(fig_xsize, fig_ysize))
    fig.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.15, hspace=-0.10)
    gs = GridSpec(3, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0:2, :])
    ax0.imshow(img1, cmap='Greys', vmin = d0, vmax=d1)
    ax0.axis(False)

    n_matches = len(xshifts)
    if n_matches > 0:
        x, y = dst_pts_ransac.T
        M = np.sqrt(xshifts*xshifts+yshifts*yshifts)
        xs = xshifts
        ys = yshifts
        # the code below is for vector map. vectors have origin coordinates x and y, and vector projections xs and ys.
        vec_field = ax0.quiver(x,y,xs,ys,M, scale=50, width =0.0015, cmap='jet')
        cbar = fig.colorbar(vec_field, pad=0.015, shrink=0.70, orientation = 'horizontal', format="%.1f")
        cbar.set_label('Magnitude of the Residual Registration Error (pix)', fontsize=fontsize)

        ax0.text(0.005, 1.00 - 0.010*XResolution/YResolution, filename, fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.023*XResolution/YResolution, Sample_ID, fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.036*XResolution/YResolution, 'thr_min={:.0e}, thr_max={:.0e}'.format(threshold_min, threshold_max), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.049*XResolution/YResolution, TransformType.__name__, fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.062*XResolution/YResolution, 'SIFT_nfeatures={:d}'.format(SIFT_nfeatures), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.075*XResolution/YResolution, 'SIFT_nOctaveLayers={:d},  SIFT_edgeThreshold={:.3f}'.format(SIFT_nOctaveLayers, SIFT_edgeThreshold), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.088*XResolution/YResolution, 'SIFT_contrastThreshold={:.3f},  SIFT_sigma={:.3f}'.format(SIFT_contrastThreshold, SIFT_sigma), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.101*XResolution/YResolution, 'RANSAC_initial_fraction={:.4f}, max_iter={:d}'.format(RANSAC_initial_fraction, max_iter), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.114*XResolution/YResolution,  'drmax={:.3f}'.format(drmax), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.127*XResolution/YResolution, '# of keypoints = {:d}, # of matches ={:d}'.format(n_kpts, n_matches), fontsize=fsize_text, transform=ax0.transAxes)
        
        ax0.text(0.75, 1.00 - 0.023*XResolution/YResolution, 'Sxx={:.5f},  Sxy={:.5f},  Tx={:.3e}'.format(transform_matrix[0,0], transform_matrix[0,1], transform_matrix[0,2]), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.75, 1.00 - 0.036*XResolution/YResolution, 'Syx={:.5f},  Syy={:.5f},  Ty={:.3e}'.format(transform_matrix[1,0], transform_matrix[1,1], transform_matrix[1,2]), fontsize=fsize_text, transform=ax0.transAxes)
    if verbose:
        print('RANSAC_initial_fraction = {:.4f}, max_iter={:d}'.format(RANSAC_initial_fraction, max_iter))
        print('# of keypoints = {:d}, # of matches ={:d}'.format(n_kpts, n_matches))

    axx = fig.add_subplot(gs[2, 0])
    xcounts, xbins, xhist_patches = axx.hist(xshifts, bins=64)
    error_FWHMx, indxi, indxa, mxx, mxx_ind = find_FWHM(xbins, xcounts[:-1], verbose=False, max_aver_aperture=5)
    dbx = (xbins[1]-xbins[0])/2.0
    axx.set_xlabel('Residual X Error (pixels)')
    axx.set_ylabel('Count')
    axy = fig.add_subplot(gs[2, 1])
    ycounts, ybins, yhist_patches = axy.hist(yshifts, bins=64)
    error_FWHMy, indyi, indya, mxy, mxy_ind = find_FWHM(ybins, ycounts[:-1], verbose=False, max_aver_aperture=5)
    dby = (ybins[1]-ybins[0])/2.0
    axy.set_xlabel('Residual Y Error (pixels)')
    axy.set_ylabel('Count')

    xcounts, xbins, xhist_patches = axx.hist(xshifts, bins=64, color='#1f77b4')
    error_FWHMx, indxi, indxa, mxx, mxx_ind = find_FWHM(xbins, xcounts[:-1], verbose=False, max_aver_aperture=5)
    dbx = (xbins[1]-xbins[0])/2.0
    #axx.plot([xbins[indxi]+dbx, xbins[indxa]+dbx], [mxx/2.0, mxx/2.0], 'r', linewidth = 4)
    axx.plot([xbins[indxi], xbins[indxa]], [mxx/2.0, mxx/2.0], 'r', linewidth = 4)
    axx.plot([xbins[mxx_ind]+dbx], [mxx], 'rd')
    axx.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(xshifts)), transform=axx.transAxes, fontsize=fontsize)
    axx.text(0.05, 0.8, 'median={:.3f}'.format(np.median(xshifts)), transform=axx.transAxes, fontsize=fontsize)
    axx.text(0.05, 0.7, 'FWHM={:.3f}'.format(error_FWHMx), transform=axx.transAxes, fontsize=fontsize)
    ycounts, ybins, yhist_patches = axy.hist(yshifts, bins=64, color='#1f77b4')
    error_FWHMy, indyi, indya, mxy, mxy_ind = find_FWHM(ybins, ycounts[:-1], verbose=False, max_aver_aperture=5)
    dby = (ybins[1]-ybins[0
        ])/2.0
    #axy.plot([ybins[indyi] + dby, ybins[indya] + dby], [mxy/2.0, mxy/2.0], 'r', linewidth = 4)
    axy.plot([ybins[indyi], ybins[indya]], [mxy/2.0, mxy/2.0], 'r', linewidth = 4)
    axy.plot([ybins[mxy_ind] + dby], [mxy], 'rd')
    axy.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(yshifts)), transform=axy.transAxes, fontsize=fontsize)
    axy.text(0.05, 0.8, 'median={:.3f}'.format(np.median(yshifts)), transform=axy.transAxes, fontsize=fontsize)
    axy.text(0.05, 0.7, 'FWHM={:.3f}'.format(error_FWHMy), transform=axy.transAxes, fontsize=fontsize)

    
    for ax in [axx, axy]:
        ax.grid(True)
        
    if save_res_png:
        axx.text(-0.07, -0.13, save_filename, transform=axx.transAxes, fontsize=fontsize-2)
        if verbose:
            print('Figure is saved into the filr: ', save_filename) 
        fig.savefig(save_filename, dpi=dpi)
    return error_FWHMx, error_FWHMy