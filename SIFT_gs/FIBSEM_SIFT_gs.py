import numpy as np
import cupy as cp
import pandas as pd
import os
import time
import glob

import matplotlib
from matplotlib import pylab, mlab, pyplot
plt = pyplot
from IPython.core.pylabtools import figsize, getfigs
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as PILImage
from PIL.TiffTags import TAGS

from struct import *
#from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm

import skimage
#print(skimage.__version__)
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform, EuclideanTransform, warp
try:
    import skimage.external.tifffile as tiff
except:
    import tifffile as tiff
from scipy.signal import savgol_filter
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

from sklearn.linear_model import (LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


import dask
import dask.array as da
from dask.distributed import Client, progress, get_task_stream
from dask.diagnostics import ProgressBar

import cv2
print('Open CV version: ', cv2. __version__)
import mrcfile
import pickle
import webbrowser
from IPython.display import IFrame

EPS = np.finfo(float).eps

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



######################################################
#    General Help Functions
######################################################
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
        sm_data = savgol_filter(data.astype(double), window, porder)
        data_spread = np.std(data-sm_data)
    except :
        print('spread error')
        data_spread = 0.0
    return data_spread


def get_min_max_thresholds(image, thr_min=1e-3, thr_max=1e-3, nbins=256, disp_res=False):
    '''
    Determines the data range (min and max) with given fractional thresholds for cumulative distribution.
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com

    Calculates the histogram of pixel intensities of the image with number of bins determined by parameter nbins (default = 256)
    and normalizes it to get the probability distribution function (PDF), from which a cumulative distribution function (CDF) is calculated.
    Then given the thr_min, thr_max parameters, the minimum and maximum values of the data range are found
    by determining the intensities at which CDF= thr_min and (1- thr_max), respectively

    Parameters:
    ----------
    image : 2D array
        Image to be analyzed
    thr_min : float
        The low CDF bound for the data range
    thr_max : float
        The high CDF bound for the data range
    nbins : int
        Number of bins for PDF histogram
    disp_res ; boolean
        If True, a plot showing the data analysis is displayed

    Returns
    (dmin, dmax) : float array
    '''
    if disp_res:
        fsz=11
        fig, axs = subplots(2,1, figsize = (6,8))
        hist, bins, patches = axs[0].hist(image.ravel(), bins=nbins)
    else:
        hist, bins = np.histogram(image.ravel(), bins=nbins)
    pdf = hist / np.prod(image.shape)
    cdf = np.cumsum(pdf)
    data_max = bins[argmin(abs(cdf-(1.0-thr_max)))]
    data_min = bins[argmin(abs(cdf-thr_min))]
    
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
    return np.array((data_min, data_max))


def argmax2d(X):
    return np.unravel_index(X.argmax(), X.shape)


def find_BW(fr, FSC, SNRt):
    npts = np.shape(FSC)[0]*0.75
    j = 15
    while (j<npts-1) and FSC[j]>SNRt:
        j = j+1
    BW = fr[j-1] + (fr[j]-fr[j-1])*(SNRt-FSC[j-1])/(FSC[j]-FSC[j-1])
    #print(fr[j-1], fr[j], FSC[j-1], FSC[j])
    #print(BW, SNRt)
    return BW


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


def radial_profile_select_angles(data, center, astart = 89, astop = 91, symm=4):
    '''
    Calculates radially average profile of the 2D array (used for FRC) within a select range of angles.
    ©G.Shtengel 08/2020 gleb.shtengel@gmail.com

    Parameters:
    data : 2D array
    center : list of two ints
        [xcenter, ycenter]

    astart : float
        Start angle for radial averaging. Default is 0
    astop : float
        Stop angle for radial averaging. Default is 90
    symm : int
        Symmetry factor (how many times Start and stop angle intervalks are repeated within 360 deg). Default is 4.


    Returns
        radialprofile : float array
            limited to x-size//2 of the input data
    '''
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_ang = (np.angle(x - center[0]+1j*(y - center[1]), deg=True)).ravel()
    r = r.astype(np.int)
    
    if symm>0:
        ind = np.squeeze(np.array(np.where((r_ang > astart) & (r_ang < astart))))
        for i in np.arange(symm):
            ai = astart +360.0/symm*i -180.0
            aa = astop +360.0/symm*i -180.0
            ind = np.concatenate((ind, np.squeeze((np.array(np.where((r_ang > ai) & (r_ang < aa)))))), axis=0)
    else:
        ind = np.squeeze(np.where((r_ang > astart) & (r_ang < astop)))
        print(size(ind))
        
    rr = np.ravel(r)[ind]
    dd = np.ravel(data)[ind]

    tbin = np.bincount(rr, dd)
    nr = np.bincount(rr)
    radialprofile = tbin / nr
    return radialprofile


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
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2-1):-(window_len//2)]


################################################
#      Single Frame Image Processing Functions
################################################
def _center_and_normalize_points_gs(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.
    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.
    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(D).
    If the points are all identical, the returned values will contain nan.

    Parameters:
    ----------
    points : (N, D) array
        The coordinates of the image points.
    Returns:
    -------
    matrix : (D+1, D+1) array
        The transformation matrix to obtain the new points.
    new_points : (N, D) array
        The transformed image points.
    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.
    """
    n, d = points.shape
    centroid = np.mean(points, axis=0)

    centered = points - centroid
    rms = np.sqrt(np.sum(centered ** 2) / n)

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if rms == 0:
        return np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan)

    norm_factor = np.sqrt(d) / rms

    part_matrix = norm_factor * np.concatenate(
            (np.eye(d), -centroid[:, np.newaxis]), axis=1
            )
    matrix = np.concatenate(
            (part_matrix, [[0,] * d + [1]]), axis=0
            )

    points_h = np.row_stack([points.T, np.ones(n)])

    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points


def Single_Image_SNR(img, **kwargs):
    '''
    Estimates SNR based on a single image.
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    Calculates SNR of a single image base on auto-correlation analysis after [1].
    
    Parameters
    ---------
    img : 2D array
     
    kwargs:
    edge_fraction : float
        fraction of the full autocetrrelation range used to calculate the "mean value" (default is 0.10)
    disp_res : boolean
        display results (plots) (default is True)
    save_res_png : boolean
        save the analysis output into a PNG file (default is True)
    res_fname : string
        filename for the result image ('SNR_result.png')
    img_label : string
        optional image label
    dpi : int
        dots-per-inch resolution for the output image
        
    Returns:
        xSNR, ySNR, rSNR : float, float, float
            SNR determined using the method in [1] along X- and Y- directions.
            If there is a direction with slow varying data - that direction provides more accurate SNR estimate
            Y-streaks in typical FIB-SEM data provide slow varying Y-component because streaks
            usually get increasingly worse with increasing Y. 
            So for typical FIB-SEM data use ySNR
    
    [1] J. T. L. Thong et al, Single-image signal-to-noise ratio estimation. Scanning, 328–336 (2001).
    '''
    edge_fraction = kwargs.get("edge_fraction", 0.10)
    disp_res = kwargs.get("disp_res", True)
    nbins_disp = kwargs.get("nbins_disp", 256)
    thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    save_res_png = kwargs.get("save_res_png", True)
    res_fname = kwargs.get("res_fname", 'SNR_results.png')
    img_label = kwargs.get("img_label", 'Orig. Image')
    dpi = kwargs.get("dpi", 300)
    
    #first make image size even
    ysz, xsz = img.shape
    img = img[0:ysz//2*2, 0:xsz//2*2]
    ysz, xsz = img.shape

    xy_ratio = xsz/ysz
    data_FT = fftshift(fftn(ifftshift(img-img.mean())))
    data_FC = (np.multiply(data_FT,np.conj(data_FT)))/xsz/ysz
    data_ACR = np.abs(fftshift(fftn(ifftshift(data_FC))))
    data_ACR_peak = data_ACR[ysz//2, xsz//2]
    data_ACR_log = np.log(data_ACR)
    data_ACR = data_ACR / data_ACR_peak
    radial_ACR = radial_profile(data_ACR, [xsz//2, ysz//2])
    #print(radial_ACR[0:10], radial_ACR[-10:-1])
    r_ACR = np.concatenate((radial_ACR[::-1], radial_ACR[1:-1]))
    
    #rsz = xsz
    rsz = len(r_ACR)
    rcr = np.linspace(-rsz//2, rsz//2-1, rsz)
    xcr = np.linspace(-xsz//2, xsz//2-1, xsz)
    ycr = np.linspace(-ysz//2, ysz//2-1, ysz)
    
    xl = xcr[xsz//2-2:xsz//2]
    xacr_left = data_ACR[ysz//2, (xsz//2-2):(xsz//2)]
    xc = xcr[xsz//2]
    xr = xcr[xsz//2+1 : xsz//2+3]
    xacr_right = data_ACR[ysz//2, (xsz//2+1):(xsz//2+3)]
    xNFacl = xacr_left[0] + (xc-xl[0])/(xl[1]-xl[0])*(xacr_left[1]-xacr_left[0])
    xNFacr = xacr_right[0] + (xc-xr[0])/(xr[1]-xr[0])*(xacr_right[1]-xacr_right[0])
    x_left = xcr[xsz//2-2:xsz//2+1]
    xacr_left = np.concatenate((xacr_left, np.array([xNFacl])))
    x_right = xcr[xsz//2 : xsz//2+3]
    xacr_right = np.concatenate((np.array([xNFacr]), xacr_right))
    
    yl = ycr[ysz//2-2:ysz//2]
    yacr_left = data_ACR[(ysz//2-2):(ysz//2), xsz//2]
    yc = ycr[ysz//2]
    yr = ycr[ysz//2+1 : ysz//2+3]
    yacr_right = data_ACR[(ysz//2+1):(ysz//2+3), xsz//2]
    yNFacl = yacr_left[0] + (yc-yl[0])/(yl[1]-yl[0])*(yacr_left[1]-yacr_left[0])
    yNFacr = yacr_right[0] + (yc-yr[0])/(yr[1]-yr[0])*(yacr_right[1]-yacr_right[0])
    y_left = ycr[ysz//2-2:ysz//2+1]
    yacr_left = np.concatenate((yacr_left, np.array([yNFacl])))
    y_right = ycr[ysz//2 : ysz//2+3]
    yacr_right = np.concatenate((np.array([yNFacr]), yacr_right))
    
    rl = rcr[rsz//2-2:rsz//2]
    racr_left = r_ACR[(rsz//2-2):(rsz//2)]
    rc = rcr[rsz//2]
    rr = rcr[rsz//2+1 : rsz//2+3]
    racr_right = r_ACR[(rsz//2+1):(rsz//2+3)]
    rNFacl = racr_left[0] + (rc-rl[0])/(rl[1]-rl[0])*(racr_left[1]-racr_left[0])
    rNFacr = racr_right[0] + (rc-rr[0])/(rr[1]-rr[0])*(racr_right[1]-racr_right[0])
    r_left = rcr[rsz//2-2:rsz//2+1]
    racr_left = np.concatenate((racr_left, np.array([rNFacl])))
    r_right = rcr[rsz//2 : rsz//2+3]
    racr_right = np.concatenate((np.array([rNFacr]), racr_right))
    
    x_acr = data_ACR[ysz//2, xsz//2]
    x_noise_free_acr = xacr_right[0]
    xedge = int32(xsz*edge_fraction)
    x_mean_value = np.mean(data_ACR[ysz//2, 0:xedge])
    xx_mean_value = np.linspace(-xsz//2, (-xsz//2+xedge-1), xedge)
    yedge = int32(ysz*edge_fraction)
    y_acr = data_ACR[ysz//2, xsz//2]
    y_noise_free_acr = yacr_right[0]
    y_mean_value = np.mean(data_ACR[0:yedge, xsz//2])
    yy_mean_value = np.linspace(-ysz//2, (-ysz//2+yedge-1), yedge)
    redge = int32(rsz*edge_fraction)
    r_acr = data_ACR[ysz//2, xsz//2]
    r_noise_free_acr = racr_right[0]
    r_mean_value = np.mean(r_ACR[0:redge])
    rr_mean_value = np.linspace(-rsz//2, (-rsz//2+redge-1), redge)
    
    xSNR = (x_noise_free_acr-x_mean_value)/(x_acr - x_noise_free_acr)
    ySNR = (y_noise_free_acr-y_mean_value)/(y_acr - y_noise_free_acr)
    rSNR = (r_noise_free_acr-r_mean_value)/(r_acr - r_noise_free_acr)
    if disp_res:
        fs=12
        
        if xy_ratio < 2.5:
            fig, axs = subplots(1,4, figsize = (20, 5))
        else:
            fig = plt.figure(figsize = (20, 5))
            ax0 = fig.add_subplot(2, 2, 1)
            ax1 = fig.add_subplot(2, 2, 3)
            ax2 = fig.add_subplot(1, 4, 3)
            ax3 = fig.add_subplot(1, 4, 4)
            axs = [ax0, ax1, ax2, ax3]
        fig.subplots_adjust(left=0.03, bottom=0.06, right=0.99, top=0.92, wspace=0.25, hspace=0.10)
        
        range_disp = get_min_max_thresholds(img, thresholds_disp[0], thresholds_disp[1], nbins_disp)
        axs[0].imshow(img, cmap='Greys', vmin=range_disp[0], vmax=range_disp[1])
        axs[0].grid(True)
        axs[0].set_title(img_label)
        axs[0].text(0, 1.1 + (xy_ratio-1.0)/20.0, res_fname, transform=axs[0].transAxes)
        axs[1].imshow(data_ACR_log, extent=[-xsz//2-1, xsz//2, -ysz//2-1, ysz//2])
        axs[1].grid(True)
        axs[1].set_title('Autocorrelation (log scale)')
    
        axs[2].plot(xcr, data_ACR[ysz//2, :], 'r', linewidth =0.5, label='X')
        axs[2].plot(ycr, data_ACR[:, xsz//2], 'b', linewidth =0.5, label='Y')
        axs[2].plot(rcr, r_ACR, 'g', linewidth =0.5, label='R')
        axs[2].plot(xx_mean_value, xx_mean_value*0 + x_mean_value, 'r--', linewidth =2.0, label='<X>={:.5f}'.format(x_mean_value))
        axs[2].plot(yy_mean_value, yy_mean_value*0 + y_mean_value, 'b--', linewidth =2.0, label='<Y>={:.5f}'.format(y_mean_value))
        axs[2].plot(rr_mean_value, rr_mean_value*0 + r_mean_value, 'g--', linewidth =2.0, label='<R>={:.5f}'.format(r_mean_value))
        axs[2].grid(True)
        axs[2].legend()
        axs[2].set_title('Normalized autocorr. cross-sections')
        axs[3].plot(xcr, data_ACR[ysz//2, :], 'rx', label='X data')
        axs[3].plot(ycr, data_ACR[:, xsz//2], 'bd', label='Y data')
        axs[3].plot(rcr, r_ACR, 'g+', ms=10, label='R data')
        axs[3].plot(xcr[xsz//2], data_ACR[ysz//2, xsz//2], 'md', label='Peak: {:.5f}, {:.5f}'.format(xcr[xsz//2], data_ACR[ysz//2, xsz//2]))
        axs[3].plot(x_left, xacr_left, 'r')
        axs[3].plot(x_right, xacr_right, 'r', label='X extrapolation: {:.5f}, {:.5f}'.format(x_right[0], xacr_right[0]))
        axs[3].plot(y_left, yacr_left, 'b')
        axs[3].plot(y_right, yacr_right, 'b', label='Y extrapolation: {:.5f}, {:.5f}'.format(y_right[0], yacr_right[0]))
        axs[3].plot(r_left, racr_left, 'g')
        axs[3].plot(r_right, racr_right, 'g', label='R extrapolation: {:.5f}, {:.5f}'.format(r_right[0], racr_right[0]))
        axs[3].text(0.3, 0.56, 'xSNR = {:.2f}'.format(xSNR), color='r', transform=axs[3].transAxes, fontsize=fs)
        axs[3].text(0.3, 0.50, 'ySNR = {:.2f}'.format(ySNR), color='b', transform=axs[3].transAxes, fontsize=fs)
        axs[3].text(0.3, 0.44, 'rSNR = {:.2f}'.format(rSNR), color='g', transform=axs[3].transAxes, fontsize=fs)
        axs[3].grid(True)
        axs[3].legend()
        axs[3].set_xlim(-5,5)
        axs[3].set_title('Normalized autocorr. cross-sections')

        if save_res_png:
            #print('X:   ACR peak = {:.4f}, Noise-Free ACR Peak = {:.4f}, Squared Mean = {:.4f}'.format(x_acr, x_noise_free_acr, x_mean_value ))
            #print('xSNR = {:.2f}'.format(xSNR))
            #print('Y:   ACR peak = {:.4f}, Noise-Free ACR Peak = {:.4f}, Squared Mean = {:.4f}'.format(y_acr, y_noise_free_acr, y_mean_value ))
            #print('ySNR = {:.4f}'.format(ySNR))
            #print('R:   ACR peak = {:.4f}, Noise-Free ACR Peak = {:.4f}, Squared Mean = {:.4f}'.format(r_acr, r_noise_free_acr, r_mean_value ))
            #print('rSNR = {:.4f}'.format(rSNR))
            fig.savefig(res_fname, dpi=dpi)
            print('Saved the results into the file: ', res_fname)
    
    return xSNR, ySNR, rSNR


def Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs):
    '''
    Analyses the noise statistics in the selected ROI's of the EM data
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    
    Performs following:
    1.  Smooth the image by 2D convolution with a given kernel.
    2.  Determine "Noise" as difference between the original raw and smoothed data.
    3.  Build a histogram of Smoothed Image.
    4.  For each histogram bin of the Smoothed Image (Step 3), calculate the mean value and variance for the same pixels in the original image.
    5.  Plot the dependence of the noise variance vs. image intensity.
    6.  One of the parameters is a DarkCount. If it is not explicitly defined as input parameter, it will be set to 0.
    7.  The equation is determined for a line that passes through the points Intensity=DarkCount and Noise Variance = 0 and is a best fit for
        the [Mean Intensity, Noise Variance] points determined for each ROI (Step 1 above).
    8.  The data is plotted. Following values of SNR are defined from the slope of the line in Step 7:
        a.  PSNR (Peak SNR) = Intensity /sqrt(Noise Variance) at the intensity at the histogram peak determined in the Step 3.
        b.  MSNR (Mean SNR) = Mean Intensity /sqrt(Noise Variance)
        c.  DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance), where Max and Min Intensity are determined by
            corresponding cumulative threshold parameters, and Noise Variance is taken at the intensity in the middle of the range
            (Min Intensity + Max Intensity)/2.0.

    Parameters
    ----------
    img : 2D array
        original image
    Noise_ROIs : list of lists: [[left, right, top, bottom]]
        list of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the noise.
    Hist_ROI : list [left, right, top, bottom]
        coordinates (indices) of the boundaries of the image subset to evaluate the real data histogram.

    kwargs:
    DarkCount : float
        the value of the Intensity Data at 0.
    kernel : 2D float array
        a kernel to perform 2D smoothing convolution.
    nbins_disp : int
        (default 256) number of histogram bins for building the PDF and CDF to determine the data range for data display.
    thresholds_disp : list [thr_min_disp, thr_max_disp]
        (default [1e-3, 1e-3]) CDF threshold for determining the min and max data values for display.
    nbins_analysis : int
        (default 256) number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5.
    thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
        (default [2e-2, 2e-2]) CDF threshold for building the data histogram in Step 5.
    nbins_analysis : int
         (default 256) number of histogram bins for building the data histogram in Step 5.
    disp_res : boolean
        (default is False) - to plot/ display the results
    save_res_png : boolean
        save the analysis output into a PNG file (default is True)
    res_fname : string
        filename for the sesult image ('SNR_result.png')
    img_label : string
        optional image label
    Notes : string
        optional additional notes
    dpi : int

    Returns:
    mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR
        mean_vals and var_vals are the Mean Intensity and Noise Variance values for the Noise_ROIs (Step 1)
        NF_slope is the slope of the linear fit curve (Step 4)
        PSNR and DSNR are Peak and Dynamic SNR's (Step 6)
    '''
    st = 1.0/np.sqrt(2.0)
    def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
    def_kernel = def_kernel/def_kernel.sum()
    kernel = kwargs.get("kernel", def_kernel)
    DarkCount = kwargs.get("DarkCount", 0)
    nbins_disp = kwargs.get("nbins_disp", 256)
    thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    nbins_analysis = kwargs.get("nbins_analysis", 100)
    thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
    disp_res = kwargs.get("disp_res", True)
    save_res_png = kwargs.get("save_res_png", True)
    res_fname = kwargs.get("res_fname", 'Noise_Analysis_ROIs.png')
    img_label = kwargs.get("img_label", '')
    Notes = kwargs.get("Notes", '')
    dpi = kwargs.get("dpi", 300)

    fs=11
    img_filtered = convolve2d(img, kernel, mode='same')
    range_disp = get_min_max_thresholds(img_filtered, thresholds_disp[0], thresholds_disp[1], nbins_disp, False)
    
    xi, xa, yi, ya = Hist_ROI
    img_hist = img[yi:ya, xi:xa]
    img_hist_filtered = img_filtered[yi:ya, xi:xa]
    
    range_analysis = get_min_max_thresholds(img_hist_filtered, thresholds_analysis[0], thresholds_analysis[1], nbins_analysis, False)
    print('The EM data range for noise analysis: {:.1f} - {:.1f},  DarkCount={:.1f}'.format(range_analysis[0], range_analysis[1], DarkCount))
    bins_analysis = np.linspace(range_analysis[0], range_analysis[1], nbins_analysis)
    
    xy_ratio = img.shape[1]/img.shape[0]
    xsz = 15
    ysz = xsz*3.5/xy_ratio

    n_ROIs = len(Noise_ROIs)+1
    mean_vals = np.zeros(n_ROIs)
    var_vals = np.zeros(n_ROIs)
    mean_vals[0] = DarkCount

    if disp_res: 
        fig = plt.figure(figsize=(xsz,ysz))
        axs0 = fig.add_subplot(3,1,1)
        axs1 = fig.add_subplot(3,1,2)
        axs2 = fig.add_subplot(3,3,7)
        axs3 = fig.add_subplot(3,3,8)
        axs4 = fig.add_subplot(3,3,9)
        fig.subplots_adjust(left=0.01, bottom=0.06, right=0.99, top=0.95, wspace=0.25, hspace=0.10)
        
        axs0.text(0.01, 1.13, res_fname + ',   ' +  Notes, transform=axs0.transAxes, fontsize=fs-3)
        axs0.imshow(img, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs0.axis(False)
        axs0.set_title('Original Image: ' + img_label, color='r', fontsize=fs+1)
        Hist_patch = patches.Rectangle((xi,yi),abs(xa-xi)-2,abs(ya-yi)-2, linewidth=1.0, edgecolor='white',facecolor='none')
        axs1.add_patch(Hist_patch)
        
        axs2.imshow(img_hist_filtered, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs2.axis(False)
        axs2.set_title('Smoothed ROI', fontsize=fs+1)
        
    if disp_res:
        hist, bins, hist_patches = axs3.hist(img_hist_filtered.ravel(), range=range_disp, bins = nbins_disp)
    else:
        hist, bins = np.histogram(img_hist_filtered.ravel(), range=range_disp, bins = nbins_disp)

    bin_centers = np.array(bins[1:] - (bins[1]-bins[0])/2.0)
    hist_center_ind = np.argwhere((bin_centers>range_analysis[0]) & (bin_centers<range_analysis[1]))
    hist_smooth = savgol_filter(np.array(hist), (nbins_disp//10)*2+1, 7)
    I_peak = bin_centers[hist_smooth.argmax()]
    I_mean = np.mean(img)
    C_peak = hist_smooth.max()

    if disp_res:
        axs3.plot(bin_centers[hist_center_ind], hist_smooth[hist_center_ind], color='grey', linestyle='dashed', linewidth=2)
        Ipeak_lbl = '$I_{peak}$' +'={:.1f}'.format(I_peak)
        axs3.plot(I_peak, C_peak, 'rd', label = Ipeak_lbl)
        axs3.set_title('Histogram of the Smoothed ROI', fontsize=fs+1)
        axs3.grid(True)
        axs3.set_xlabel('Smoothed ROI Image Intensity', fontsize=fs+1)
        for hist_patch in np.array(hist_patches)[bin_centers<range_analysis[0]]:
            hist_patch.set_facecolor('lime')
        for hist_patch in np.array(hist_patches)[bin_centers>range_analysis[1]]:
            hist_patch.set_facecolor('red')
        ylim3=array(axs3.get_ylim())
        I_min, I_max = range_analysis 
        axs3.plot([I_min, I_min],[ylim3[0]-1000, ylim3[1]], color='lime', linestyle='dashed', label='$I_{min}$' +'={:.1f}'.format(I_min))
        axs3.plot([I_max, I_max],[ylim3[0]-1000, ylim3[1]], color='red', linestyle='dashed', label='$I_{max}$' +'={:.1f}'.format(I_max))
        axs3.set_ylim(ylim3)
        axs3.legend(loc='upper right', fontsize=fs+1)
        axs1.imshow(img_filtered, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs1.axis(False)
        axs1.set_title('Smoothed Image')
        axs4.plot(DarkCount, 0, 'd', color = 'black', label='Dark Count')
    
    for j, ROI in enumerate(tqdm(Noise_ROIs, desc = 'analyzing ROIs')):
        xi, xa, yi, ya = ROI
        img_ROI = img[yi:ya, xi:xa]
        img_ROI_filtered = img_filtered[yi:ya, xi:xa]

        imdiff = (img_ROI - img_ROI_filtered)
        x = np.mean(img_ROI)
        y = np.var(imdiff)
        mean_vals[j+1] = x
        var_vals[j+1] = y
        
        if disp_res:
            patch_col = get_cmap("gist_rainbow_r")((j)/(n_ROIs))
            rect_patch = patches.Rectangle((xi,yi),abs(xa-xi)-2,abs(ya-yi)-2, linewidth=0.5, edgecolor=patch_col,facecolor='none')
            axs0.add_patch(rect_patch)
            axs4.plot(x, y, 'd', color = patch_col) #, label='patch {:d}'.format(j))
    
    NF_slope = np.mean(var_vals[1:]/(mean_vals[1:]-DarkCount))
    mean_val_fit = np.array([mean_vals.min(), mean_vals.max()])
    var_val_fit = (mean_val_fit-DarkCount)*NF_slope
    VAR = (I_peak - DarkCount) * NF_slope
    VAR_at_mean = ((I_max + I_min) /2.0 - DarkCount) * NF_slope
    PSNR = (I_peak - DarkCount)/np.sqrt(VAR)
    MSNR = (I_mean - DarkCount)/np.sqrt((I_mean - DarkCount)*NF_slope)
    DSNR = (I_max - I_min)/np.sqrt(VAR_at_mean)
    
    if disp_res:
        axs4.grid(True)
        axs4.set_title('Noise Distribution', fontsize=fs+1)
        axs4.set_xlabel('ROI Image Intensity Mean', fontsize=fs+1)
        axs4.set_ylabel('ROI Image Intensity Variance', fontsize=fs+1)
        axs4.plot(mean_val_fit, var_val_fit, color='orange', label='Fit:  y = (x {:.1f}) * {:.2f}'.format(DarkCount, NF_slope))
        axs4.legend(loc = 'upper left', fontsize=fs+2)
        ylim4=array(axs4.get_ylim())
        V_min = (I_min-DarkCount)*NF_slope
        V_max = (I_max-DarkCount)*NF_slope
        V_peak = (I_peak-DarkCount)*NF_slope
        axs4.plot([I_min, I_min],[ylim4[0], V_min], color='lime', linestyle='dashed', label='$I_{min}$' +'={:.1f}'.format(I_min))
        axs4.plot([I_max, I_max],[ylim4[0], V_max], color='red', linestyle='dashed', label='$I_{max}$' +'={:.1f}'.format(I_max))
        axs4.plot([I_peak, I_peak],[ylim4[0], V_peak], color='black', linestyle='dashed', label='$I_{peak}$' +'={:.1f}'.format(I_peak))
        axs4.set_ylim(ylim4)
        txt1 = 'Peak Intensity:  {:.1f}'.format(I_peak)
        axs4.text(0.05, 0.65, txt1, transform=axs4.transAxes, fontsize=fs+1)
        txt2 = 'Variance={:.1f}, STD={:.1f}'.format(VAR, np.sqrt(VAR))
        axs4.text(0.05, 0.55, txt2, transform=axs4.transAxes, fontsize=fs+1)
        txt3 = 'PSNR = {:.2f}'.format(PSNR)
        axs4.text(0.05, 0.45, txt3, transform=axs4.transAxes, fontsize=fs+1)
        txt3 = 'DSNR = {:.2f}'.format(DSNR)
        axs4.text(0.05, 0.35, txt3, transform=axs4.transAxes, fontsize=fs+1)
        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print('results saved into the file: '+res_fname)

    return mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR


def Single_Image_Noise_Statistics(img, **kwargs):
    '''
    Analyses the noise statistics of the EM data image.
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    
    Performs following:
    1. Smooth the image by 2D convolution with a given kernel.
    2. Determine "Noise" as difference between the original raw and smoothed data.
    3. Build a histogram of Smoothed Image.
    4. For each histogram bin of the Smoothed Image (Step 3), calculate the mean value and variance for the same pixels in the original image.
    5. Plot the dependence of the noise variance vs. image intensity.
    6. One of the parameters is a DarkCount. If it is not explicitly defined as input parameter,
        it will be set to 0
    7. The equation is determined for a line that passes through the point:
            Intensity=DarkCount and Noise Variance = 0
            and is a best fit for the [Mean Intensity, Noise Variance] points
            determined for each ROI (Step 1 above).
    8. The data is plotted. Two values of SNR are defined from the slope of the line in Step 4:
        PSNR (Peak SNR) = Intensity /sqrt(Noise Variance) at the intensity
            at the histogram peak determined in the Step 5.
        MSNR (Mean SNR) = Mean Intensity /sqrt(Noise Variance)
        DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance),
            where Max and Min Intensity are determined by corresponding cummulative
            threshold parameters, and Noise Variance is taken at the intensity
            in the middle of the range (Min Intensity + Max Intensity)/2.0

    Parameters
    ----------
        img : 2d array

        kwargs:
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        DarkCount : float
            the value of the Intensity Data at 0.
        kernel : 2D float array
            a kernel to perfrom 2D smoothing convolution.
        nbins_disp : int
            (default 256) number of histogram bins for building the PDF and CDF to determine the data range for data display.
        thresholds_disp : list [thr_min_disp, thr_max_disp]
            (default [1e-3, 1e-3]) CDF threshold for determining the min and max data values for display.
        nbins_analysis : int
            (default 256) number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5.
        thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
            (default [2e-2, 2e-2]) CDF threshold for building the data histogram in Step 5.
        nbins_analysis : int
             (default 256) number of histogram bins for building the data histogram in Step 5.
        disp_res : boolean
            (default is False) - to plot/ display the results
        save_res_png : boolean
            save the analysis output into a PNG file (default is True)
        res_fname : string
            filename for the result image ('Noise_Analysis.png')
        img_label : string
            optional image label
        Notes : string
            optional additional notes
        dpi : int

    Returns:
    mean_vals, var_vals, I0, PSNR, DSNR, popt, result
        mean_vals and var_vals are the Mean Intensity and Noise Variance values for Step5, I0 is zero intercept (should be close to DarkCount)
        PSNR and DSNR are Peak and Dynamic SNR's (Step 8)
    '''
    st = 1.0/np.sqrt(2.0)
    def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    def_kernel = def_kernel/def_kernel.sum()
    kernel = kwargs.get("kernel", def_kernel)
    DarkCount = kwargs.get("DarkCount", 0)
    nbins_disp = kwargs.get("nbins_disp", 256)
    thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    nbins_analysis = kwargs.get("nbins_analysis", 100)
    thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
    disp_res = kwargs.get("disp_res", True)
    save_res_png = kwargs.get("save_res_png", True)
    res_fname = kwargs.get("res_fname", 'Noise_Analysis.png')
    image_name = kwargs.get("image_name", '')
    Notes = kwargs.get("Notes", '')
    dpi = kwargs.get("dpi", 300)

    xi = 0
    yi = 0
    ysz, xsz = img.shape
    xa = xi + xsz
    ya = yi + ysz

    xi_eval = xi + evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = xa
    yi_eval = yi + evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ya

    img = img[yi_eval:ya_eval, xi_eval:xa_eval]
    img_filtered = convolve2d(img, kernel, mode='same')[1:-1, 1:-1]
    img = img[1:-1, 1:-1]
    
    range_disp = get_min_max_thresholds(img_filtered, thresholds_disp[0], thresholds_disp[1], nbins_disp)

    print('The EM data range for display:            {:.1f} - {:.1f}'.format(range_disp[0], range_disp[1]))
    range_analysis = get_min_max_thresholds(img_filtered, thresholds_analysis[0], thresholds_analysis[1], nbins_analysis, False)
    print('The EM data range for noise analysis:     {:.1f} - {:.1f}'.format(range_analysis[0], range_analysis[1]))
    bins_analysis = np.linspace(range_analysis[0], range_analysis[1], nbins_analysis)

    imdiff = (img-img_filtered)
    range_imdiff = get_min_max_thresholds(imdiff, thresholds_disp[0], thresholds_disp[1], nbins_disp)

    xy_ratio = img.shape[1]/img.shape[0]
    xsz = 15
    ysz = xsz/1.5*xy_ratio
    if disp_res:
        fs=11
        fig, axss = subplots(2,3, figsize=(xsz,ysz),  gridspec_kw={"height_ratios" : [1,1]})
        fig.subplots_adjust(left=0.07, bottom=0.06, right=0.99, top=0.92, wspace=0.15, hspace=0.10)
        axs = axss.ravel()
        axs[0].text(-0.1, 1.1, res_fname + ',       ' +  Notes, transform=axs[0].transAxes, fontsize=fs)

        axs[0].imshow(img, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs[0].axis(False)
        axs[0].set_title('Original Image: ' + image_name, color='r', fontsize=fs+1)

        axs[1].imshow(img_filtered, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs[1].axis(False)
        axs[1].set_title('Smoothed Image')
        Low_mask = img*0.0+255.0
        High_mask = Low_mask.copy()
        Low_mask[img_filtered > range_analysis[0]] = np.nan
        axs[1].imshow(Low_mask, cmap="brg_r")
        High_mask[img_filtered < range_analysis[1]] = np.nan
        axs[1].imshow(High_mask, cmap="gist_rainbow")


        axs[2].imshow(imdiff, cmap="Greys", vmin = range_imdiff[0], vmax = range_imdiff[1])
        axs[2].axis(False)
        axs[2].set_title('Image Difference', fontsize=fs+1)

    if disp_res:
        hist, bins, patches = axs[4].hist(img_filtered.ravel(), range=range_disp, bins = nbins_disp)
    else:
        hist, bins = np.histogram(img_hist_filtered.ravel(), range=range_disp, bins = nbins_disp)
    bin_centers = np.array(bins[1:] - (bins[1]-bins[0])/2.0)
    hist_center_ind = np.argwhere((bin_centers>range_analysis[0]) & (bin_centers<range_analysis[1]))
    hist_smooth = savgol_filter(np.array(hist), (nbins_disp//10)*2+1, 7)
    I_peak = bin_centers[hist_smooth.argmax()]
    C_peak = hist_smooth.max()
    Ipeak_lbl = '$I_{peak}$' +'={:.1f}'.format(I_peak)
    
    if disp_res:
        axs[4].plot(bin_centers[hist_center_ind], hist_smooth[hist_center_ind], color='grey', linestyle='dashed', linewidth=2)
        axs[4].plot(I_peak, C_peak, 'rd', label = Ipeak_lbl)
        axs[4].legend(loc='upper left', fontsize=fs+1)
        axs[4].set_title('Histogram of the Smoothed Image', fontsize=fs+1)
        axs[4].grid(True)
        axs[4].set_xlabel('Image Intensity', fontsize=fs+1)
        for patch in np.array(patches)[bin_centers<range_analysis[0]]:
            patch.set_facecolor('lime')
        for patch in np.array(patches)[bin_centers>range_analysis[1]]:
            patch.set_facecolor('red')
        ylim4=array(axs[4].get_ylim())
        axs[4].plot([range_analysis[0], range_analysis[0]],[ylim4[0]-1000, ylim4[1]], color='lime', linestyle='dashed', label='Ilow')
        axs[4].plot([range_analysis[1], range_analysis[1]],[ylim4[0]-1000, ylim4[1]], color='red', linestyle='dashed', label='Ihigh')
        axs[4].set_ylim(ylim4)
        txt1 = 'Smoothing Kernel'
        axs[4].text(0.69, 0.955, txt1, transform=axs[4].transAxes, backgroundcolor='white', fontsize=fs-1)
        txt2 = '{:.3f}  {:.3f}  {:.3f}'.format(kernel[0,0], kernel[0,1], kernel[0,2])
        axs[4].text(0.69, 0.910, txt2, transform=axs[4].transAxes, backgroundcolor='white', fontsize=fs-2)
        txt3 = '{:.3f}  {:.3f}  {:.3f}'.format(kernel[1,0], kernel[1,1], kernel[1,2])
        axs[4].text(0.69, 0.865, txt3, transform=axs[4].transAxes, backgroundcolor='white', fontsize=fs-2)
        txt3 = '{:.3f}  {:.3f}  {:.3f}'.format(kernel[2,0], kernel[2,1], kernel[2,2])
        axs[4].text(0.69, 0.820, txt3, transform=axs[4].transAxes, backgroundcolor='white', fontsize=fs-2)
    
    if disp_res:
        hist, bins, patches = axs[5].hist(imdiff.ravel(), bins = nbins_disp)
        axs[5].grid(True)
        axs[5].set_title('Histogram of the Difference Map', fontsize=fs+1)
        axs[5].set_xlabel('Image Difference', fontsize=fs+1) 
    else:
        hist, bins = np.histogram(imdiff.ravel(), bins = nbins_disp)
 
    ind_new = np.digitize(img_filtered, bins_analysis)
    result = np.array([(np.mean(img_filtered[ind_new == j]), np.var(imdiff[ind_new == j]))  for j in range(1, nbins_analysis)])
    non_nan_ind = np.argwhere(np.invert(np.isnan(result[:, 0])))
    mean_vals = np.squeeze(result[non_nan_ind, 0])
    var_vals = np.squeeze(result[non_nan_ind, 1])
    try:
        popt = np.polyfit(mean_vals, var_vals, 1)
        print('popt: ', popt)
        
        I_array = np.array((range_analysis[0], range_analysis[1], I_peak))
        print('I_array: ', I_array)
        Var_array = np.polyval(popt, I_array)
        Var_peak = Var_array[2]
    except:
        print("np.polyfit could not converge")
        popt = np.array([np.var(imdiff)/np.mean(img_filtered-DarkCount), 0])
        I_array = np.array((range_analysis[0], range_analysis[1], I_peak))
        Var_peak = np.var(imdiff)
    var_fit = np.polyval(popt, mean_vals)
    I0 = -popt[1]/popt[0]
    Slope_header = np.mean(var_vals/(mean_vals-DarkCount))
    var_fit_header = (mean_vals-DarkCount) * Slope_header
    if disp_res:
        axs[3].plot(mean_vals, var_vals, 'r.', label='data')
        axs[3].plot(mean_vals, var_fit, 'b', label='linear fit: {:.1f}*x + {:.1f}'.format(popt[0], popt[1]))
        axs[3].plot(mean_vals, var_fit_header, 'magenta', label='linear fit with header offset')
        axs[3].grid(True)
        axs[3].set_title('Noise Distribution', fontsize=fs+1)
        axs[3].set_xlabel('Image Intensity Mean', fontsize=fs+1)
        axs[3].set_ylabel('Image Intensity Variance', fontsize=fs+1)
        ylim3=array(axs[3].get_ylim())
        lbl_low = '$I_{low}$'+', thr={:.1e}'.format(thresholds_analysis[0])
        lbl_high = '$I_{high}$'+', thr={:.1e}'.format(thresholds_analysis[1])
        axs[3].plot([range_analysis[0], range_analysis[0]],[ylim3[0]-1000, ylim3[1]], color='lime', linestyle='dashed', label=lbl_low)
        axs[3].plot([range_analysis[1], range_analysis[1]],[ylim3[0]-1000, ylim3[1]], color='red', linestyle='dashed', label=lbl_high)
        axs[3].legend(loc='upper center', fontsize=fs+1)
        axs[3].set_ylim(ylim3)
    
    PSNR = (I_peak-I0)/np.sqrt(Var_peak)
    PSNR_header = (I_peak-DarkCount)/np.sqrt(Var_peak)
    DSNR = (range_analysis[1]-range_analysis[0])/np.sqrt(Var_peak)
    if disp_res:
        print('Var at peak: {:.1f}'.format(Var_peak))
        print('PSNR={:.2f}, PSNR_header={:.2f}, DSNR={:.2f}'.format(PSNR, PSNR_header, DSNR))
    
        txt1 = 'Histogram Peak:  ' + Ipeak_lbl
        axs[3].text(0.25, 0.27, txt1, transform=axs[3].transAxes, fontsize=fs+1)
        txt2 = 'DSNR = ' +'$(I_{high}$' +'$ - I_{low})$' +'/'+'$σ_{peak}$' + ' = {:.2f}'.format(DSNR)
        axs[3].text(0.25, 0.22, txt2, transform=axs[3].transAxes, fontsize=fs+1)
        
        txt3 = 'Zero Intercept:    ' +'$I_{0}$' +'={:.1f}'.format(I0)
        axs[3].text(0.25, 0.17, txt3, transform=axs[3].transAxes, color='blue', fontsize=fs+1)
        txt4 = 'PSNR = ' +'$(I_{peak}$' +'$ - I_{0})$' +'/'+'$σ_{peak}$' + ' = {:.2f}'.format(PSNR)
        axs[3].text(0.25, 0.12, txt4, transform=axs[3].transAxes, color='blue', fontsize=fs+1)

        txt5 = 'Header Zero Int.:    ' +'$I_{0}$' +'={:.1f}'.format(DarkCount)
        axs[3].text(0.25, 0.07, txt5, transform=axs[3].transAxes, color='magenta', fontsize=fs+1)
        txt6 = 'PSNR = ' +'$(I_{peak}$' +'$ - I_{0})$' +'/'+'$σ_{peak}$' + ' = {:.2f}'.format(PSNR_header)
        axs[3].text(0.25, 0.02, txt6, transform=axs[3].transAxes, color='magenta', fontsize=fs+1)

        if save_res_png:
            fig.savefig(res_fname, dpi=300)
            print('results saved into the file: '+res_fname)
    return mean_vals, var_vals, I0, PSNR, DSNR, popt, result


def Perform_2D_fit(img, estimator, **kwargs):
    '''
    Bin the image and then perform 2D polynomial (currently only 2D parabolic) fit on the binned image.
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    
    Parameters
    ----------
    img : 2D array
        original image
    estimator : RANSACRegressor,
                LinearRegression,
                TheilSenRegressor,
                HuberRegressor
    kwargs:
    bins : int
        binsize for image binning. If not provided, bins=10
    calc_corr : boolean
        If True - the full image correction is calculated
    ignore_Y  : boolean
        If True - the parabolic fit to only X is perfromed
    disp_res : boolean
        (default is False) - to plot/ display the results
    save_res_png : boolean
        save the analysis output into a PNG file (default is False)
    res_fname : string
        filename for the sesult image ('2D_Parabolic_Fit.png')
    label : string
        optional image label
    dpi : int

    Returns:
    intercept, coefs, mse, img_full_correction
    '''
    ysz, xsz = img.shape
    calc_corr = kwargs.get("calc_corr", False)
    ignore_Y = kwargs.get("ignore_Y", False)
    lbl = kwargs.get("label", '')
    Xsect = kwargs.get("Xsect", xsz//2)
    Ysect = kwargs.get("Ysect", ysz//2)
    disp_res = kwargs.get("disp_res", True)
    bins = kwargs.get("bins", 10) #bins = 10
    save_res_png = kwargs.get("save_res_png", False)
    res_fname = kwargs.get("res_fname", '2D_Parabolic_Fit.png')
    dpi = kwargs.get("dpi", 300)
    
    img_binned = img.astype(float).reshape(ysz//bins, bins, xsz//bins, bins).sum(3).sum(1)/bins/bins
    vmin, vmax = get_min_max_thresholds(img_binned)
    yszb, xszb = img_binned.shape
    yb, xb = np.indices(img_binned.shape)
    img_1D = img_binned.ravel()
    xb_1d = xb.ravel()
    yb_1d = yb.ravel()
    X = np.vstack((xb_1d, yb_1d)).T
    
    ysz, xsz = img.shape
    yf, xf = np.indices(img.shape)
    xf_1d = xf.ravel()/bins
    yf_1d = yf.ravel()/bins
    Xf = np.vstack((xf_1d, yf_1d)).T
    
    model = make_pipeline(PolynomialFeatures(2), estimator)
    model.fit(X, img_1D)
    if hasattr(model[-1], 'estimator_'):
        if ignore_Y:
            model[-1].estimator_.coef_[2]=0.0
            model[-1].estimator_.coef_[4]=0.0
            model[-1].estimator_.coef_[5]=0.0
        coefs = model[-1].estimator_.coef_
        intercept = model[-1].estimator_.intercept_
    else:
        if ignore_Y:
            model[-1].coef_[2]=0.0
            model[-1].coef_[4]=0.0
            model[-1].coef_[5]=0.0
        coefs = model[-1].coef_
        intercept = model[-1].intercept_
    img_fit_1d = model.predict(X)
    scr = model.score(X, img_1D)
    mse = mean_squared_error(img_fit_1d, img_1D)
    img_fit = img_fit_1d.reshape(yszb, xszb)
    if calc_corr:
        img_full_correction = np.mean(img_fit_1d) / model.predict(Xf).reshape(ysz, xsz)
    else:
        img_full_correction = img * 0.0
        
    if disp_res:
        print('Estimator coefficients ( 1  x  y  x^2  x*y  y^2): ', coefs)
        print('Estimator intercept: ', intercept)
        
        fig, axs = subplots(2,2, figsize = (12, 8))
        axs[0, 0].imshow(img_binned, cmap='Greys', vmin=vmin, vmax=vmax)
        axs[0, 0].grid(True)
        axs[0, 0].plot([Xsect//bins, Xsect//bins], [0, yszb], 'lime', linewidth = 0.5)
        axs[0, 0].plot([0, xszb], [Ysect//bins, Ysect//bins], 'cyan', linewidth = 0.5)
        axs[0, 0].set_xlim((0, xszb))
        axs[0, 0].set_ylim((yszb, 0))
        axs[0, 0].set_title('{:d}-x Binned Original Image'.format(bins))
  
        axs[0, 1].imshow(img_fit, cmap='Greys', vmin=vmin, vmax=vmax)
        axs[0, 1].grid(True)
        axs[0, 1].plot([Xsect//bins, Xsect//bins], [0, yszb], 'lime', linewidth = 0.5)
        axs[0, 1].plot([0, xszb], [Ysect//bins, Ysect//bins], 'cyan', linewidth = 0.5)
        axs[0, 1].set_xlim((0, xszb))
        axs[0, 1].set_ylim((yszb,0))
        axs[0, 1].set_title('{:d}-x Binned Fit Image: '.format(bins) + lbl)

        axs[1, 0].plot(img[Ysect, :],'b', label = 'Raw Image A', linewidth =0.5)
        axs[1, 0].plot(xb[0,:]*bins, img_binned[Ysect//bins, :],'cyan', label = 'Binned Orig Image A')
        axs[1, 0].plot(xb[0,:]*bins, img_fit[Ysect//bins, :], 'yellow', linewidth=4, linestyle='--', label = 'Fit: '+lbl)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_xlabel('X-coordinate')

        axs[1, 1].plot(img[:, Xsect],'g', label = 'Raw Image A', linewidth =0.5)
        axs[1, 1].plot(yb[:, 0]*bins, img_binned[:, Xsect//bins],'lime', label = 'Binned Orig Image A')
        axs[1, 1].plot(yb[:, 0]*bins, img_fit[:, Xsect//bins], 'yellow', linewidth=4, linestyle='--', label = 'Fit: '+lbl)
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_xlabel('Y-coordinate')
        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print('results saved into the file: '+res_fname)
    return intercept, coefs, mse, img_full_correction



##############################################
#      Two-Frame Image processing Functions
##############################################
def mutual_information_2d(x, y, sigma=1, bin=256, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    mi: float
        the computed similarity measure
    """
    bins = (bin, bin)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    #ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
    #                             output=jh)
    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))
    return mi


def mutual_information_2d_cp(x, y, sigma=1, bin=256, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram using CUPY package.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    mi: float
        the computed similarity measure
    """
    bins = (bin, bin)

    jhf = cp.histogram2d(x, y, bins=bins)

    # smooth the jh with a gaussian filter of given sigma
    #ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
    #                             output=jh)
    # compute marginal histograms
    jh = jhf[0] + EPS
    sh = cp.sum(jh)
    jh = jh / sh
    s1 = cp.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = cp.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((cp.sum(s1 * cp.log(s1)) + cp.sum(s2 * cp.log(s2)))
                / cp.sum(jh * cp.log(jh))) - 1
    else:
        mi = ( cp.sum(jh * cp.log(jh)) - np.sum(s1 * cp.log(s1))
               - cp.sum(s2 * cp.log(s2)))
    return mi


def Two_Image_NCC_SNR(img1, img2, **kwargs):
    '''
    Estimates normalized cross-correlation and SNR of two images.
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    
    Calculates SNR from cross-correlation of two images after [1, 2, 3].
    
    Parameters
    ---------
    img1 : 2D array
    img2 : 2D array
     
    kwargs:
    zero_mean: boolean
        if True, cross-correlation is zero-mean
    
    Returns:
        NCC, SNR : float, float
            NCC - normalized cross-correlation coefficient
            SNR - Signal-to-Noise ratio based on NCC
    
   [1] J. Frank, L. AI-Ali, Signal-to-noise ratio of electron micrographs obtained by cross correlation. Nature 256, 4 (1975).
   [2] J. Frank, in: Computer Processing of Electron Microscopic Images. Ed. P.W. Hawkes (Springer, Berlin, 1980).
   [3] M. Radermacher, T. Ruiz, On cross-correlations, averages and noise in electron microscopy. Acta Crystallogr. Sect. F Struct. Biol. Commun. 75, 12–18 (2019).
    
    '''
    zero_mean = kwargs.get("zero_mean", True)
    
    if img1.shape==img2.shape:
        ysz, xsz = img1.shape
        if zero_mean:
            img1 = img1- img1.mean()
            img2 = img2- img2.mean()
        xy = np.sum((img1.ravel()*img2.ravel()))/(xsz*ysz)
        xx = np.sum((img1.ravel()*img1.ravel()))/(xsz*ysz)
        yy = np.sum((img2.ravel()*img2.ravel()))/(xsz*ysz)
        NCC = xy / np.sqrt(xx*yy)
        SNR = NCC / (1-NCC)

    else:
        print("img1 and img2 shapes must be equal")
        NCC = 0.0
        SNR = 0.0
    
    return NCC, SNR

def Two_Image_FSC(img1, img2, **kwargs):
    '''
    Perform Fourier Shell Correlation to determine the image resolution, after [1]. ©G.Shtengel, 10/2019. gleb.shtengel@gmail.com
    FSC is determined from radially averaged foirier cross-correlation (with optional selection of range of angles for radial averaging).
    
    Parameters
    ---------
    img1 : 2D array
    img2 : 2D array
     
    kwargs:
    SNRt : float
        SNR threshold for determining the resolution bandwidth
    astart : float
        Start angle for radial averaging. Default is 0
    astop : float
        Stop angle for radial averaging. Default is 90
    symm : int
        Symmetry factor (how many times Start and stop angle intervalks are repeated within 360 deg). Default is 4.
    disp_res : boolean
        display results (plots) (default is False)
    ax : axis object (matplotlip)
        to export the plot
    save_res_png : boolean
        save results into PNG file (default is False)
    res_fname : string
        filename for the result image ('SNR_result.png')
    img_labels : [string, string]
        optional image labels
    dpi : int
        dots-per-inch resolution for the output image
    pixel : float
        optional pixel size in nm. If not provided, will be ignored.
        if provided, second axis will be added on top with inverse pixels
    xrange : [float, float]
        range of x axis in FSC plot in inverse pixels
        if not provided [0, 0.5] range will be used
    
    Returns FSC_sp_frequencies, FSC_data, x2, T
        FSC_sp_frequencies : float array 
            Spatial Frequency (/Nyquist) - for FSC plot
        FSC_data: float array
        x2 : float array
            Spatial Frequency (/Nyquist) - for threshold line plot
        T : float array
            threshold line plot
        FSC_bw : float
            the value of FSC determined as an intersection of smoothed data threshold
        
    [1]. M. van Heela, and M. Schatzb, "Fourier shell correlation threshold criteria," Journal of Structural Biology 151, 250-262 (2005)
    '''   
    SNRt = kwargs.get("SNRt", 0.1)
    astart = kwargs.get("astart", 0.0)
    astop = kwargs.get("astop", 90.0)
    symm = kwargs.get("symm", 4)
    disp_res = kwargs.get("disp_res", False)
    ax = kwargs.get("ax", '')
    save_res_png = kwargs.get("save_res_png", False)
    res_fname = kwargs.get("res_fname", 'FSC_results.png')
    img_labels = kwargs.get("img_labels", ['Image 1', 'Image 2'])
    dpi = kwargs.get("dpi", 300)
    pixel = kwargs.get("pixel", 0.0)
    xrange = kwargs.get("xrange", [0, 0.5])
    
    #Check whether the inputs dimensions match and the images are square
    if ( np.shape(img1) != np.shape(img2) ) :
        print('input images must have the same dimensions')
    if ( np.shape(img1)[0] != np.shape(img1)[1]) :
        print('input images must be squares')
    I1 = fftshift(fftn(ifftshift(img1)))  # I1 and I2 store the FFT of the images to be used in the calcuation for the FSC
    I2 = fftshift(fftn(ifftshift(img2)))

    C_imre = np.multiply(I1,np.conj(I2))
    C12_ar = abs(np.multiply((I1+I2),np.conj(I1+I2)))
    y0,x0 = argmax2d(C12_ar)
    C1 = radial_profile_select_angles(abs(np.multiply(I1,np.conj(I1))), [x0,y0], astart, astop, symm)
    C2 = radial_profile_select_angles(abs(np.multiply(I2,np.conj(I2))), [x0,y0], astart, astop, symm)
    C  = radial_profile_select_angles(np.real(C_imre), [x0,y0], astart, astop, symm) + 1j * radial_profile_select_angles(np.imag(C_imre), [x0,y0], astart, astop, symm)

    FSC_data = abs(C)/np.sqrt(abs(np.multiply(C1,C2)))
    '''
    T is the SNR threshold calculated accoring to the input SNRt, if nothing is given
    a default value of 0.1 is used.
    
    x2 contains the normalized spatial frequencies
    '''
    r = np.arange(1+np.shape(img1)[0])
    n = 2*np.pi*r
    n[0] = 1
    eps = np.finfo(float).eps
    t1 = np.divide(np.ones(np.shape(n)),n+eps)
    t2 = SNRt + 2*np.sqrt(SNRt)*t1 + np.divide(np.ones(np.shape(n)),np.sqrt(n))
    t3 = SNRt + 2*np.sqrt(SNRt)*t1 + 1
    T = np.divide(t2,t3)
    #FSC_sp_frequencies = np.arange(np.shape(C)[0])/(np.shape(img1)[0]/sqrt(2.0))
    #x2 = r/(np.shape(img1)[0]/sqrt(2.0))
    FSC_sp_frequencies = np.arange(np.shape(C)[0])/(np.shape(img1)[0])
    x2 = r/(np.shape(img1)[0])
    FSC_data_smooth = smooth(FSC_data, 20)
    FSC_bw = find_BW(FSC_sp_frequencies, FSC_data_smooth, SNRt)
    '''
    If the disp_res input is set to True, an output plot is generated. 
    '''
    if disp_res:
        if ax=='':
            fig = plt.figure(figsize=(8,12))
            axs0 = fig.add_subplot(3,2,1)
            axs1 = fig.add_subplot(3,2,2)
            axs2 = fig.add_subplot(3,2,3)
            axs3 = fig.add_subplot(3,2,4)
            ax = fig.add_subplot(3,1,3)
            fig.subplots_adjust(left=0.01, bottom=0.06, right=0.99, top=0.975, wspace=0.25, hspace=0.10)
            vmin1, vmax1 = get_min_max_thresholds(img1)
            vmin2, vmax2 = get_min_max_thresholds(img2)
            axs0.imshow(img1, cmap='Greys', vmin=vmin1, vmax=vmax1)
            axs1.imshow(img2, cmap='Greys', vmin=vmin2, vmax=vmax2)
            x = np.linspace(0, 1.41, 500)
            axs2.set_xlim(-1,1)
            axs2.set_ylim(-1,1)
            axs2.imshow(np.log(abs(I1)), extent=[-1, 1, -1, 1], cmap = 'Greys_r')
            axs3.set_xlim(-1,1)
            axs3.set_ylim(-1,1)
            axs3.imshow(np.log(abs(I2)), extent=[-1, 1, -1, 1], cmap = 'Greys_r')
            for i in np.arange(symm):
                ai = np.radians(astart + 360.0/symm*i)
                aa = np.radians(astop + 360.0/symm*i)
                axs2.plot(x * np.cos(ai), x * np.sin(ai), color='orange', linewidth = 0.5)
                axs3.plot(x * np.cos(ai), x * np.sin(ai), color='orange', linewidth = 0.5)
                axs2.plot(x * np.cos(aa), x * np.sin(aa), color='orange', linewidth = 0.5)
                axs3.plot(x * np.cos(aa), x * np.sin(aa), color='orange', linewidth = 0.5)
            ttls = img_labels.copy()
            ttls.append('FFT of '+img_labels[0])
            ttls.append('FFT of '+img_labels[1])
            for axi, ttl in zip([axs0, axs1, axs2, axs3], ttls):
                axi.grid(False)
                axi.axis(False)
                axi.set_title(ttl)

    if disp_res or ax != '':
        ax.plot(FSC_sp_frequencies, FSC_data, label = 'FSC data', color='r')
        ax.plot(FSC_sp_frequencies, FSC_data_smooth, label = 'FSC data smoothed', color='b')
        ax.plot(x2, x2*0.0+SNRt, '--', label = 'Threshold SNR = {:.3f}'.format(SNRt), color='m')
        if pixel>1e-6:
            label = 'FSC BW = {:.3f} inv.pix., or {:.2f} nm'.format(FSC_bw, pixel/FSC_bw)
        else:
            label = 'FSC BW = {:.3f}'.format(FSC_bw)
        ax.plot(np.array((FSC_bw,FSC_bw)), np.array((0.0,1.0)), '--', label = label, color = 'g')
        ax.set_xlim(xrange)
        ax.legend()
        ax.set_xlabel('Spatial Frequency (inverse pixels)')
        ax.set_ylabel('FSC Magnitude')
        ax.grid(True)
        if pixel>1e-6:
            
            def forward(x):
                return x/pixel
            def inverse(x):
                return x*pixel
            secax = ax.secondary_xaxis('top', functions=(forward, inverse))
            secax.set_xlabel('Spatial Frequency ($nm^{-1}$)') 
        
    if disp_res:
        ax.set_position([0.1, 0.05, 0.85, 0.28])
        print('FSC BW = {:.5f}'.format(FSC_bw))
        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print('Saved the results into the file: ', res_fname)
    return (FSC_sp_frequencies, FSC_data, x2, T, FSC_bw)


##########################################
#         MRC stack analysis functions
##########################################

def evaluate_registration_two_frames(params_mrc):
    '''
    Helper function used by DASK routine. Analyzes registration between two frames.
    ©G.Shtengel, 10/2020. gleb.shtengel@gmail.com

    Parameters:
    params_mrc : list of mrc_filename, fr, evals
    mrc_filename  : string
        full path to mrc filename
    fr : int
        Index of the SECOND frame
    evals :  list of image bounds to be used for evaluation exi_eval, xa_eval, yi_eval, ya_eval 


    Returns:
    image_nsad, image_ncc, image_mi   : float, float, float

    '''
    mrc_filename, fr, invert_data, evals = params_mrc
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r')
    header = mrc_obj.header
    try: 
        dmin = float(header['dmin'])
        dmax = float(header['dmax'])
        if (dmin >= 0 and dmax<=255):
            dt = uint8
        else:
            dt = int16
    except:
        dt = int16

    xi_eval, xa_eval, yi_eval, ya_eval = evals
    if invert_data:
        prev_frame = -1.0 * (mrc_obj.data[fr-1, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt)).astype(float)
        curr_frame = -1.0 * (mrc_obj.data[fr, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt)).astype(float)
    else:
        prev_frame = (mrc_obj.data[fr-1, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt)).astype(float)
        curr_frame = (mrc_obj.data[fr, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt)).astype(float)
    fr_mean = curr_frame/2.0 + prev_frame/2.0
    #image_nsad =  np.mean(np.abs(curr_frame-prev_frame))/(np.mean(fr_mean)-np.amin(fr_mean))
    image_nsad =  np.mean(np.abs(curr_frame-prev_frame))/(np.mean(fr_mean)
    image_ncc = Two_Image_NCC_SNR(curr_frame, prev_frame)[0]
    image_mi = mutual_information_2d(prev_frame.ravel(), curr_frame.ravel(), sigma=1.0, bin=2048, normalized=True)
    mrc_obj.close()
    return image_nsad, image_ncc, image_mi


def analyze_mrc_stack_registration(mrc_filename, DASK_client, **kwargs):
    '''
    Read MRC stack and analyze registration - calculate NSAD, NCC, and MI.
    ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

    Parameters
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed
    DASK client (needs to be initialized and running by this time)

    kwargs:
     use_DASK : boolean
        use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
    frame_inds : array
        Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    sliding_evaluation_box : boolean
        if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    save_filename : str
        Path to the filename to save the results. If empty, mrc_filename+'_RegistrationQuality.csv' will be used

    Returns reg_summary : PD data frame
    '''
    Sample_ID = kwargs.get("Sample_ID", '')
    save_res_png  = kwargs.get("save_res_png", True )
    save_filename = kwargs.get("save_filename", mrc_filename )
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])

    if sliding_evaluation_box:
        print('Will use sliding (linearly) evaluation box')
        print('   Starting with box:  ', start_evaluation_box)
        print('   Finishing with box: ', stop_evaluation_box)
    else:
        print('Will use fixed evaluation box: ', evaluation_box)

    use_DASK = kwargs.get("use_DASK", False)
    invert_data =  kwargs.get("invert_data", False)
   
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r')
    header = mrc_obj.header
    nx, ny, nz = int32(header['nx']), int32(header['ny']), int32(header['nz'])
    try: 
        dmin = float(header['dmin'])
        dmax = float(header['dmax'])
        if (dmin >= 0 and dmax<=255):
            dt = uint8
        else:
            dt = int16
    except:
        dt = int16

    xi_eval = evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = nx
    yi_eval = evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ny
    evals = [xi_eval, xa_eval, yi_eval, ya_eval]
    
    frame_inds_default = np.arange(nz-1)+1
    frame_inds = np.array(kwargs.get("frame_inds", frame_inds_default))
    nf = frame_inds[-1]-frame_inds[0]+1
    if frame_inds[0]==0:
        frame_inds = frame_inds+1
    print('Will analyze regstrations in {:d} frames'.format(len(frame_inds)))
    print('Will save the data into '+os.path.splitext(save_filename)[0] + '_RegistrationQuality.csv')
    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0
    
    params_mrc_mult = []
    for j, fr in enumerate(frame_inds):
        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval*(fr-frame_inds[0])//nf
            yi_eval = start_evaluation_box[0] + dy_eval*(fr-frame_inds[0])//nf
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny
            evals = [xi_eval, xa_eval, yi_eval, ya_eval]
        params_mrc_mult.append([mrc_filename, fr, invert_data, evals])
    #params_mrc_mult = [[mrc_filename, fr, evals] for fr in frame_inds]
        
    if use_DASK:
        mrc_obj.close()
        print('Using DASK distributed')
        futures = DASK_client.map(evaluate_registration_two_frames, params_mrc_mult)
        dask_results = DASK_client.gather(futures)
        image_nsad = np.array([res[0] for res in dask_results])
        image_ncc = np.array([res[1] for res in dask_results])
        image_mi = np.array([res[2] for res in dask_results])
    else:
        print('Using Local Computation')
        image_nsad = np.zeros((nf), dtype=float)
        image_ncc = np.zeros((nf), dtype=float)
        image_mi = np.zeros((nf), dtype=float)
        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval*frame_inds[0]//nf
            yi_eval = start_evaluation_box[0] + dy_eval*frame_inds[0]//nf
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny
        prev_frame = (mrc_obj.data[frame_inds[0]-1, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt)).astype(float)
        for j in tqdm(frame_inds, desc='Evaluating frame registration: '):
            if sliding_evaluation_box:
                xi_eval = start_evaluation_box[2] + dx_eval*j//nf
                yi_eval = start_evaluation_box[0] + dy_eval*j//nf
                if start_evaluation_box[3] > 0:
                    xa_eval = xi_eval + start_evaluation_box[3]
                else:
                    xa_eval = nx
                if start_evaluation_box[1] > 0:
                    ya_eval = yi_eval + start_evaluation_box[1]
                else:
                    ya_eval = ny
            curr_frame = (mrc_obj.data[j, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt)).astype(float)
            curr_frame_cp = cp.array(curr_frame)
            prev_frame_cp = cp.array(prev_frame)
            fr_mean = curr_frame_cp/2.0 + prev_frame_cp/2.0
            image_nsad[j-1] =  cp.asnumpy(cp.mean(cp.abs(curr_frame_cp-prev_frame_cp))/(cp.mean(fr_mean)-cp.amin(fr_mean)))
            image_ncc[j-1] = Two_Image_NCC_SNR(curr_frame, prev_frame)[0]
            image_mi[j-1] = cp.asnumpy(mutual_information_2d_cp(prev_frame_cp.ravel(), curr_frame_cp.ravel(), sigma=1.0, bin=2048, normalized=True))
            prev_frame = curr_frame.copy()
            del curr_frame_cp, prev_frame_cp
        mrc_obj.close()
    
    nsads = [np.mean(image_nsad), np.median(image_nsad), np.std(image_nsad)] 
    #image_ncc = image_ncc[1:-1]
    nccs = [np.mean(image_ncc), np.median(image_ncc), np.std(image_ncc)]
    nmis = [np.mean(image_mi), np.median(image_mi), np.std(image_mi)]

    fs=12
    lwl=1
    fig, axs = subplots(2,2, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.92, wspace=0.18, hspace=0.04)
    axs[0,0].axis(False)

    axs[1,0].plot(image_nsad, 'r', linewidth=lwl)
    axs[1,0].set_ylabel('Normalized Sum of Abs. Diff')
    axs[1,0].text(0.02, 0.04, 'NSAD mean = {:.3f}   NSAD median = {:.3f}  NSAD STD = {:.3f}'.format(nsads[0], nsads[1], nsads[2]), transform=axs[1,0].transAxes, fontsize = fs-1)
    axs[1,0].set_xlabel('Frame #')

    axs[0,1].plot(image_ncc, 'b', linewidth=lwl)
    axs[0,1].set_ylabel('Normalized Cross-Correlation')
    axs[0,1].grid(True)
    axs[0,1].text(0.02, 0.04, 'NCC mean = {:.3f}   NCC median = {:.3f}  NCC STD = {:.3f}'.format(nccs[0], nccs[1], nccs[2]), transform=axs[0,1].transAxes, fontsize = fs-1)

    axs[1,1].plot(image_mi, 'g', linewidth=lwl)
    axs[1,1].set_ylabel('Normalized Mutual Information')
    axs[1,1].set_xlabel('Frame #')
    axs[1,1].grid(True)
    axs[1,1].text(0.02, 0.04, 'NMI mean = {:.3f}   NMI median = {:.3f}  NMI STD = {:.3f}'.format(nmis[0], nmis[1], nmis[2]), transform=axs[1,1].transAxes, fontsize = fs-1)

    for ax in axs.ravel():
        ax.grid(True)

    # show three frames with eval box
    axs_fr0 = fig.add_subplot(6,2,1)
    axs_fr1 = fig.add_subplot(6,2,3)
    axs_fr2 = fig.add_subplot(6,2,5)
    axs_frms = [axs_fr0, axs_fr1, axs_fr2]
    frame_inds = [frame_inds[nf//10],  frame_inds[nf//2], frame_inds[nf//10*9]]
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r')
    for fr, ax in zip(frame_inds, axs_frms):
        eval_frame = (mrc_obj.data[fr, :, :].astype(dt)).astype(float)
        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval*(fr-frame_inds[0])//nf
            yi_eval = start_evaluation_box[0] + dy_eval*(fr-frame_inds[0])//nf
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny
        dmin, dmax = get_min_max_thresholds(eval_frame[yi_eval:ya_eval, xi_eval:xa_eval], 1e-3, 1e-3, 256, False)
        if invert_data:
            ax.imshow(eval_frame, cmap='Greys_r', vmin=dmin, vmax=dmax)
        else:
            ax.imshow(eval_frame, cmap='Greys', vmin=dmin, vmax=dmax)

        ax.text(0.03, 0.75, Sample_ID +'  frame={:d}'.format(fr), color='cyan', transform=ax.transAxes)
        rect_patch = patches.Rectangle((xi_eval,yi_eval),abs(xa_eval-xi_eval)-2,abs(ya_eval-yi_eval)-2, linewidth=0.5, edgecolor='yellow',facecolor='none')
        ax.add_patch(rect_patch)
        ax.axis('off')
    mrc_obj.close()

    fig.suptitle(mrc_filename, fontsize = fs-4)
    if save_res_png :
        fig.savefig(os.path.splitext(save_filename)[0] +'_RegistrationQuality.png', dpi=300)

    registration_summary_fnm = os.path.splitext(save_filename)[0] + '_RegistrationQuality.csv'
    columns=['Image NSAD', 'Image NCC', 'Image MI']
    reg_summary = pd.DataFrame(np.vstack((image_nsad, image_ncc, image_mi)).T, columns = columns, index = None)
    reg_summary.to_csv(registration_summary_fnm, index = None)
    
    return reg_summary


def show_eval_box_mrc_stack(mrc_filename, **kwargs):
    '''
    Read MRC stack and display the eval box for each frame from the list.
    ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

    Parameters
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed
     
    kwargs:
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    ax : matplotlib ax artist
        if provided, the data is exported to external ax object.
    frame_inds : array
        List of frame indices to display the evaluation box. If not provided, three frames will be used:
        [nz//10,  nz//2, nz//10*9] where nz is number of frames in mrc stack
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    sliding_evaluation_box : boolean
        if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above
    '''
    Sample_ID = kwargs.get("Sample_ID", '')
    save_res_png  = kwargs.get("save_res_png", True )
    save_filename = kwargs.get("save_filename", mrc_filename )
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    invert_data =  kwargs.get("invert_data", False)
    ax = kwargs.get("ax", '')
    plot_internal = (ax == '')

    mrc = mrcfile.mmap(mrc_filename, mode='r')
    header = mrc.header
    nx, ny, nz = int32(header['nx']), int32(header['ny']), int32(header['nz'])
    try: 
        dmin = float(header['dmin'])
        dmax = float(header['dmax'])
        if (dmin >= 0 and dmax<=255):
            dt = uint8
        else:
            dt = int16
    except:
        dt = int16

    frame_inds = kwargs.get("frame_inds", [nz//10,  nz//2, nz//10*9] )
    
    xi_eval = evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = nx
    yi_eval = evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ny

    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0

    for fr_ind in frame_inds: 
        eval_frame = (mrc.data[fr_ind, :, :].astype(dt)).astype(float)

        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval*fr_ind//nz
            yi_eval = start_evaluation_box[0] + dy_eval*fr_ind//nz
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny

        if plot_internal:
            fig, ax = subplots(1,1, figsize = (10.0, 11.0*ny/nx))
        dmin, dmax = get_min_max_thresholds(eval_frame[yi_eval:ya_eval, xi_eval:xa_eval], 1e-3, 1e-3, 256, False)
        if invert_data:
            ax.imshow(eval_frame, cmap='Greys_r', vmin=dmin, vmax=dmax)
        else:
            ax.imshow(eval_frame, cmap='Greys', vmin=dmin, vmax=dmax)
        ax.grid(True, color = "cyan")
        ax.set_title(Sample_ID + ' '+mrc_filename +',  frame={:d}'.format(fr_ind))
        rect_patch = patches.Rectangle((xi_eval,yi_eval),abs(xa_eval-xi_eval)-2,abs(ya_eval-yi_eval)-2, linewidth=1.0, edgecolor='yellow',facecolor='none')
        ax.add_patch(rect_patch)
        if save_res_png  and plot_internal:
            fname = os.path.splitext(save_filename)[0] + '_frame_{:d}_evaluation_box.png'.format(fr_ind)
            fig.savefig(fname, dpi=300)

    mrc.close()



##########################################
#         helper functions for results presentation
##########################################
def plot_registrtion_quality_csvs(data_files, labels, **kwargs):
    '''
    Read and plot together multiple registration quality summaries.
    ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

    Parameters:
    data_files : array of str
        Filenames (full paths) of the registration summaries (*.csv files)
    labels : array of str
        Labels (for each registration)

    kwargs:
    save_res_png : boolean
        If True, the PNG's of summary plots as well as summary Excel notebook are saved 
    save_filename : str
        Filename (full path) to save the results (default is data_dir +'Regstration_Summary.png')
    nsad_bounds : list of floats
        Bounds for NSAD plot (default is determined by get_min_max_thresholds with thresholds of 1e-4)
    ncc_bounds : list of floats
        Bounds for NCC plot (default is determined by get_min_max_thresholds with thresholds of 1e-4)
    nmi_bounds : list of floats
        Bounds for NMI plot (default is determined by get_min_max_thresholds with thresholds of 1e-4)
    colors : array or list of colors
        Optional colors for each plot/file. If not provided, will be auto-generated.
    linewidths : array of float
        linewidths for individual files. If not provided, all linewidts are set to 0.5

    Returns
    xlsx_fname : str
        Filename of the summary Excel notebook
    '''
    save_res_png  = kwargs.get("save_res_png", True )
    linewidths = kwargs.get("linewidths", np.ones(len(data_files))*0.5)
    data_dir = os.path.split(data_files[0])[0]
    default_save_filename = os.path.join(data_dir, 'Regstration_Summary.png')
    save_filename = kwargs.get("save_filename", default_save_filename)
    nsad_bounds = kwargs.get("nsad_bounds", [0.0, 0.0])
    ncc_bounds = kwargs.get("ncc_bounds", [0.0, 0.0])
    nmi_bounds = kwargs.get("nmi_bounds", [0.0, 0.0])
    
    nfls = len(data_files)
    reg_datas = []
    for df in data_files:
        fl = os.path.join(data_dir, df)
        data = pd.read_csv(fl)
        reg_datas.append(data)

    lw0 = 0.5
    lw1 = 1
    
    fs=12
    fs2=10
    fig1, axs1 = subplots(3,1, figsize=(7, 11), sharex=True)
    fig1.subplots_adjust(left=0.1, bottom=0.05, right=0.99, top=0.96, wspace=0.2, hspace=0.1)

    ax_nsad = axs1[0]
    ax_ncc = axs1[1]
    ax_nmi = axs1[2]
    ax_nsad.set_ylabel('Normalized Sum of Abs. Differences', fontsize=fs)
    ax_ncc.set_ylabel('Normalized Cross-Correlation', fontsize=fs)
    ax_nmi.set_ylabel('Normalized Mutual Information', fontsize=fs)
    ax_nmi.set_xlabel('Frame', fontsize=fs)

    spreads=[]
    my_cols = [get_cmap("gist_rainbow_r")((nfls-j)/(nfls)) for j in np.arange(nfls)]
    my_cols[0] = 'grey'
    my_cols[-1] = 'red'
    my_cols = kwargs.get("colors", my_cols)

    means = []
    image_nsads= []
    image_nccs= []
    image_snrs= []
    image_nmis = []
    for j, reg_data in enumerate(tqdm(reg_datas, desc='generating the registration quality summary plots')):
        #my_col = get_cmap("gist_rainbow_r")((nfls-j)/(nfls))
        #my_cols.append(my_col)
        my_col = my_cols[j]
        pf = labels[j]
        lw0 = linewidths[j]
        image_nsad = np.array(reg_data['Image NSAD'])
        image_nsads.append(image_nsad)
        image_ncc = np.array(reg_data['Image NCC'])
        image_nccs.append(image_ncc)
        image_snr = image_ncc/(1.0-image_ncc)
        image_snrs.append(image_snr)
        image_nmi = np.array(reg_data['Image MI'])
        image_nmis.append(image_nmi)

        metrics = [image_nsad, image_ncc, image_snr, image_nmi]
        spreads.append([get_spread(metr) for metr in metrics])
        means.append([np.mean(metr) for metr in metrics])

        ax_nsad.plot(image_nsad, c=my_col, linewidth=lw0)
        ax_nsad.plot(image_nsad[0], c=my_col, linewidth=lw1, label=pf)
        ax_ncc.plot(image_ncc, c=my_col, linewidth=lw0)
        ax_ncc.plot(image_ncc[0], c=my_col, linewidth=lw1, label=pf)
        ax_nmi.plot(image_nmi, c=my_col, linewidth=lw0)
        ax_nmi.plot(image_nmi[0], c=my_col, linewidth=lw1, label=pf)

    for ax in axs1.ravel():
        ax.grid(True)
        ax.legend(fontsize=fs2)
        
    if nsad_bounds[0]==nsad_bounds[1]:
        nsad_min, nsad_max = get_min_max_thresholds(np.concatenate(image_nsads),
                                                    thr_min=1e-4, thr_max=1e-4,
                                                    nbins=256, disp_res=False)
    else:
        nsad_min, nsad_max = nsad_bounds
    ax_nsad.set_ylim(nsad_min, nsad_max)

    if ncc_bounds[0]==ncc_bounds[1]:
        ncc_min, ncc_max = get_min_max_thresholds(np.concatenate(image_nccs),
                                                    thr_min=1e-4, thr_max=1e-4,
                                                    nbins=256, disp_res=False)
    else:
        ncc_min, ncc_max = ncc_bounds
    ax_ncc.set_ylim(ncc_min, ncc_max)

    if nmi_bounds[0]==nmi_bounds[1]:
        nmi_min, nmi_max = get_min_max_thresholds(np.concatenate(image_nmis),
                                                    thr_min=1e-4, thr_max=1e-4,
                                                    nbins=256, disp_res=False)
    else:
        nmi_min, nmi_max = nmi_bounds
    ax_nmi.set_ylim(nmi_min, nmi_max)    
    
    ax_nsad.text(-0.05, 1.05, data_dir, transform=ax_nsad.transAxes, fontsize=10)
    if save_res_png:
        fig1.savefig(save_filename, dpi=300)
        
    # Generate the Cell Text
    cell_text = []
    fig2_data = []
    limits = []
    rows = labels
    fst=9

    for j, (mean, spread) in enumerate(zip(means, spreads)):
        cell_text.append(['{:.4f}'.format(mean[0]), '{:.4f}'.format(spread[0]),
                          '{:.4f}'.format(mean[1]), '{:.4f}'.format(spread[1]), '{:.4f}'.format(mean[2]),
                          '{:.4f}'.format(mean[3]), '{:.4f}'.format(spread[3])])
        fig2_data.append([mean[0], spread[0], mean[1], spread[1], mean[2], mean[3], spread[3]])
        
    # Generate the table
    fig2, ax = subplots(1, 1, figsize=(9.5,1.3))
    fig2.subplots_adjust(left=0.32, bottom=0.01, right=0.98, top=0.86, wspace=0.05, hspace=0.05)
    ax.axis(False)
    ax.text(-0.30, 1.07, 'SIFT Registration Comparisons:  ' + data_dir, fontsize=fst)
    llw1=0.3
    clw = [llw1, llw1]

    columns = ['NSAD Mean', 'NSAD Spread', 'NCC Mean', 'NCC Spread', 'Mean SNR', 'NMI Mean', 'NMI Spread']

    n_cols = len(columns)
    n_rows = len(rows)

    tbl = ax.table(cellText = cell_text,
                   rowLabels = rows,
                   colLabels = columns,
                   cellLoc = 'center',
                   colLoc = 'center',
                   bbox = [0.01, 0, 0.995, 1.0],
                  fontsize=16)
    tbl.auto_set_column_width(col=3)

    table_props = tbl.properties()
    try:
        table_cells = table_props['child_artists']
    except:
        table_cells = table_props['children']

    tbl.auto_set_font_size(False)
    for j, cell in enumerate(table_cells[0:n_cols*n_rows]):
        cell.get_text().set_color(my_cols[j//n_cols])
        cell.get_text().set_fontsize(fst)
    for j, cell in enumerate(table_cells[n_cols*(n_rows+1):]):
        cell.get_text().set_color(my_cols[j])
    for cell in table_cells[n_cols*n_rows:]:
    #    cell.get_text().set_fontweight('bold')
        cell.get_text().set_fontsize(fst)
    save_filename2 = save_filename.replace('.png', '_table.png')
    if save_res_png:
        fig2.savefig(save_filename2, dpi=300)   
        
    ysize_fig = 4
    ysize_tbl = 0.25 * nfls
    fst3 = 8
    fig3, axs3 = subplots(2, 1, figsize=(7, ysize_fig+ysize_tbl),  gridspec_kw={"height_ratios" : [ysize_tbl, ysize_fig]})
    fig3.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.96, wspace=0.05, hspace=0.05)

    for j, reg_data in enumerate(reg_datas):
        my_col = my_cols[j]
        pf = labels[j]
        lw0 = linewidths[j]
        image_ncc = reg_data['Image NCC']
        axs3[1].plot(image_ncc, c=my_col, linewidth=lw0)
        axs3[1].plot(image_ncc[0], c=my_col, linewidth=lw1, label=pf)
    axs3[1].grid(True)
    axs3[1].legend(fontsize=fs2)
    axs3[1].set_ylabel('Normalized Cross-Correlation', fontsize=fs)
    axs3[1].set_xlabel('Frame', fontsize=fs)
    axs3[1].set_ylim(ncc_min, ncc_max)
    axs3[0].axis(False)
    axs3[0].text(-0.1, 1.07, 'SIFT Registration Comparisons:  ' + data_dir, fontsize=fst3)
    llw1=0.3
    clw = [llw1, llw1]

    columns2 = ['NCC Mean', 'NCC Spread', 'SNR Mean', 'SNR Spread']
    cell2_text = []
    fig3_data = []
    limits = []
    rows = labels
    
    for j, (mean, spread) in enumerate(zip(means, spreads)):
        cell2_text.append(['{:.4f}'.format(mean[1]), '{:.4f}'.format(spread[1]),
                          '{:.4f}'.format(mean[2]), '{:.4f}'.format(spread[2])])
    n_cols = len(columns2)
    n_rows = len(rows)

    tbl2 = axs3[0].table(cellText = cell2_text,
                   rowLabels = rows,
                   colLabels = columns2,
                   cellLoc = 'center',
                   colLoc = 'center',
                   bbox = [0.38, 0, 0.55, 1.0],  # (left, bottom, width, height)
                  fontsize=16)
    rl = max([len(pf) for pf in labels])
    #tbl2.auto_set_column_width(col=[rl+5, len(columns2[0]), len(columns2[1]), len(columns2[2]), len(columns2[3])])
    tbl2.auto_set_column_width(col=list(range(len(columns2)+1)))
    tbl2.auto_set_font_size(False)

    table2_props = tbl2.properties()
    try:
        table2_cells = table2_props['child_artists']
    except:
        table2_cells = table2_props['children']

    tbl.auto_set_font_size(False)
    for j, cell in enumerate(table2_cells[0:n_cols*n_rows]):
        cell.get_text().set_color(my_cols[j//n_cols])
        cell.get_text().set_fontsize(fst3)
    for j, cell in enumerate(table2_cells[n_cols*(n_rows+1):]):
        cell.get_text().set_color(my_cols[j])
    for cell in table2_cells[n_cols*n_rows:]:
    #    cell.get_text().set_fontweight('bold')
        cell.get_text().set_fontsize(fst3)
    save_filename3 = save_filename.replace('.png', '_fig_and_table.png')
    if save_res_png:
        fig3.savefig(save_filename3, dpi=300)
    
    if save_res_png:
        # Generate a single multi-page CSV file
        xlsx_fname = save_filename.replace('.png', '.xlsx')
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(xlsx_fname, engine='xlsxwriter')
        fig2_df = pd.DataFrame(fig2_data, columns=columns)
        fig2_df.insert(0, '', labels)
        fig2_df.insert(1, 'Path', data_files)
        fig2_df.to_excel(writer, index=None, sheet_name='Summary')
        for reg_data, label in zip(tqdm(reg_datas, desc='saving the data into xlsx file'), labels):
            data_fn = label[0:31]
            reg_data.to_excel(writer, sheet_name=data_fn)
        writer.save()
    return xlsx_fname


#######################################
#    class FIBSEM_frame
#######################################

class FIBSEM_frame:
    """
    A class representing single FIB-SEM data frame.
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    Contains the info/settings on a single FIB-SEM data frame and the procedures that can be performed on it.

    Attributes (only some more important are listed here)
    ----------
    fname : str
        filename of the individual data frame
    header : str
        1024 bytes - header
    FileVersion : int
        file version number
    ChanNum : int
        Number of channels
    EightBit : int
        8-bit data switch: 0 for 16-bit data, 1 for 8-bit data
    ScalingS : 2D array of floats
        scaling parameters allowing to convert I16 data into actual electron counts 
    Sample_ID : str
        Sample_ID
    Notes : str
        Experiment notes
    DetA : str
        Detector A name
    DetB : str
        Detector B name ('None' if there is no Detector B)
    XResolution : int
        number of pixels - frame size in horizontal direction
    YResolution : int
        number of pixels - frame size in vertical direction

    Methods
    -------
    print_header()
        Prints a formatted content of the file header

    display_images()
        Display auto-scaled detector images without saving the figure into the file.

    save_images_jpeg(**kwargs)
        Display auto-scaled detector images and save the figure into JPEG file (s).

    save_images_tif(images_to_save = 'Both')
        Save the detector images into TIF file (s).

    get_image_min_max(image_name = 'ImageA', thr_min = 1.0e-4, thr_max = 1.0e-3, nbins=256, disp_res = False)
        Calculates the data range of the EM data.

    RawImageA_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
        Convert the Image A into 8-bit array

    RawImageB_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
            Convert the Image B into 8-bit array

    save_snapshot(display = True, dpi=300, thr_min = 1.0e-3, thr_max = 1.0e-3, nbins=256):
        Builds an image that contains both the Detector A and Detector B (if present) images as well as a table with important FIB-SEM parameters.

    analyze_noise_ROIs(**kwargs):
        Analyses the noise statistics in the selected ROI's of the EM data. (Calls Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs):)

    analyze_noise_statistics(**kwargs):
        Analyses the noise statistics of the EM data image. (Calls Single_Image_Noise_Statistics(img, **kwargs):)

    analyze_SNR_autocorr(**kwargs):
        Estimates SNR using auto-correlation analysis of a single image. (Calls Single_Image_SNR(img, **kwargs):)

    show_eval_box(**kwargs):
        Show the box used for evaluating the noise

    determine_field_fattening_parameters(**kwargs):
        Perfrom 2D parabolic fit (calls Perform_2D_fit(Img, estimator, **kwargs)) and determine the field-flattening parameters

    flatten_image(**kwargs):
        Flatten the image
    """

    def __init__(self, fname, **kwargs):   
        self.fname = fname
        self.ftype = kwargs.get("ftype", 0) # ftype=0 - Shan Xu's binary format  ftype=1 - tif files
        self.use_dask_arrays = kwargs.get("use_dask_arrays", False)
        if self.ftype == 1:
            self.RawImageA = tiff.imread(fname)

    # for tif files
        if self.ftype == 1:
            self.FileVersion = -1
            self.DetA = 'Detector A'     # Name of detector A
            self.DetB = 'None'     # Name of detector B
            try:
                with PILImage.open(self.fname) as img:
                    tif_header = {TAGS[key] : img.tag[key] for key in img.tag_v2}
                    self.header = tif_header
                try:
                    if tif_header['BitsPerSample'][0]==8:
                        self.EightBit = 1
                    else:
                        self.EightBit = 0
                except:
                    self.EightBit = int(type(self.RawImageA[0,0])==np.uint8)
            except:
                self.header = ''
                self.EightBit = int(type(self.RawImageA[0,0])==np.uint8)

            self.PixelSize = kwargs.get("PixelSize", 8.0)
            self.Sample_ID = kwargs.get("Sample_ID", '')
            self.YResolution, self.XResolution = self.RawImageA.shape

    # for Shan Xu's data files 
        if self.ftype == 0:
            fid = open(self.fname, "rb")
            fid.seek(0, 0)
            self.header = fid.read(1024) # Read in self.header
            self.FileMagicNum = unpack('>L',self.header[0:4])[0]                       # Read in magic number, should be 3555587570
            self.FileVersion = unpack('>h',self.header[4:6])[0]                        # Read in file version number
            self.FileType = unpack('>h',self.header[6:8])[0]                           # Read in file type, 1 is Zeiss Neon detectors
            self.SWdate = (unpack('10s',self.header[8:18])[0]).decode("utf-8")         # Read in SW date
            self.TimeStep = unpack('>d',self.header[24:32])[0]                         # Read in AI sampling time (including oversampling) in seconds
            self.ChanNum = unpack('b',self.header[32:33])[0]                           # Read in number of channels
            self.EightBit = unpack('b',self.header[33:34])[0]                          # Read in 8-bit data switch

            if self.FileVersion == 1:
                # Read in self.AI channel self.Scaling factors, (col#: self.AI#), (row#: offset, gain, 2nd order, 3rd order)
                self.ScalingS = unpack('>'+str(4*self.ChanNum)+'d',self.header[36:(36+self.ChanNum*32)])
            elif self.FileVersion == 2 or self.FileVersion == 3 or self.FileVersion == 4 or self.FileVersion == 5 or self.FileVersion == 6:
                self.ScalingS = unpack('>'+str(4*self.ChanNum)+'f',self.header[36:(36+self.ChanNum*16)])
            else:
                self.ScalingS = unpack('>8f',self.header[36:68])
                self.Scaling = transpose(np.asarray(self.ScalingS).reshape(2,4))
            
            if self.FileVersion > 8 :
                self.RestartFlag = unpack('b',self.header[68:69])[0]              # Read in restart flag
                self.StageMove = unpack('b',self.header[69:70])[0]                # Read in stage move flag
                self.FirstPixelX = unpack('>l',self.header[70:74])[0]              # Read in first pixel X coordinate (center = 0)
                self.FirstPixelY = unpack('>l',self.header[74:78])[0]              # Read in first pixel X coordinate (center = 0)
            
            self.XResolution = unpack('>L',self.header[100:104])[0]                # Read X resolution
            self.YResolution = unpack('>L',self.header[104:108])[0]                # Read Y resolution

            if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3:
                self.Oversampling = unpack('>B',self.header[108:109])[0]               # self.AI oversampling     
                self.AIDelay = unpack('>h',self.header[109:111])[0]                    # self.AI delay (# of samples)
            else:
                self.Oversampling = unpack('>H',self.header[108:110])[0]

            self.ZeissScanSpeed = unpack('b',self.header[111:112])[0] # Scan speed (Zeiss #)

            if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3:
                self.ScanRate = unpack('>d',self.header[112:120])[0]                   # Actual AO (scanning) rate    
                self.FramelineRampdownRatio = unpack('>d',self.header[120:128])[0]     # Frameline rampdown ratio
                self.Xmin = unpack('>d',self.header[128:136])[0]                       # X coil minimum voltage
                self.Xmax = unpack('>d',self.header[136:144])[0]                       # X coil maximum voltage
                self.Detmin = -10                                                      # Detector minimum voltage
                self.Detmax = 10                                                       # Detector maximum voltage
            else:
                self.ScanRate = unpack('>f',self.header[112:116])[0]                   # Actual AO (scanning) rate    
                self.FramelineRampdownRatio = unpack('>f',self.header[116:120])[0]     # Frameline rampdown ratio
                self.Xmin = unpack('>f',self.header[120:124])[0]                       # X coil minimum voltage
                self.Xmax = unpack('>f',self.header[124:128])[0]                       # X coil maximum voltage
                self.Detmin = unpack('>f',self.header[128:132])[0]                     # Detector minimum voltage
                self.Detmax = unpack('>f',self.header[132:136])[0]                     # Detector maximum voltage
                self.DecimatingFactor = unpack('>H',self.header[136:138])[0]           # Decimating factor

            self.AI1 = unpack('b',self.header[151:152])[0]                              # self.AI Ch1
            self.AI2 = unpack('b',self.header[152:153])[0]                              # self.AI Ch2 
            self.AI3 = unpack('b',self.header[153:154])[0]                              # self.AI Ch3
            self.AI4 = unpack('b',self.header[154:155])[0]                              # self.AI Ch4
            
            self.Notes = (unpack('200s',self.header[180:380])[0]).decode("utf-8")       # Read in notes

            if self.FileVersion > 8 :
                self.Sample_ID = (unpack('25s',self.header[155:180])[0]).decode("utf-8").strip('\x00') # Read in Sample ID
            else:
                self.Sample_ID = self.Notes.split(',')[0].strip('\x00')

            if self.FileVersion == 1 or self.FileVersion == 2:
                self.DetA = (unpack('10s',self.header[380:390])[0]).decode("utf-8")     # Name of detector A
                self.DetB = (unpack('18s',self.header[390:408])[0]).decode("utf-8")     # Name of detector B
                self.DetC = (unpack('20s',self.header[700:720])[0]).decode("utf-8")     # Name of detector C
                self.DetD = (unpack('20s',self.header[720:740])[0]).decode("utf-8")     # Name of detector D
                self.Mag = unpack('>d',self.header[408:416])[0]                         # Magnification
                self.PixelSize = unpack('>d',self.header[416:424])[0]                   # Pixel size in nm
                self.WD = unpack('>d',self.header[424:432])[0]                          # Working distance in mm
                self.EHT = unpack('>d',self.header[432:440])[0]                         # EHT in kV
                self.SEMApr = unpack('b',self.header[440:441])[0]                       # SEM aperture number
                self.HighCurrent = unpack('b',self.header[441:442])[0]                  # high current mode (1=on, 0=off)
                self.SEMCurr = unpack('>d',self.header[448:456])[0]                     # SEM probe current in A
                self.SEMRot = unpack('>d',self.header[456:464])[0]                      # SEM scan roation in degree
                self.ChamVac = unpack('>d',self.header[464:472])[0]                     # Chamber vacuum
                self.GunVac = unpack('>d',self.header[472:480])[0]                      # E-gun vacuum
                self.SEMStiX = unpack('>d',self.header[480:488])[0]                     # SEM stigmation X
                self.SEMStiY = unpack('>d',self.header[488:496])[0]                     # SEM stigmation Y
                self.SEMAlnX = unpack('>d',self.header[496:504])[0]                     # SEM aperture alignment X
                self.SEMAlnY = unpack('>d',self.header[504:512])[0]                     # SEM aperture alignment Y
                self.StageX = unpack('>d',self.header[512:520])[0]                      # Stage position X in mm
                self.StageY = unpack('>d',self.header[520:528])[0]                      # Stage position Y in mm
                self.StageZ = unpack('>d',self.header[528:536])[0]                      # Stage position Z in mm
                self.StageT = unpack('>d',self.header[536:544])[0]                      # Stage position T in degree
                self.StageR = unpack('>d',self.header[544:552])[0]                      # Stage position R in degree
                self.StageM = unpack('>d',self.header[552:560])[0]                      # Stage position M in mm
                self.BrightnessA = unpack('>d',self.header[560:568])[0]                 # Detector A brightness (%)
                self.ContrastA = unpack('>d',self.header[568:576])[0]                   # Detector A contrast (%)
                self.BrightnessB = unpack('>d',self.header[576:584])[0]                 # Detector B brightness (%)
                self.ContrastB = unpack('>d',self.header[584:592])[0]                   # Detector B contrast (%)
                self.Mode = unpack('b',self.header[600:601])[0]                         # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
                self.FIBFocus = unpack('>d',self.header[608:616])[0]                    # FIB focus in kV
                self.FIBProb = unpack('b',self.header[616:617])[0]                      # FIB probe number
                self.FIBCurr = unpack('>d',self.header[624:632])[0]                     # FIB emission current
                self.FIBRot = unpack('>d',self.header[632:640])[0]                      # FIB scan rotation
                self.FIBAlnX = unpack('>d',self.header[640:648])[0]                     # FIB aperture alignment X
                self.FIBAlnY = unpack('>d',self.header[648:656])[0]                     # FIB aperture alignment Y
                self.FIBStiX = unpack('>d',self.header[656:664])[0]                     # FIB stigmation X
                self.FIBStiY = unpack('>d',self.header[664:672])[0]                     # FIB stigmation Y
                self.FIBShiftX = unpack('>d',self.header[672:680])[0]                   # FIB beam shift X in micron
                self.FIBShiftY = unpack('>d',self.header[680:688])[0]                   # FIB beam shift Y in micron
            else:
                self.DetA = (unpack('10s',self.header[380:390])[0]).decode("utf-8")     # Name of detector A
                self.DetB = (unpack('18s',self.header[390:408])[0]).decode("utf-8")     # Name of detector B
                self.DetC = (unpack('20s',self.header[410:430])[0]).decode("utf-8")     # Name of detector C
                self.DetD = (unpack('20s',self.header[430:450])[0]).decode("utf-8")     # Name of detector D
                self.Mag = unpack('>f',self.header[460:464])[0]                         # Magnification
                self.PixelSize = unpack('>f',self.header[464:468])[0]                   # Pixel size in nm
                self.WD = unpack('>f',self.header[468:472])[0]                          # Working distance in mm
                self.EHT = unpack('>f',self.header[472:476])[0]                         # EHT in kV
                self.SEMApr = unpack('b',self.header[480:481])[0]                       # SEM aperture number
                self.HighCurrent = unpack('b',self.header[481:482])[0]                  # high current mode (1=on, 0=off)
                self.SEMCurr = unpack('>f',self.header[490:494])[0]                     # SEM probe current in A
                self.SEMRot = unpack('>f',self.header[494:498])[0]                      # SEM scan roation in degree
                self.ChamVac = unpack('>f',self.header[498:502])[0]                     # Chamber vacuum
                self.GunVac = unpack('>f',self.header[502:506])[0]                      # E-gun vacuum
                self.SEMShiftX = unpack('>f',self.header[510:514])[0]                   # SEM beam shift X
                self.SEMShiftY = unpack('>f',self.header[514:518])[0]                   # SEM beam shift Y
                self.SEMStiX = unpack('>f',self.header[518:522])[0]                     # SEM stigmation X
                self.SEMStiY = unpack('>f',self.header[522:526])[0]                     # SEM stigmation Y
                self.SEMAlnX = unpack('>f',self.header[526:530])[0]                     # SEM aperture alignment X
                self.SEMAlnY = unpack('>f',self.header[530:534])[0]                     # SEM aperture alignment Y
                self.StageX = unpack('>f',self.header[534:538])[0]                      # Stage position X in mm
                self.StageY = unpack('>f',self.header[538:542])[0]                      # Stage position Y in mm
                self.StageZ = unpack('>f',self.header[542:546])[0]                      # Stage position Z in mm
                self.StageT = unpack('>f',self.header[546:550])[0]                      # Stage position T in degree
                self.StageR = unpack('>f',self.header[550:554])[0]                      # Stage position R in degree
                self.StageM = unpack('>f',self.header[554:558])[0]                      # Stage position M in mm
                self.BrightnessA = unpack('>f',self.header[560:564])[0]                 # Detector A brightness (%)
                self.ContrastA = unpack('>f',self.header[564:568])[0]                   # Detector A contrast (%)
                self.BrightnessB = unpack('>f',self.header[568:572])[0]                 # Detector B brightness (%)
                self.ContrastB = unpack('>f',self.header[572:576])[0]                   # Detector B contrast (%)
                self.Mode = unpack('b',self.header[600:601])[0]                         # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
                self.FIBFocus = unpack('>f',self.header[604:608])[0]                    # FIB focus in kV
                self.FIBProb = unpack('b',self.header[608:609])[0]                      # FIB probe number
                self.FIBCurr = unpack('>f',self.header[620:624])[0]                     # FIB emission current
                self.FIBRot = unpack('>f',self.header[624:628])[0]                      # FIB scan rotation
                self.FIBAlnX = unpack('>f',self.header[628:632])[0]                     # FIB aperture alignment X
                self.FIBAlnY = unpack('>f',self.header[632:636])[0]                     # FIB aperture alignment Y
                self.FIBStiX = unpack('>f',self.header[636:640])[0]                     # FIB stigmation X
                self.FIBStiY = unpack('>f',self.header[640:644])[0]                     # FIB stigmation Y
                self.FIBShiftX = unpack('>f',self.header[644:648])[0]                   # FIB beam shift X in micron
                self.FIBShiftY = unpack('>f',self.header[648:652])[0]                   # FIB beam shift Y in micron

            if self.FileVersion > 4:
                self.MillingXResolution = unpack('>L',self.header[652:656])[0]                       # FIB milling X resolution
                self.MillingYResolution = unpack('>L',self.header[656:660])[0]                       # FIB milling Y resolution
                self.MillingXSize = unpack('>f',self.header[660:664])[0]                             # FIB milling X size (um)
                self.MillingYSize = unpack('>f',self.header[664:668])[0]                             # FIB milling Y size (um)
                self.MillingULAng = unpack('>f',self.header[668:672])[0]                             # FIB milling upper left inner angle (deg)
                self.MillingURAng = unpack('>f',self.header[672:676])[0]                             # FIB milling upper right inner angle (deg)
                self.MillingLineTime = unpack('>f',self.header[676:680])[0]                          # FIB line milling time (s)
                self.FIBFOV = unpack('>f',self.header[680:684])[0]                                   # FIB FOV (um)
                self.MillingLinesPerImage = unpack('>H',self.header[684:686])[0]                     # FIB milling lines per image
                self.MillingPIDOn = unpack('>b',self.header[686:687])[0]                             # FIB milling PID on
                self.MillingPIDMeasured = unpack('>b',self.header[689:690])[0]                       # FIB milling PID measured (0:specimen, 1:beamdump)
                self.MillingPIDTarget = unpack('>f',self.header[690:694])[0]                         # FIB milling PID target
                self.MillingPIDTargetSlope = unpack('>f',self.header[694:698])[0]                    # FIB milling PID target slope
                self.MillingPIDP = unpack('>f',self.header[698:702])[0]                              # FIB milling PID P
                self.MillingPIDI = unpack('>f',self.header[702:706])[0]                              # FIB milling PID I
                self.MillingPIDD = unpack('>f',self.header[706:710])[0]                              # FIB milling PID D
                self.MachineID = (unpack('30s',self.header[800:830])[0]).decode("utf-8")             # Machine ID
                self.SEMSpecimenI = unpack('>f',self.header[672:676])[0]                             # SEM specimen current (nA)

            if self.FileVersion > 5 :
                self.Temperature = unpack('>f',self.header[850:854])[0]                              # Temperature (F)
                self.FaradayCupI = unpack('>f',self.header[854:858])[0]                              # Faraday cup current (nA)
                self.FIBSpecimenI = unpack('>f',self.header[858:862])[0]                             # FIB specimen current (nA)
                self.BeamDump1I = unpack('>f',self.header[862:866])[0]                               # Beam dump 1 current (nA)
                self.SEMSpecimenI = unpack('>f',self.header[866:870])[0]                             # SEM specimen current (nA)
                self.MillingYVoltage = unpack('>f',self.header[870:874])[0]                          # Milling Y voltage (V)
                self.FocusIndex = unpack('>f',self.header[874:878])[0]                               # Focus index
                self.FIBSliceNum = unpack('>L',self.header[878:882])[0]                              # FIB slice #

            if self.FileVersion > 7:
                self.BeamDump2I = unpack('>f',self.header[882:886])[0]                              # Beam dump 2 current (nA)
                self.MillingI = unpack('>f',self.header[886:890])[0]                                # Milling current (nA)

            self.FileLength = unpack('>q',self.header[1000:1008])[0]                                # Read in file length in bytes 

#                Finish self.header read
#
#                Read raw data
#                fid.seek(1024, 0)
#                n_elements = self.ChanNum * self.XResolution * self.YResolution
#                print(n_elements, self.ChanNum, self.XResolution, self.YResolution)
#                if self.EightBit==1:
#                    raw_data = fid.read(n_elements) # Read in data
#                    Raw = unpack('>'+str(n_elements)+'B',raw_data)
#                else:
#                    #raw_data = fid.read(2*n_elements) # Read in data
#                    #Raw = unpack('>'+str(n_elements)+'h',raw_data)
#                fid.close
#                finish reading raw data

            n_elements = self.ChanNum * self.XResolution * self.YResolution
            fid.seek(1024, 0)
            if self.EightBit==1:
                dt = np.dtype(np.uint8)
                dt = dt.newbyteorder('>')
                if self.use_dask_arrays:
                    Raw = da.from_array(np.frombuffer(fid.read(n_elements), dtype=dt))
                else:
                    Raw = np.frombuffer(fid.read(n_elements), dtype=dt)
            else:
                dt = np.dtype(np.int16)
                dt = dt.newbyteorder('>')
                if self.use_dask_arrays:
                    Raw = da.from_array(np.frombuffer(fid.read(2*n_elements),dtype=dt))
                else:
                    Raw = np.frombuffer(fid.read(2*n_elements),dtype=dt)
            fid.close
            # finish reading raw data
     
            Raw = np.asarray(Raw).reshape(self.YResolution, self.XResolution, self.ChanNum)
            #print(shape(Raw), type(Raw), type(Raw[0,0]))

            #data = np.asarray(datab).reshape(self.YResolution,self.XResolution,ChanNum)
            if self.EightBit == 1:
                if self.AI1 == 1:
                    self.RawImageA = Raw[:,:,0]
                    self.ImageA = (Raw[:,:,0].astype(float32)*self.ScanRate/self.Scaling[0,0]/self.Scaling[2,0]/self.Scaling[3,0]+self.Scaling[1,0]).astype(int32)
                    if self.AI2 == 1:
                        self.RawImageB = Raw[:,:,1]
                        self.ImageB = (Raw[:,:,1].astype(float32)*self.ScanRate/self.Scaling[0,1]/self.Scaling[2,1]/self.Scaling[3,1]+self.Scaling[1,1]).astype(int32)
                elif self.AI2 == 1:
                    self.RawImageB = Raw[:,:,0]
                    self.ImageB = (Raw[:,:,0].astype(float32)*self.ScanRate/self.Scaling[0,0]/self.Scaling[2,0]/self.Scaling[3,0]+self.Scaling[1,0]).astype(int32)
            else:
                if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3 or self.FileVersion == 4 or self.FileVersion == 5 or self.FileVersion == 6:
                    if self.AI1 == 1:
                        self.RawImageA = Raw[:,:,0]
                        self.ImageA = self.Scaling[0,0] + self.RawImageA * self.Scaling[1,0]  # Converts raw I16 data to voltage based on self.Scaling factors
                        if self.AI2 == 1:
                            self.RawImageB = Raw[:,:,1]
                            self.ImageB = self.Scaling[0,1] + self.RawImageB * self.Scaling[1,1]
                            if self.AI3 == 1:
                                self.RawImageC = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                self.ImageC = self.Scaling[0,2] + self.RawImageC * self.Scaling[1,2]
                                if self.AI4 == 1:
                                    self.RawImageD = (Raw[:,:,3]).reshape(self.YResolution,self.XResolution)
                                    self.ImageD = self.Scaling[0,3] + self.RawImageD * self.Scaling[1,3]
                            elif self.AI4 == 1:
                                self.RawImageD = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                self.ImageD = self.Scaling[0,2] + self.RawImageD * self.Scaling[1,2]
                        elif self.AI3 == 1:
                            self.RawImageC = Raw[:,:,1]
                            self.ImageC = self.Scaling[0,1] + self.RawImageC * self.Scaling[1,1]
                            if self.AI4 == 1:
                                self.RawImageD = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                self.ImageD = self.Scaling[0,2] + self.RawImageD * self.Scaling[1,2]
                        elif self.AI4 == 1:
                            self.RawImageD = Raw[:,:,1]
                            self.ImageD = self.Scaling[0,1] + self.RawImageD * self.Scaling[1,1]
                    elif self.AI2 == 1:
                        self.RawImageB = Raw[:,:,0]
                        self.ImageB = self.Scaling[0,0] + self.RawImageB * self.Scaling[1,0]
                        if self.AI3 == 1:
                            self.RawImageC = Raw[:,:,1]
                            self.ImageC = self.Scaling[0,1] + self.RawImageC * self.Scaling[1,1]
                            if self.AI4 == 1:
                                self.RawImageD = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                self.ImageD = self.Scaling[0,2] + self.RawImageD * self.Scaling[1,2]
                        elif self.AI4 == 1:
                            self.RawImageD = Raw[:,:,1]
                            self.ImageD = self.Scaling[0,1] + self.RawImageD * self.Scaling[1,1]
                    elif self.AI3 == 1:
                        self.RawImageC = Raw[:,:,0]
                        self.ImageC = self.Scaling[0,0] + self.RawImageC * self.Scaling[1,0]
                        if self.AI4 == 1:
                            self.RawImageD = Raw[:,:,1]
                            self.ImageD = self.Scaling[0,1] + self.RawImageD * self.Scaling[1,1]
                    elif self.AI4 == 1:
                        self.RawImageD = Raw[:,:,0]
                        self.ImageD = self.Scaling[0,0] + self.RawImageD * self.Scaling[1,0]
                        
                elif self.FileVersion == 7:
                    if self.AI1 == 1:
                        self.RawImageA = Raw[:,:,0]
                        self.ImageA = (self.RawImageA - self.Scaling[1,0])*self.Scaling[2,0]
                        if self.AI2 == 1:
                            self.RawImageB = Raw[:,:,1]
                            self.ImageB = (self.RawImageB - self.Scaling[1,1])*self.Scaling[2,1]
                    elif self.AI2 == 1:
                        self.RawImageB = Raw[:,:,0]
                        self.ImageB = (self.RawImageB - self.Scaling[1,1])*self.Scaling[2,1]
                        
                elif  self.FileVersion == 8 or self.FileVersion == 9:
                    self.ElectronFactor1 = 0.1;             # 16-bit intensity is 10x electron counts
                    self.Scaling[3,0] = self.ElectronFactor1
                    self.ElectronFactor2 = 0.1;             # 16-bit intensity is 10x electron counts
                    self.Scaling[3,1] = self.ElectronFactor2
                    if self.AI1 == 1:
                        self.RawImageA = Raw[:,:,0]
                        self.ImageA = (self.RawImageA - self.Scaling[1,0]) * self.Scaling[2,0] / self.ScanRate * self.Scaling[0,0] / self.ElectronFactor1                        
                        # Converts raw I16 data to voltage based on self.Scaling factors
                        if self.AI2 == 1:
                            self.RawImageB = Raw[:,:,1]
                            self.ImageB = (self.RawImageB - self.Scaling[1,1]) * self.Scaling[2,1] / self.ScanRate * self.Scaling[0,1] / self.ElectronFactor2
                    elif self.AI2 == 1:
                        self.RawImageB = Raw[:,:,0]
                        self.ImageB = (self.RawImageB - self.Scaling[1,1]) * self.Scaling[2,1] / self.ScanRate * self.Scaling[0,1] / self.ElectronFactor2

    def print_header(self):
        '''
        Prints a formatted content of the file header

        '''
        if self.FileVersion == -1 :
            print('Sample_ID=', self.Sample_ID)
            print('DetA=', self.DetA)
            print('DetB=', self.DetB)
            print('EightBit=', self.EightBit)
            print('XResolution=', self.XResolution)
            print('YResolution=', self.YResolution)
            print('PixelSize=', self.PixelSize)
        else:
            print('FileMagicNum=', self.FileMagicNum)
            print('FileVersion=', self.FileVersion)
            print('FileType=', self.FileType)
            print('SWdate=', self.SWdate)
            print('TimeStep=', self.TimeStep)                
            print('ChanNum=', self.ChanNum)
            print('EightBit=', self.EightBit)
            print('Scaling=', self.Scaling)
            if self.FileVersion > 8 :
                print('RestartFlag=', self.RestartFlag)
                print('StageMove=', self.StageMove)
                print('FirstPixelX=', self.FirstPixelX)
                print('FirstPixelY=', self.FirstPixelY)
            print('XResolution=', self.XResolution)
            print('YResolution=', self.YResolution)
            if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3:
                print('AIDelay=', self.AIDelay)
            print('Oversampling=', self.Oversampling)
            print('ZeissScanSpeed=', self.ZeissScanSpeed)
            print('DecimatingFactor=', self.DecimatingFactor)
            print('ScanRate=', self.ScanRate)
            print('FramelineRampdownRatio=', self.FramelineRampdownRatio)
            print('Xmin=', self.Xmin)
            print('Xmax=', self.Xmax)
            print('Detmin=', self.Detmin)
            print('Detmax=', self.Detmax)
            print('AI1=', self.AI1)
            print('AI2=', self.AI2)
            print('AI3=', self.AI3)
            print('AI4=', self.AI4)
            if self.FileVersion > 8 :
                 print('Sample_ID=', self.Sample_ID)
            print('Notes=', self.Notes)
            print('SEMShiftX=', self.SEMShiftX)
            print('SEMShiftY=', self.SEMShiftY)
            print('DetA=', self.DetA)
            print('DetB=', self.DetB)
            print('DetC=', self.DetC)
            print('DetD=', self.DetD)
            print('Mag=', self.Mag)
            print('PixelSize=', self.PixelSize)
            print('WD=', self.WD)
            print('EHT=', self.EHT)
            print('SEMApr=', self.SEMApr)
            print('HighCurrent=', self.HighCurrent)
            print('SEMCurr=', self.SEMCurr)
            print('SEMRot=', self.SEMRot)
            print('ChamVac=', self.ChamVac)
            print('GunVac=', self.GunVac)
            print('SEMStiX=', self.SEMStiX)
            print('SEMStiY=', self.SEMStiY)
            print('SEMAlnX=', self.SEMAlnX)
            print('SEMAlnY=', self.SEMAlnY)
            print('StageX=', self.StageX)
            print('StageY=', self.StageY)
            print('StageZ=', self.StageZ)
            print('StageT=', self.StageT)
            print('StageR=', self.StageR)
            print('StageM=', self.StageM)
            print('BrightnessA=', self.BrightnessA)
            print('ContrastA=', self.ContrastA)
            print('BrightnessB=', self.BrightnessB)
            print('ContrastB=', self.ContrastB)
            print('Mode=', self.Mode)
            print('FIBFocus=', self.FIBFocus)
            print('FIBProb=', self.FIBProb)
            print('FIBCurr=', self.FIBCurr)
            print('FIBRot=', self.FIBRot)
            print('FIBAlnX=', self.FIBAlnX)
            print('FIBAlnY=', self.FIBAlnY)
            print('FIBStiX=', self.FIBStiX)
            print('FIBStiY=', self.FIBStiY)
            print('FIBShiftX=', self.FIBShiftX)
            print('FIBShiftY=', self.FIBShiftY)
            if self.FileVersion > 4:
                print('MillingXResolution=', self.MillingXResolution)
                print('MillingYResolution=', self.MillingYResolution)
                print('MillingXSize=', self.MillingXSize)
                print('MillingYSize=', self.MillingYSize)
                print('MillingULAng=', self.MillingULAng)
                print('MillingURAng=', self.MillingURAng)
                print('MillingLineTime=', self.MillingLineTime)
                print('FIBFOV (um)=', self.FIBFOV)
                print('MillingPIDOn=', self.MillingPIDOn)
                print('MillingPIDMeasured=', self.MillingPIDMeasured)
                print('MillingPIDTarget=', self.MillingPIDTarget)
                print('MillingPIDTargetSlope=', self.MillingPIDTargetSlope)
                print('MillingPIDP=', self.MillingPIDP)
                print('MillingPIDI=', self.MillingPIDI)
                print('MillingPIDD=', self.MillingPIDD)
                print('MachineID=', self.MachineID)
                print('SEMSpecimenI=', self.SEMSpecimenI)
            if self.FileVersion > 5:
                print('Temperature=', self.Temperature)
                print('FaradayCupI=', self.FaradayCupI)
                print('FIBSpecimenI=', self.FIBSpecimenI)
                print('BeamDump1I=', self.BeamDump1I)
                print('MillingYVoltage=', self.MillingYVoltage)
                print('FocusIndex=', self.FocusIndex)
                print('FIBSliceNum=', self.FIBSliceNum)
            if self.FileVersion > 7:
                print('BeamDump2I=', self.BeamDump2I)
                print('MillingI=', self.MillingI)
            print('SEMSpecimenI=', self.SEMSpecimenI)   
            print('FileLength=', self.FileLength)
    
    def display_images(self):
        '''
        Display auto-scaled detector images without saving the figure into the file.

        '''
        fig, axs = subplots(2, 1, figsize=(10,5))
        axs[0].imshow(self.RawImageA, cmap='Greys')
        axs[1].imshow(self.RawImageB, cmap='Greys')
        ttls = ['Detector A: '+self.DetA.strip('\x00'), 'Detector B: '+self.DetB.strip('\x00')]
        for ax, ttl in zip(axs, ttls):
            ax.axis(False)
            ax.set_title(ttl, fontsize=10)
        fig.suptitle(self.fname)
        
    def save_images_jpeg(self, **kwargs):
        '''
        Display auto-scaled detector images and save the figure into JPEG file (s).

        Parameters
        ----------
        kwargs:
        images_to_save : str
            Images to save. options are: 'A', 'B', or 'Both' (default).
        invert : boolean
            If True, the image will be inverted.
        
        '''
        images_to_save = kwargs.get("images_to_save", 'Both')
        invert = kwargs.get("invert", False)

        if images_to_save == 'Both' or images_to_save == 'A':
            if self.ftype == 0:
                fname_jpg = os.path.splitext(self.fname)[0] + '_' + self.DetA.strip('\x00') + '.jpg'
            else:
                fname_jpg = os.path.splitext(self.fname)[0] + 'DetA.jpg'
            Img = self.RawImageA_8bit_thresholds()[0]
            if invert:
                Img =  uint8(255) - Img
            PILImage.fromarray(Img).save(fname_jpg)

        try:
            if images_to_save == 'Both' or images_to_save == 'B':
                if self.ftype == 0:
                    fname_jpg = os.path.splitext(self.fname)[0] +  '_' + self.DetB.strip('\x00') + '.jpg'
                else:
                    fname_jpg = os.path.splitext(self.fname)[0] + 'DetB.jpg'
                Img = self.RawImageB_8bit_thresholds()[0]
                if invert:
                    Img =  uint8(255) - Img
                PILImage.fromarray(Img).save(fname_jpg)
        except:
            print('No Detector B image to save')

    def save_images_tif(self, images_to_save = 'Both'):
        '''
        Save the detector images into TIF file (s).

        Parameters
        ----------
        images_to_save : str
            Images to save. options are: 'A', 'B', or 'Both' (default).
        
        '''
        if self.ftype == 0:
            if images_to_save == 'Both' or images_to_save == 'A':
                fnameA = os.path.splitext(self.fname)[0] + '_' + self.DetA.strip('\x00') + '.tif'
                tiff.imsave(fnameA, self.RawImageA)
            if self.DetB != 'None':
                if images_to_save == 'Both' or images_to_save == 'B':
                    fnameB = os.path.splitext(self.fname)[0] + '_' + self.DetB.strip('\x00') + '.tif'
                    tiff.imsave(fnameB, self.RawImageB)
        else:
            print('original File is already in TIF format')
        
    def get_image_min_max(self, image_name = 'ImageA', thr_min = 1.0e-4, thr_max = 1.0e-3, nbins=256, disp_res = False):
        '''
        Calculates the data range of the EM data. ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

        Calculates histogram of pixel intensities of of the loaded image
        with number of bins determined by parameter nbins (default = 256)
        and normalizes it to get the probability distribution function (PDF),
        from which a cumulative distribution function (CDF) is calculated.
        Then given the threshold_min, threshold_max parameters,
        the minimum and maximum values for the image are found by finding
        the intensities at which CDF= threshold_min and (1- threshold_max), respectively.

        Parameters
        ----------
        image_name : string
            the name of the image to perform this operations (defaulut is 'RawImageA')
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        disp_res : boolean
            (default is False) - to plot/ display the results

        Returns:
            dmin, dmax: (float) minimum and maximum values of the data range.   
        '''
        if image_name == 'ImageA':
            im = self.ImageA
        if image_name == 'ImageB':
            im = self.ImageB
        if image_name == 'RawImageA':
            im = self.RawImageA
        if image_name == 'RawImageB':
            im = self.RawImageB
        return get_min_max_thresholds(im, thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=disp_res)

    def RawImageA_8bit_thresholds(self, thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
        '''
        Convert the Image A into 8-bit array

        Parameters
        ----------
        thr_min : float
            lower CDF threshold for determining the minimum data value
        thr_max : float
            upper CDF threshold for determining the maximum data value
        data_min : float
            If different from data_max, this value will be used as low bound for I8 data conversion
        data_max : float
            If different from data_min, this value will be used as high bound for I8 data conversion
        nbins : int
            number of histogram bins for building the PDF and CDF
        
        Returns
        dt, data_min, data_max
            dt : 2D uint8 array
                Converted data
            data_min : float
                value used as low bound for I8 data conversion
            data_max : float
                value used as high bound for I8 data conversion
        '''
        if self.EightBit==1:
            #print('8-bit image already - no need to convert')
            dt = self.RawImageA
        else:
            if data_min == data_max:
                data_min, data_max = self.get_image_min_max(image_name ='RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
            dt = ((np.clip(self.RawImageA, data_min, data_max) - data_min)/(data_max-data_min)*255.0).astype(np.uint8)
        return dt, data_min, data_max

    def RawImageB_8bit_thresholds(self, thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
        '''
        Convert the Image B into 8-bit array

        Parameters
        ----------
        thr_min : float
            lower CDF threshold for determining the minimum data value
        thr_max : float
            upper CDF threshold for determining the maximum data value
        data_min : float
            If different from data_max, this value will be used as low bound for I8 data conversion
        data_max : float
            If different from data_min, this value will be used as high bound for I8 data conversion
        nbins : int
            number of histogram bins for building the PDF and CDF
        
        Returns
        dt, data_min, data_max
            dt : 2D uint8 array
                Converted data
            data_min : float
                value used as low bound for I8 data conversion
            data_max : float
                value used as high bound for I8 data conversion
        '''
        if self.EightBit==1:
            #print('8-bit image already - no need to convert')
            dt = self.RawImageB
        else:
            if data_min == data_max:
                data_min, data_max = self.get_image_min_max(image_name ='RawImageB', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
            dt = ((np.clip(self.RawImageB, data_min, data_max) - data_min)/(data_max-data_min)*255.0).astype(np.uint8)
        return dt, data_min, data_max
    
    def save_snapshot(self, display = True, dpi=300, thr_min = 1.0e-3, thr_max = 1.0e-3, nbins=256):
        '''
        Builds an image that contains both the Detector A and Detector B (if present) images as well as a table with important FIB-SEM parameters.

        Parameters
        ----------
        thr_min : float
            lower CDF threshold for determining the minimum data value
        thr_max : float
            upper CDF threshold for determining the maximum data value
        data_min : float
            If different from data_max, this value will be used as low bound for I8 data conversion
        data_max : float
            If different from data_min, this value will be used as high bound for I8 data conversion
        nbins : int
            number of histogram bins for building the PDF and CDF
        
        Returns
        dt, data_min, data_max
            dt : 2D uint8 array
                Converted data
            data_min : float
                value used as low bound for I8 data conversion
            data_max : float
                value used as high bound for I8 data conversion
        '''
        ifDetB = (self.DetB != 'None')
        if ifDetB:
            fig, axs = subplots(3, 1, figsize=(11,8))
        else:
            fig, axs = subplots(2, 1, figsize=(7,8))
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.15, hspace=0.1)
        dminA, dmaxA = self.get_image_min_max(image_name ='RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
        axs[1].imshow(self.RawImageA, cmap='Greys', vmin=dminA, vmax=dmaxA)
        if ifDetB:
            dminB, dmaxB = self.get_image_min_max(image_name ='RawImageB', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
            axs[2].imshow(self.RawImageB, cmap='Greys', vmin=dminB, vmax=dmaxB)
        try:
            ttls = [self.Notes.strip('\x00'),
                'Detector A:  '+ self.DetA.strip('\x00') + ',  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}'.format(dminA, dmaxA, thr_min, thr_max) + '    (Brightness: {:.1f}, Contrast: {:.1f})'.format(self.BrightnessA, self.ContrastA),
                'Detector B:  '+ self.DetB.strip('\x00') + ',  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}'.format(dminB, dmaxB, thr_min, thr_max) + '    (Brightness: {:.1f}, Contrast: {:.1f})'.format(self.BrightnessB, self.ContrastB)]
        except:
            ttls = ['', 'Detector A', '']
        for ax, ttl in zip(axs, ttls):
            ax.axis(False)
            ax.set_title(ttl, fontsize=10)
        fig.suptitle(self.fname)
        
        if self.FileVersion > 8:
            cell_text = [['Sample ID', '{:s}'.format(self.Sample_ID.strip('\x00')), '',
                          'Frame Size', '{:d} x {:d}'.format(self.XResolution, self.YResolution), '',
                          'Scan Rate', '{:.3f} MHz'.format(self.ScanRate/1.0e6)],
                        ['Machine ID', '{:s}'.format(self.MachineID.strip('\x00')), '',
                          'Pixel Size', '{:.1f} nm'.format(self.PixelSize), '',
                          'Oversampling', '{:d}'.format(self.Oversampling)],
                         ['FileVersion', '{:d}'.format(self.FileVersion), '',
                          'Working Dist.', '{:.3f} mm'.format(self.WD), '',
                          'FIB Focus', '{:.1f}  V'.format(self.FIBFocus)],
                         ['Bit Depth', '{:d}'.format(8 *(2 - self.EightBit)), '',
                         'EHT Voltage\n\nSEM Current', '{:.3f} kV \n\n{:.3f} nA'.format(self.EHT, self.SEMCurr*1.0e9), '',
                         'FIB Probe', '{:d}'.format(self.FIBProb)]]
        else:
            if self.FileVersion > 0:
                cell_text = [['', '', '',
                              'Frame Size', '{:d} x {:d}'.format(self.XResolution, self.YResolution), '',
                              'Scan Rate', '{:.3f} MHz'.format(self.ScanRate/1.0e6)],
                            ['Machine ID', '{:s}'.format(self.MachineID.strip('\x00')), '',
                              'Pixel Size', '{:.1f} nm'.format(self.PixelSize), '',
                              'Oversampling', '{:d}'.format(self.Oversampling)],
                             ['FileVersion', '{:d}'.format(self.FileVersion), '',
                              'Working Dist.', '{:.3f} mm'.format(self.WD), '',
                              'FIB Focus', '{:.1f}  V'.format(self.FIBFocus)],
                             ['Bit Depth', '{:d}'.format(8 *(2 - self.EightBit)), '',
                             'EHT Voltage', '{:.3f} kV'.format(self.EHT), '',
                             'FIB Probe', '{:d}'.format(self.FIBProb)]]
            else:
                cell_text = [['', '', '',
                              'Frame Size', '{:d} x {:d}'.format(self.XResolution, self.YResolution), '',
                              'Scan Rate', ''],
                            ['Machine ID', '', '',
                              'Pixel Size', '{:.1f} nm'.format(self.PixelSize), '',
                              'Oversampling', ''],
                             ['FileVersion', '{:d}'.format(self.FileVersion), '',
                              'Working Dist.', ' ', '',
                              'FIB Focus', ''],
                             ['Bit Depth', '{:d}'.format(8 *(2 - self.EightBit)), '',
                             'EHT Voltage', '', '',
                             'FIB Probe', '']]
        llw0=0.3
        llw1=0.18
        llw2=0.02
        clw = [llw1, llw0, llw2, llw1, llw1, llw2, llw1, llw1]
        tbl = axs[0].table(cellText=cell_text,
                           colWidths=clw,
                           cellLoc='center',
                           colLoc='center',
                           bbox = [0.02, 0, 0.96, 1.0],
                           #bbox = [0.45, 1.02, 2.8, 0.55],
                           zorder=10)

        fig.savefig(os.path.splitext(self.fname)[0] + '_snapshot.png', dpi=dpi)
        if display == False:
            plt.close(fig)
    
    def analyze_noise_ROIs(self, Noise_ROIs, Hist_ROI, **kwargs):
        '''
        Analyses the noise statistics in the selected ROI's of the EM data.
        ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
        
        Calls Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs)
        Performs following:
        1. For ach of the selected ROI's, this method will perfrom the following:
            1a. Smooth the data by 2D convolution with a given kernel.
            1b. Determine "Noise" as difference between the original raw and smoothed data.        
            1c. Calculate the mean intensity value of the data and variance of the above "Noise"
        2. Plot the dependence of the noise variance vs. image intensity.
        3. One of the parameters is a DarkCount. If it is not explicitly defined as input parameter,
            it will be determined from the header data:
                for RawImageA it is self.Scaling[1,0]
                for RawImageB it is self.Scaling[1,1]
        4. The equation is determined for a line that passes through the point:
                Intensity=DarkCount and Noise Variance = 0
                and is a best fit for the [Mean Intensity, Noise Variance] points
                determined for each ROI (Step 1 above).
        5. Another ROI (defined by Hist_ROI parameter) is used to built an
            intensity histogram of the actual data. Peak of that histogram is determined.
        6. The data is plotted. Two values of SNR are defined from the slope of the line in Step 4:
            PSNR (Peak SNR) = Mean Intensity/sqrt(Noise Variance) at the intensity
                at the histogram peak determined in the Step 5.
            DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance),
                where Max and Min Intensity are determined by corresponding cummulative
                threshold parameters, and Noise Variance is taken at the intensity
                in the middle of the range (Min Intensity + Max Intensity)/2.0

        Parameters
        ----------
        Noise_ROIs : list of lists: [[left, right, top, bottom]]
            list of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the noise.
        Hist_ROI : list [left, right, top, bottom]
            coordinates (indices) of the boundaries of the image subset to evaluate the real data histogram.

        kwargs:
        image_name : string
            the name of the image to perform this operations (defaulut is 'RawImageA').
        DarkCount : float
            the value of the Intensity Data at 0.
        kernel : 2D float array
            a kernel to perfrom 2D smoothing convolution.
        filename : str
            filename - used for plotting the data. If not explicitly defined will use the instance attribute self.fname
        nbins_disp : int
            (default 256) number of histogram bins for building the PDF and CDF to determine the data range for data display.
        thresholds_disp : list [thr_min_disp, thr_max_disp]
            (default [1e-3, 1e-3]) CDF threshold for determining the min and max data values for display.
        nbins_analysis : int
            (default 256) number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5.
        thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
            (default [2e-2, 2e-2]) CDF threshold for building the data histogram in Step 5.
        nbins_analysis : int
             (default 256) number of histogram bins for building the data histogram in Step 5.
        disp_res : boolean
            (default is False) - to plot/ display the results

        Returns:
        mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR
            mean_vals and var_vals are the Mean Intensity and Noise Variance values for the Noise_ROIs (Step 1)
            NF_slope is the slope of the linear fit curve (Step 4)
            PSNR and DSNR are Peak and Dynamic SNR's (Step 6)
        '''
        image_name = kwargs.get("image_name", '')

        if image_name == 'RawImageA':
            ImgEM = self.RawImageA.astype(float)
            DarkCount = self.Scaling[1,0]
        if image_name == 'RawImageB' and self.DetB != 'None':
            ImgEM = self.RawImageB.astype(float)
            DarkCount = self.Scaling[1,1]

        if (image_name == 'RawImageA') or (image_name == 'RawImageB' and self.DetB != 'None'):
            st = 1.0/np.sqrt(2.0)
            def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
            def_kernel = def_kernel/def_kernel.sum()
            kernel = kwargs.get("kernel", def_kernel)
            DarkCount = kwargs.get("DarkCount", DarkCount)
            nbins_disp = kwargs.get("nbins_disp", 256)
            thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
            nbins_analysis = kwargs.get("nbins_analysis", 100)
            thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
            Notes = kwargs.get("Notes", self.Notes.strip('\x00'))
            kwargs['kernel'] = kernel
            kwargs['DarkCount'] = DarkCount
            kwargs['img_label'] = image_name
            kwargs['res_fname'] = os.path.splitext(self.fname)[0] + '_' + image_name + '_Noise_Analysis_ROIs.png'
            kwargs['Notes'] = Notes
            mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR = Single_Image_Noise_ROIs(ImgEM, Noise_ROIs, Hist_ROI, **kwargs)

        else:
            print('No valid image name selected')
            mean_vals = 0.0
            var_vals = 0.0
            NF_slope = 0.0
            PSNR = 0.0
            MSNR = 0.0
            DSNR = 0.0

        return mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR

    def analyze_noise_statistics(self, **kwargs):
        '''
        Analyses the noise statistics of the EM data image.
        ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
        
        Calls Single_Image_Noise_Statistics(img, **kwargs)
        Performs following:
        1. Smooth the image by 2D convolution with a given kernel.
        2. Determine "Noise" as difference between the original raw and smoothed data.
        3. Build a histogram of Smoothed Image.
        4. For each histogram bin of the Smoothed Image (Step 3), calculate the mean value and variance for the same pixels in the original image.
        5. Plot the dependence of the noise variance vs. image intensity.
        6. One of the parameters is a DarkCount. If it is not explicitly defined as input parameter,
            it will be set to 0
        7. The equation is determined for a line that passes through the point:
                Intensity=DarkCount and Noise Variance = 0
                and is a best fit for the [Mean Intensity, Noise Variance] points
                determined for each ROI (Step 1 above).
        8. The data is plotted. Two values of SNR are defined from the slope of the line in Step 7:
            PSNR (Peak SNR) = Intensity /sqrt(Noise Variance) at the intensity
                at the histogram peak determined in the Step 3.
            MSNR (Mean SNR) = Mean Intensity /sqrt(Noise Variance)
            DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance),
                where Max and Min Intensity are determined by corresponding cummulative
                threshold parameters, and Noise Variance is taken at the intensity
                in the middle of the range (Min Intensity + Max Intensity)/2.0

        Parameters
        ----------
            kwargs:
            image_name : str
                Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
            evaluation_box : list of 4 int
                evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
                if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
            DarkCount : float
                the value of the Intensity Data at 0.
            kernel : 2D float array
                a kernel to perfrom 2D smoothing convolution.
            nbins_disp : int
                (default 256) number of histogram bins for building the PDF and CDF to determine the data range for data display.
            thresholds_disp : list [thr_min_disp, thr_max_disp]
                (default [1e-3, 1e-3]) CDF threshold for determining the min and max data values for display.
            nbins_analysis : int
                (default 256) number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5.
            thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
                (default [2e-2, 2e-2]) CDF threshold for building the data histogram in Step 5.
            nbins_analysis : int
                 (default 256) number of histogram bins for building the data histogram in Step 5.
            disp_res : boolean
                (default is False) - to plot/ display the results
            save_res_png : boolean
                save the analysis output into a PNG file (default is True)
            res_fname : string
                filename for the result image ('Noise_Analysis.png')
            img_label : string
                optional image label
            Notes : string
                optional additional notes
            dpi : int

        Returns:
        mean_vals, var_vals, I0, PSNR, DSNR, popt, result
            mean_vals and var_vals are the Mean Intensity and Noise Variance values for Step 5
            I0 is zero intercept (should be close to DarkCount)
            PSNR and DSNR are Peak and Dynamic SNR's (Step 8)
        '''
        image_name = kwargs.get("image_name", 'RawImageA')

        if image_name == 'RawImageA':
            ImgEM = self.RawImageA.astype(float)
            DarkCount = self.Scaling[1,0]
        if image_name == 'RawImageB' and self.DetB != 'None':
            ImgEM = self.RawImageB.astype(float)
            DarkCount = self.Scaling[1,1]

        if (image_name == 'RawImageA') or (image_name == 'RawImageB' and self.DetB != 'None'):
            st = 1.0/np.sqrt(2.0)
            evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
            def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
            def_kernel = def_kernel/def_kernel.sum()
            kernel = kwargs.get("kernel", def_kernel)
            DarkCount = kwargs.get("DarkCount", DarkCount)
            nbins_disp = kwargs.get("nbins_disp", 256)
            thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
            nbins_analysis = kwargs.get("nbins_analysis", 100)
            thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
            disp_res = kwargs.get("disp_res", True)
            save_res_png = kwargs.get("save_res_png", True)
            default_res_name = os.path.splitext(self.fname)[0] + '_Noise_Analysis_' + image_name + '.png'
            res_fname = kwargs.get("res_fname", default_res_name)
            img_label = kwargs.get("img_label", self.Sample_ID)
            Notes = kwargs.get("Notes", self.Notes.strip('\x00'))
            dpi = kwargs.get("dpi", 300)

            noise_kwargs = {'image_name' : image_name,
                            'evaluation_box' : evaluation_box,
                            'kernel' : kernel,
                            'DarkCount' : DarkCount,
                            'nbins_disp' : nbins_disp,
                            'thresholds_disp' : thresholds_disp,
                            'nbins_analysis' : nbins_analysis,
                            'thresholds_analysis' : thresholds_analysis,
                            'disp_res' : disp_res,
                            'save_res_png' : save_res_png,
                            'res_fname' : res_fname,
                            'Notes' : Notes,
                            'dpi' : dpi}

            mean_vals, var_vals, I0, PSNR, DSNR, popt, result =  Single_Image_Noise_Statistics(ImgEM, **noise_kwargs)
        else:
            mean_vals, var_vals, I0, PSNR, DSNR, popt, result = [], [], 0.0, 0.0, np.array((0.0, 0.0)), [] 
        return mean_vals, var_vals, I0, PSNR, DSNR, popt, result
    

    def analyze_SNR_autocorr(self, **kwargs):
        '''
        Estimates SNR using auto-correlation analysis of a single image.
        ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
        
        Calculates SNR of a single image base on auto-correlation analysis of a single image, after [1].
        Calls function Single_Image_SNR(img, **kwargs)
        
        Parameters
        ---------
        kwargs:
        image_name : str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        edge_fraction : float
            fraction of the full autocetrrelation range used to calculate the "mean value" (default is 0.10)
        disp_res : boolean
            display results (plots) (default is True)
        save_res_png : boolean
            save the analysis output into a PNG file (default is True)
        res_fname : string
            filename for the sesult image ('SNR_result.png')
        img_label : string
            optional image label
        dpi : int
            dots-per-inch resolution for the output image
            
        Returns:
            xSNR, ySNR : float, float
                SNR determind using the method in [1] along X- and Y- directions.
                If there is a direction with slow varying data - that direction provides more accurate SNR estimate
                Y-streaks in typical FIB-SEM data provide slow varying Y-component becuase streaks
                usually get increasingly worse with increasing Y. 
                So for typical FIB-SEM data use ySNR
        
        [1] J. T. L. Thong et al, Single-image signal-tonoise ratio estimation. Scanning, 328–336 (2001).
        '''
        image_name = kwargs.get("image_name", 'RawImageA')
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        edge_fraction = kwargs.get("edge_fraction", 0.10)
        disp_res = kwargs.get("disp_res", True)
        save_res_png = kwargs.get("save_res_png", True)
        default_res_name = os.path.splitext(self.fname)[0] + '_AutoCorr_Noise_Analysis_' + image_name + '.png'
        res_fname = kwargs.get("res_fname", default_res_name)
        dpi = kwargs.get("dpi", 300)

        SNR_kwargs = {'edge_fraction' : edge_fraction,
                        'disp_res' : disp_res,
                        'save_res_png' : save_res_png,
                        'res_fname' : res_fname,
                        'img_label' : image_name,
                        'dpi' : dpi}

        if image_name == 'RawImageA':
            img = self.RawImageA
        if image_name == 'RawImageB':
            img = self.RawImageB
        if image_name == 'ImageA':
            img = self.ImageA
        if image_name == 'ImageB':
            img = self.ImageB

        xi = 0
        yi = 0
        ysz, xsz = img.shape
        xa = xi + xsz
        ya = yi + ysz
        xi_eval = xi + evaluation_box[2]
        if evaluation_box[3] > 0:
            xa_eval = xi_eval + evaluation_box[3]
        else:
            xa_eval = xa
        yi_eval = yi + evaluation_box[0]
        if evaluation_box[1] > 0:
            ya_eval = yi_eval + evaluation_box[1]
        else:
            ya_eval = ya

        xSNR, ySNR, rSNR= Single_Image_SNR(img[yi_eval:ya_eval, xi_eval:xa_eval], **SNR_kwargs)

        return xSNR, ySNR, rSNR


    def show_eval_box(self, **kwargs):
        '''
        Show the box used for noise analysis.
        ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

        kwargs
        ---------
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        image_name : str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        data_dir : str
            data direcory (path)
        Sample_ID : str
            Sample ID
        invert_data : boolean
            If True - the data is inverted
        save_res_png  : boolean
            Save PNG image of the frame overlaid with with evaluation box
        '''
        image_name = kwargs.get("image_name", 'RawImageA')
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0]) 
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        nbins_disp = kwargs.get("nbins_disp", 256)
        thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])    
        invert_data =  kwargs.get("invert_data", False)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )

        if image_name == 'RawImageA':
            img = self.RawImageA
        if image_name == 'RawImageB':
            img = self.RawImageB
        if image_name == 'ImageA':
            img = self.ImageA
        if image_name == 'ImageB':
            img = self.ImageB
        range_disp = get_min_max_thresholds(img, thresholds_disp[0], thresholds_disp[1], nbins_disp)

        xi = 0
        yi = 0
        ysz, xsz = img.shape
        xa = xi + xsz
        ya = yi + ysz
        xi_eval = xi + evaluation_box[2]
        if evaluation_box[3] > 0:
            xa_eval = xi_eval + evaluation_box[3]
        else:
            xa_eval = xa
        yi_eval = yi + evaluation_box[0]
        if evaluation_box[1] > 0:
            ya_eval = yi_eval + evaluation_box[1]
        else:
            ya_eval = ya
            
        fig, ax = subplots(1,1, figsize = (10.0, 11.0*ysz/xsz))
        ax.imshow(img, cmap='Greys', vmin = range_disp[0], vmax = range_disp[1])
        ax.grid(True, color = "cyan")
        ax.set_title(self.fname)
        rect_patch = patches.Rectangle((xi_eval,yi_eval),abs(xa_eval-xi_eval)-2,abs(ya_eval-yi_eval)-2, linewidth=2.0, edgecolor='yellow',facecolor='none')
        ax.add_patch(rect_patch)
        if save_res_png :
            fig.savefig(os.path.splitext(self.fname+'_evaluation_box.png', dpi=300))


    def determine_field_fattening_parameters(self, **kwargs):
        '''
        Perfrom 2D parabolic fit (calls Perform_2D_fit(Img, estimator, **kwargs)) and determine the field-flattening parameters
        
        Parameters
        ----------
        kwargs:
        image_name : str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        estimator : RANSACRegressor(),
                    LinearRegression(),
                    TheilSenRegressor(),
                    HuberRegressor()
        bins : int
            binsize for image binning. If not provided, bins=10
        calc_corr : boolean
            If True - the full image correction is calculated
        ignore_Y  : boolean
            If True - the parabolic fit to only X is perfromed
        disp_res : boolean
            (default is False) - to plot/ display the results
        save_res_png : boolean
            save the analysis output into a PNG file (default is False)
        res_fname : string
            filename for the sesult image ('2D_Parabolic_Fit.png')
        label : string
            optional image label
        dpi : int

        Returns:
        intercept, coefs, mse, img_full_correction
        '''
        image_name = kwargs.get("image_name", 'RawImageA')
        estimator = kwargs.get("estimator", LinearRegression())
        del kwargs["estimator"]
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0]) 
        calc_corr = kwargs.get("calc_corr", False)
        ignore_Y = kwargs.get("ignore_Y", False)
        lbl = kwargs.get("label", '')
        disp_res = kwargs.get("disp_res", True)
        bins = kwargs.get("bins", 10) #bins = 10
        save_res_png = kwargs.get("save_res_png", False)
        res_fname = kwargs.get("res_fname", '2D_Parabolic_Fit.png')
        dpi = kwargs.get("dpi", 300)

        if image_name == 'RawImageA':
            img = self.RawImageA - self.Scaling[1,0]
        if image_name == 'RawImageB':
            img = self.RawImageB - self.Scaling[1,1]
        if image_name == 'ImageA':
            img = self.ImageA
        if image_name == 'ImageB':
            img = self.ImageB

        ysz, xsz = img.shape
        Xsect = kwargs.get("Xsect", xsz//2)
        Ysect = kwargs.get("Ysect", ysz//2)

        intercept, coefs, mse, img_full_correction = Perform_2D_fit(img, estimator, **kwargs)
        if calc_corr:
            self.image_correction_source = image_name
            self.img_full_correction = img_full_correction
        self.intercept = intercept
        self.coefs = coefs
        return intercept, coefs, mse, img_full_correction

        
    def flatten_image(self, **kwargs):
        '''
        Flatten the image. Imah=ge flattening parameters must be determined (determine_field_fattening_parameters)

        Parameters
        ----------
        kwargs:
        image_name : str

        Returns:
        flattened_image : 2D array
        '''
        image_name = kwargs.get("image_name", 'RawImageA')

        if hasattr(self, 'image_correction_source') and hasattr(self, 'img_full_correction'):
            if image_name == self.image_correction_source:
                if image_name == 'RawImageA':
                    flattened_image = (self.RawImageA - self.Scaling[1,0])*self.img_full_correction + self.Scaling[1,0]
                if image_name == 'RawImageB':
                    flattened_image = (self.RawImageB - self.Scaling[1,1])*self.img_full_correction + self.Scaling[1,1]
                if image_name == 'ImageA':
                    flattened_image = self.ImageA*self.img_full_correction
                if image_name == 'ImageB':
                    flattened_image = self.ImageB*self.img_full_correction
                flattened=True
            else:
                print('Inconsistent Image='+ image_name + ' and Image Correction Source='+self.image_correction_source)
                print('Image flattening not performed')
                flattened=False
        else:
            print('Image Correction Parameters not Determined')
            print('execute method determine_field_fattening_parameters()')
            print('Image flattening not performed')
            flattened=False

        if not flattened:
            if image_name == 'RawImageA':
                flattened_image = self.RawImageA
            if image_name == 'RawImageB':
                flattened_image = self.RawImageB
            if image_name == 'ImageA':
                flattened_image = self.ImageA
            if image_name == 'ImageB':
                flattened_image = self.ImageB

        return flattened_image


###################################################
#   Helper functions for FIBSEM_dataset class
###################################################
def determine_regularized_affine_transform(src_pts, dst_pts, l2_matrix = None, targ_vector = None):
    """
    Estimate N-D affine transformation with regularization from a set of corresponding points.
    ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

        We can determine the over-, well- and under-determined parameters
        with the total least-squares method.
        Number of source and destination coordinates must match.
        The transformation is defined as:
            X = (a0*x + a1*y + a2) 
            Y = (b0*x + b1*y + b2)
        This is regularized Affine estimation - it is regularized so that the penalty is for deviation from a target (default target is rigid shift) transformation
        a0 =1, a1=0, b0=1, b1=1 are parameters for target (shift) transform. Deviation from these is penalized.

        The coefficients appear linearly so we can write
        A x = B, where:
            A   = [[x y 1 0 0 0]
                   [0 0 0 x y 1]]
            Htarget.T = [a0 a1 a2 b0 b1 b2]
            B.T = [X Y]

        In case of ordinary least-squares (OLS) the solution of this system
        of equations is:
        H = np.linalg.inv(A.T @ A) @ A @ B
        
        In case of least-squares with Tikhonov-like regularization:
        H = np.linalg.inv(A.T @ A + Γ.T @ Γ) @ (A @ B + Γ.T @ Γ @ Htarget)
        where Γ.T @ Γ (for simplicity will call it L2 vector) is regularization term and Htarget 
        is a target solution, deviation from which is minimized in L2 sense
     """ 

    src_matrix, src = _center_and_normalize_points_gs(src_pts)
    dst_matrix, dst = _center_and_normalize_points_gs(dst_pts)

    n, d = src.shape
    n2 = n*n   # normalization factor, so that shrinkage parameter does not depend on the number of points

    A = np.zeros((n * d, d * (d + 1)))
    # fill the A matrix with the appropriate block matrices; see docstring
    # for 2D example — this can be generalised to more blocks in the 3D and
    # higher-dimensional cases.
    for ddim in range(d):
        A[ddim*n : (ddim + 1) * n, ddim * (d + 1) : ddim * (d + 1) + d] = src
        A[ddim*n : (ddim + 1) * n, ddim * (d + 1) + d] = 1

    AtA = A.T @ A / n2
    
    if l2_matrix is None:
        l2 = 1.0e-5   # default shrinkage parameter
        l2_matrix = np.eye(2 * (d + 1)) * l2
        for ddim in range(d):
            ii = (d + 1) * (ddim + 1) - 1
            l2_matrix[ii,ii] = 0

    if targ_vector is None:
        targ_vector = np.zeros(2 * (d + 1))
        targ_vector[0] = 1
        targ_vector[4] = 1


    Hp = np.linalg.inv(AtA + l2_matrix) @ (A.T @ dst.T.ravel() / n2 + l2_matrix @ targ_vector)
    Hm = np.eye(d + 1)
    Hm[0:d, 0:d+1] = Hp.reshape(d, d + 1)
    H = np.linalg.inv(dst_matrix) @ Hm @ src_matrix
    return H

def _umeyama(src, dst, estimate_scale):
    """
    Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


class ShiftTransform(ProjectiveTransform):
    """
    ScaleShift transformation. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

    Has the following form:
        X = x +  a2
        Y = y + b2
    and the homogeneous transformation matrix is::
        [[1  0   a2]
         [0   1  b2]
         [0   0    1]]
    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.
    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.
        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.
    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.
    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.
    """

    def __init__(self, matrix=None, translation=None, *, dimensionality=2):
        
        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if translation is not None and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if translation is not None and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
            self.params[0,1] = 0
            self.params[1,0] = 0
        elif translation is not None:  # note: 2D only
            self.params = np.array([[1.0, 0.0,  0.0],
                                    [0.0,  1.0, 0.0],
                                    [0.0,  0.0, 1.0]])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)
    def estimate(self, src, dst):
        '''
                Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        '''
        translation = np.mean(np.array(dst.astype(np.float)-src.astype(np.float)), axis=0)
        self.params = np.array([[1.0, 0.0,  0.0],
                                [0.0,  1.0, 0.0],
                                [0.0,  0.0, 1.0]])
        self.params[0:2, 2] = translation
        return True
        
    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]


class XScaleShiftTransform(ProjectiveTransform):
    '''
    XScaleShift transformation. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

    Has the following form::
        X = a0*x +  a2 = sx*x + a2
        Y = y + b2 = y + b2
    where ``sx`` and ``sy`` are scale factors in the x and y directions,
    and the homogeneous transformation matrix is::
        [[a0  0   a2]
         [0   1   b2]
         [0   0    1]]
    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.
    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.
        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.
    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.
    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.
    '''

    def __init__(self, matrix=None, scale=None, translation=None, *, dimensionality=2):
        params = (scale is not None) or (translation is not None)
        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if params and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
            self.params[0,1] = 0
            self.params[1,0] = 0
        elif params:  # note: 2D only
            if scale is None:
                scale = (1, 1)

            if translation is None:
                translation = (0, 0)

            if np.isscalar(scale):
                sx = scale
            else:
                sx = scale

            self.params = np.array([[sx, 0,  0],
                                    [0,  1, 0],
                                    [0,  0,  1]])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)
    def estimate(self, src, dst):
        """
                Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
 
        n, d = src.shape
        xsrc = np.array(src)[:,0].astype(np.float)
        ysrc = np.array(src)[:,1].astype(np.float)
        xdst = np.array(dst)[:,0].astype(np.float)
        ydst = np.array(dst)[:,1].astype(np.float)
        s00 = np.sum(xsrc)
        s01 = np.sum(xdst)
        sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xsrc*xsrc) - s00*s00)
        #s10 = np.sum(ysrc)
        #s11 = np.sum(ydst)
        #sy = (n*np.dot(ysrc, ydst) - s10*s11)/(n*np.sum(ysrc*ysrc) - s10*s10)
        sy = 1.0

        tx = np.mean(xdst) - sx * np.mean(xsrc)
        ty = np.mean(ydst) - sy * np.mean(ysrc)

        self.params = np.array([[sx, 0,  tx],
                                [0,  sy, ty],
                                [0,  0,  1]])
        return True
    
    def print_res(self):
        print('Printing from iside the class XScaleShiftTransform')

    @property
    def scale(self):
        return np.sqrt(np.sum(self.params ** 2, axis=0))[:self.dimensionality]

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]


class ScaleShiftTransform(ProjectiveTransform):
    '''
    ScaleShift transformation. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

    Has the following form::
        X = a0*x +  a2 = sx*x + a2
        Y = b1*y + b2 = sy*y + b2
    where ``sx`` and ``sy`` are scale factors in the x and y directions,
    and the homogeneous transformation matrix is::
        [[a0  0   a2]
         [0   b1  b2]
         [0   0    1]]
    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.
    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.
        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.
    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.
    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.
    '''

    def __init__(self, matrix=None, scale=None, translation=None, *, dimensionality=2):
        params = (scale is not None) or (translation is not None)
        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if params and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
            self.params[0,1] = 0
            self.params[1,0] = 0
        elif params:  # note: 2D only
            if scale is None:
                scale = (1, 1)

            if translation is None:
                translation = (0, 0)

            if np.isscalar(scale):
                sx = sy = scale
            else:
                sx, sy = scale

            self.params = np.array([[sx, 0,  0],
                                    [0,  sy, 0],
                                    [0,  0,  1]])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)
    def estimate(self, src, dst):
        """
                Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
 
        n, d = src.shape
        xsrc = np.array(src)[:,0].astype(np.float)
        ysrc = np.array(src)[:,1].astype(np.float)
        xdst = np.array(dst)[:,0].astype(np.float)
        ydst = np.array(dst)[:,1].astype(np.float)
        s00 = np.sum(xsrc)
        s01 = np.sum(xdst)
        sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xsrc*xsrc) - s00*s00)
        s10 = np.sum(ysrc)
        s11 = np.sum(ydst)
        sy = (n*np.dot(ysrc, ydst) - s10*s11)/(n*np.sum(ysrc*ysrc) - s10*s10)

        tx = np.mean(xdst) - sx * np.mean(xsrc)
        ty = np.mean(ydst) - sy * np.mean(ysrc)

        self.params = np.array([[sx, 0,  tx],
                                [0,  sy, ty],
                                [0,  0,  1]])
        return True
        
    @property
    def scale(self):
        return np.sqrt(np.sum(self.params ** 2, axis=0))[:self.dimensionality]

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]


class RegularizedAffineTransform(ProjectiveTransform):
    """
    Regularized Affine transformation. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

    Has the following form::
        X = a0*x + a1*y + a2 =
          = sx*x*cos(rotation) - sy*y*sin(rotation + shear) + a2
        Y = b0*x + b1*y + b2 =
          = sx*x*sin(rotation) + sy*y*cos(rotation + shear) + b2
    where ``sx`` and ``sy`` are scale factors in the x and y directions,
    and the homogeneous transformation matrix is::
        [[a0  a1  a2]
         [b0  b1  b2]
         [0   0    1]]
    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.
    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.
        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    rotation : float, optional
        Rotation angle in counter-clockwise direction as radians. Only
        available for 2D.
    shear : float, optional
        Shear angle in counter-clockwise direction as radians. Only available
        for 2D.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.
    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.
    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.
    """

    def __init__(self, matrix=None, scale=None, rotation=None, shear=None,
                 translation=None, l2_matrix =None, targ_vector=None, *, dimensionality=2):

        self.l2_matrix = l2_matrix      # regularization vector
        self.targ_vector = targ_vector  # target 
        params = any(param is not None
                     for param in (scale, rotation, shear, translation))

        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if params and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
        elif params:  # note: 2D only
            if scale is None:
                scale = (1, 1)
            if rotation is None:
                rotation = 0
            if shear is None:
                shear = 0
            if translation is None:
                translation = (0, 0)

            if np.isscalar(scale):
                sx = sy = scale
            else:
                sx, sy = scale

            self.params = np.array([
                [sx * math.cos(rotation), -sy * math.sin(rotation + shear), 0],
                [sx * math.sin(rotation),  sy * math.cos(rotation + shear), 0],
                [                      0,                                0, 1]
            ])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)


    @property
    def scale(self):
        return np.sqrt(np.sum(self.params ** 2, axis=0))[:self.dimensionality]

    @property
    def rotation(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                'The rotation property is only implemented for 2D transforms.'
            )
        return math.atan2(self.params[1, 0], self.params[0, 0])

    @property
    def shear(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                'The shear property is only implemented for 2D transforms.'
            )
        beta = math.atan2(- self.params[0, 1], self.params[1, 1])
        return beta - self.rotation

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]



        # Thise are functions used for different steps of image analysis and registration

def ShiftTransform0(matrix=None, translation=None):
    return EuclideanTransform(matrix=matrix, rotation = 0, translation = translation)

def ScaleShiftTransform0(matrix=None, scale=None, translation=None):
    return AffineTransform(matrix=matrix, scale=scale, rotation = 0, shear=0, translation = translation)


def kp_to_list(kp):
    '''
    Convert a keypont object to a list (so that it can be "pickled").
    
    Returns
    pt, angle, size, response, class_id, octave
    (all extracted from corresponding cv2.KeyPoint() object attributes)
    '''
    x, y = kp.pt
    pt = float(x), float(y)
    angle = float(kp.angle) if kp.angle is not None else None
    size = float(kp.size) if kp.size is not None else None
    response = float(kp.response) if kp.response is not None else None
    class_id = int(kp.class_id) if kp.class_id is not None else None
    octave = int(kp.octave) if kp.octave is not None else None
    return pt, angle, size, response, class_id, octave

def list_to_kp(inp_list):
    '''
    Convert a list to a keypont object

    Parameters:
    inp_list : list
        List of Key-Point properties to initialize the following cv2.KeyPoint() object attributes:
        [pt, angle, size, response, class_id, octave]

    Returns:
    kp : Instance of cv2.KeyPoint() object 
    '''
    kp = cv2.KeyPoint()
    kp.pt = inp_list[0]
    kp.angle = inp_list[1]
    kp.size = inp_list[2]
    kp.response = inp_list[3]
    kp.octave = inp_list[4]
    kp.class_id = inp_list[5]
    return kp

def get_min_max_thresholds_file(params):
    '''
    Calculates the data range of the EM data ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

    Calculates histogram of pixel intensities of of the loaded image
    with number of bins determined by parameter nbins (default = 256)
    and normalizes it to get the probability distribution function (PDF),
    from which a cumulative distribution function (CDF) is calculated.
    Then given the threshold_min, threshold_max parameters,
    the minimum and maximum values for the image are found by finding
    the intensities at which CDF= threshold_min and (1- threshold_max), respectively.
    
    Parameters:
    ----------
    params =  fl, kwargs
        fl : str
            The string containing a full path to the EM data file.
        kwargs: dictioanry of kwargs:
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        image_name: string
            the name of the image to perform this operations (defaulut is 'RawImageA')
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF

    Returns:
        dmin, dmax: (float) minimum and maximum values of the data range.   
    '''
    fl, kwargs = params
    ftype = kwargs.get("ftype", 0)
    image_name = kwargs.get("image_name", 'RawImageA')
    thr_min = kwargs.get("threshold_min", 1e-3)
    thr_max = kwargs.get("threshold_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    dmin, dmax = FIBSEM_frame(fl, ftype=ftype).get_image_min_max(image_name = 'RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins)
    return [dmin, dmax]


# Routines to extract Key-Points and Descriptors

def extract_keypoints_descr_files(params):
    '''
    Extracts Key-Points and Descriptors (single image) for SIFT procedure.
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    
    Parameters:
    -----------
    params = fl, dmin, dmax, kwargs
        fl : str
            image filename (full path)
        dmin : float   
            min data value for I8 conversion (open CV SIFT requires I8)
        dmax : float   
            max data value for I8 conversion (open CV SIFT requires I8)
        kwargs:
        -------
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        thr_min : float
            CDF threshold for determining the minimum data value
        thr_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        kp_max_num : int
            Max number of key-points to be matched.
            Key-points in every frame are indexed (in descending order)
            by the strength of the response. Only kp_max_num is kept for
            further processing.
            Set this value to -1 if you want to keep ALL keypoints (may take forever to process!)

    Returns:
        fnm : str
            path to the file containing Key-Points and Descriptors
    '''
    fl, dmin, dmax, kwargs = params
    ftype = kwargs.get("ftype", 0)
    thr_min = kwargs.get("threshold_min", 1e-3)
    thr_max = kwargs.get("threshold_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    kp_max_num = kwargs.get("kp_max_num", 10000)

    sift = cv2.xfeatures2d.SIFT_create()
    img, d1, d2 = FIBSEM_frame(fl, ftype=ftype).RawImageA_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = dmin, data_max = dmax, nbins=256)
    # extract keypoints and descriptors for both images
    kps, dess = sift.detectAndCompute(img, None)
    if kp_max_num != -1 and (len(kps) > kp_max_num):
        kp_ind = np.argsort([-kp.response for kp in kps])[0:kp_max_num]
        kps = np.array(kps)[kp_ind]
        dess = np.array(dess)[kp_ind]
    #key_points = [KeyPoint(kp) for kp in kps]
    key_points = [kp_to_list(kp) for kp in kps]
    kpd = [key_points, dess]
    fnm = os.path.splitext(fl)[0] + '_kpdes.bin'
    pickle.dump(kpd, open(fnm, 'wb')) # converts array to binary and writes to output
    #pickle.dump(dess, open(fnm, 'wb')) # converts array to binary and writes to output
    return fnm

def extract_keypoints_dataset(fls, data_minmax, DASK_client, **kwargs):
    '''
    Extracts Key-Points and Descriptors for SIFT procedure for all images (files) in the dataset.
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    
    Parameters:
    -----------
    params = fl, data_minmax, kwargs

    fls : str array
        array of image filenames (full paths)
    data_minmax : list of 4 parameters
        data_min_glob : float   
            min data value for I8 conversion (open CV SIFT requires I8)
        data_max_glob : float   
            max data value for I8 conversion (open CV SIFT requires I8)
        data_min_sliding : float array
            min data values (one per file) for I8 conversion
        data_max_sliding : float array
            max data values (one per file) for I8 conversion
    DASK_client : DASK client object
        DASK client (needs to be initialized and running by this time)

    kwargs:
    sliding_minmax : boolean
        if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
        if False - same data_min_glob and data_max_glob will be used for all files
    use_DASK : boolean
        use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
    ftype : int
        file type (0 - Shan Xu's .dat, 1 - tif)
    thr_min : float
        CDF threshold for determining the minimum data value
    thr_max : float
        CDF threshold for determining the maximum data value
    nbins : int
        number of histogram bins for building the PDF and CDF
    kp_max_num : int
        Max number of key-points to be matched.
        Key-points in every frame are indexed (in descending order)
        by the strength of the response. Only kp_max_num is kept for
        further processing.
        Set this value to -1 if you want to keep ALL keypoints (may take forever to process!)
    
    Returns:
    fnms : str array
        array of paths to the files containing Key-Points and Descriptors
    '''
    data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax
    sliding_minmax = kwargs.get("sliding_minmax", True)
    use_DASK = kwargs.get("use_DASK", False)
    if sliding_minmax:
        params_s3 = [[dts3[0], dts3[1], dts3[2], kwargs] for dts3 in zip(fls, data_min_sliding, data_max_sliding)]
    else:
        params_s3 = [[fl, data_min_glob, data_max_glob, kwargs] for fl in fls]        
    if use_DASK:
        print('Using DASK distributed')
        futures_s3 = DASK_client.map(extract_keypoints_descr_files, params_s3)
        fnms = DASK_client.gather(futures_s3)
    else:
        print('Using Local Computation')
        fnms = []
        for j, param_s3 in enumerate(tqdm(params_s3, desc='Extracting Key Points and Descriptors: ')):
            fnms.append(extract_keypoints_descr_files(param_s3))
    return fnms


def estimate_kpts_transform_error(src_pts, dst_pts, transform_matrix):
    """ 
    Estimate the transformation error for key-point pairs and known transformation matrix.
    ©G.Shtengel, 09/2021. gleb.shtengel@gmail.com

    Image transformation matrix in a form:
        A = [[a0  a1   a2]
             [b0   b1  b2]
             [0   0    1]]
     Thransofrmation is supposed to be in a form:
     Xnew = a0 * Xoriginal + a1 * Yoriginal + a2
     Ynew = b0 * Xoriginal + b1 * Yoriginal + b2
     source and destination points are pairs of coordinates (2xN array)
     errors are estimated as norm(dest_pts - A*src_pts) so that least square regression can be performed

    Returns:
        np.linalg.norm(dst_pts - src_pts_transformed, ord=2, axis=1)
    """
    src_pts_transformed = src_pts @ transform_matrix[0:2, 0:2].T + transform_matrix[0:2, 2]
    return np.linalg.norm(dst_pts - src_pts_transformed, ord=2, axis=1)


def determine_transformation_matrix(src_pts, dst_pts, TransformType, drmax = 2, max_iter = 100):
    '''
    Determine the transformation matrix.
    ©G.Shtengel, 09/2021. gleb.shtengel@gmail.com

    Determine the transformation matrix in a form:
            A = [[a0  a1   a2]
                 [b0   b1  b2]
                 [0   0    1]]
    based on the given source and destination points using linear regression such that the error is minimized for 
    sum(dst_pts - A*src_pts).
    
    For each matched pair of keypoins the error is calculated as err[j] = dst_pts[j] - A*src_pts[j]
    The iterative procedure throws away the matched keypoint pair with worst error on every iteration
    untill the worst error falls below drmax or the max number of iterations is reached.

    Returns
    transform_matrix, kpts, error_abs_mean, iteration
    '''
    transform_matrix = np.eye(3,3)
    iteration = 1
    max_error = drmax * 2.0
    errors = []
    while iteration <= max_iter and max_error > drmax:
        # determine the new transformation matrix     
        if TransformType == ShiftTransform:
            transform_matrix[0:2, 2] = np.mean(np.array(dst_pts.astype(np.float) - src_pts.astype(np.float)), axis=0)
            
        if TransformType == XScaleShiftTransform:
            n, d = src_pts.shape
            xsrc = np.array(src_pts)[:,0].astype(np.float)
            ysrc = np.array(src_pts)[:,1].astype(np.float)
            xdst = np.array(dst_pts)[:,0].astype(np.float)
            ydst = np.array(dst_pts)[:,1].astype(np.float)
            s00 = np.sum(xsrc)
            s01 = np.sum(xdst)
            sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xsrc*xsrc) - s00*s00)
            #sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xdst*xdst) - s01*s01)
            #s10 = np.sum(ysrc)
            #s11 = np.sum(ydst)
            #sy = (n*np.dot(ysrc, ydst) - s10*s11)/(n*np.sum(ysrc*ysrc) - s10*s10)
            sy = 1.00 # force sy=1 if there are not enough keypoints points
            # spread over wide range of y-range (and y-range is small) to determine y-scale accuartely
            tx = np.mean(xdst) - sx * np.mean(xsrc)
            ty = np.mean(ydst) - sy * np.mean(ysrc)
            transform_matrix = np.array([[sx, 0,  tx],
                                         [0,  sy, ty],
                                         [0,  0,  1]])
            
        if TransformType == ScaleShiftTransform:
            n, d = src_pts.shape
            xsrc = np.array(src_pts)[:,0].astype(np.float)
            ysrc = np.array(src_pts)[:,1].astype(np.float)
            xdst = np.array(dst_pts)[:,0].astype(np.float)
            ydst = np.array(dst_pts)[:,1].astype(np.float)
            s00 = np.sum(xsrc)
            s01 = np.sum(xdst)
            sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xsrc*xsrc) - s00*s00)
            s10 = np.sum(ysrc)
            s11 = np.sum(ydst)
            sy = (n*np.dot(ysrc, ydst) - s10*s11)/(n*np.sum(ysrc*ysrc) - s10*s10)
            tx = np.mean(xdst) - sx * np.mean(xsrc)
            ty = np.mean(ydst) - sy * np.mean(ysrc)
            transform_matrix = np.array([[sx, 0,  tx],
                                         [0,  sy, ty],
                                         [0,  0,  1]])
        
        if TransformType == AffineTransform:
            #estimate_scale = True
            #transform_matrix = _umeyama(src_pts, dst_pts, estimate_scale)
            tform = AffineTransform()
            tform.estimate(src_pts, dst_pts)
            transform_matrix = tform.params

        if TransformType == RegularizedAffineTransform:
            tform = AffineTransform()
            tform.estimate(src_pts, dst_pts)  # regularization parameters are already part of estimate procedure 
            # this is implemented this way because the other code - RANSAC does not work otherwise
            transform_matrix = tform.params
        
        # estimate transformation errors and find outliers
        errs = estimate_kpts_transform_error(src_pts, dst_pts, transform_matrix)
        max_error = np.max(errs)
        ind = np.argmax(errs)
        src_pts = np.delete(src_pts, ind, axis=0)
        dst_pts = np.delete(dst_pts, ind, axis=0)
        #print('Iteration {:d}, max_error={:.2f} '.format(iteration, max_error), (iteration <= max_iter), (max_error > drmax))
        iteration +=1
    kpts = [src_pts, dst_pts]
    error_abs_mean = np.mean(np.abs(np.delete(errs, ind, axis=0)))
    return transform_matrix, kpts, error_abs_mean, iteration


def determine_transformations_files(params_dsf):
    ''' Determine the transformation matrix from two sets of Key-Points and Descriptors.
    ©G.Shtengel, 09/2021. gleb.shtengel@gmail.com

    This is a faster version of the procedure - it loads the keypoints and matches for each frame from files.
    params_dsf = fnm_1, fnm_2, kwargs
    where 
    fnm_1 - keypoints for the first image (source)
    fnm_2 - keypoints for the first image (destination)
    and kwargs must include:
    TransformType - transformation type to be used (ShiftTransform, XScaleShiftTransform, ScaleShiftTransform, AffineTransform, RegularizedAffineTransform)
    BF_Matcher -  if True - use BF matcher, otherwise use FLANN matcher for keypoint matching
    solver - a string indicating which solver to use:
    'LinReg' will use Linear Regression with iterative "Throwing out the Worst Residual" Heuristic
    'RANSAC' will use RANSAC (Random Sample Consensus) algorithm.
    Lowe_Ratio_Threshold - threshold for Lowe's Ratio Test
    drmax - in the case of 'LinReg' - outlier threshold for iterative regression
           - in the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
    max_iter - max number of iterations
    save_matches - if True - save the matched keypoints into a binary dump file

    Returns:
    transform_matrix, fnm_matches, kpts, error_abs_mean, iteration
    '''
    fnm_1, fnm_2, kwargs = params_dsf

    ftype = kwargs.get("ftype", 0)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
    kp_max_num = kwargs.get("kp_max_num", -1)
    Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)    # threshold for Lowe's Ratio Test

    if TransformType == RegularizedAffineTransform:

        def estimate(self, src, dst):
            self.params = determine_regularized_affine_transform(src, dst, l2_matrix, targ_vector)
        RegularizedAffineTransform.estimate = estimate

    kpp1s, des1 = pickle.load(open(fnm_1, 'rb'))
    kpp2s, des2 = pickle.load(open(fnm_2, 'rb'))
    
    kp1 = [list_to_kp(kpp1) for kpp1 in kpp1s]     # this converts a list of lists to a list of keypoint objects to be used by a matcher later
    kp2 = [list_to_kp(kpp2) for kpp2 in kpp2s]     # same for the second frame
    
    # establish matches
    if BFMatcher:    # if BFMatcher==True - use BF (Brute Force) matcher
        # This procedure uses BF (Brute-Force) Matcher.
        # BF matcher takes the descriptor of one feature in the first image and matches it with all other features
        # in second image using some distance calculation. The closest match in teh second image is returned.
        # For BF matcher, first we have to create the cv.DescriptorMatcher object with BFMatcher as type.
        # It takes two optional params:
        #
        # First parameter one is NormType. It specifies the distance measurement to be used. By default, it is L2.
        # It is good for SIFT, SURF, etc. (L1 is also there).
        # For binary string-based descriptors like ORB, BRIEF, BRISK, etc., Hamming should be used,
        # which uses Hamming distance as measurement. If ORB is using WTA_K of 3 or 4, Hamming2 should be used.
        #
        # Second parameter is boolean variable, CrossCheck which is false by default.
        # If it is true, Matcher returns only those matches with value (i,j)
        # such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa.
        # That is, the two features in both sets should match each other.
        # It provides consistant result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.
        # http://amroamroamro.github.io/mexopencv/opencv_contrib/SURF_descriptor.html
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
    else:            # otherwise - use FLANN matcher
        # This procedure uses FLANN (Fast Library for Approximate Nearest Neighbors) Matcher (FlannBasedMatcher):
        # https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        #
        # FLANN contains a collection of algorithms optimized for fast nearest neighbor search in large datasets
        # and for high dimensional features. It works faster than BFMatcher for large datasets.
        # For FlannBasedMatcher, it accepts two sets of options which specifies the algorithm to be used, its related parameters etc.
        # First one is Index. For various algorithms, the information to be passed is explained in FLANN docs.
        # http://amroamroamro.github.io/mexopencv/opencv_contrib/SURF_descriptor.html
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
    
    if solver == 'LinReg':
        # Determine the transformation matrix via iterative liear regression
        transform_matrix, kpts, error_abs_mean, iteration = determine_transformation_matrix(src_pts, dst_pts, TransformType, drmax = drmax, max_iter = max_iter)
        n_kpts = len(kpts[0])
    else:  # the other option is solver = 'RANSAC'
        try:
            min_samples = len(src_pts)//20
            model, inliers = ransac((src_pts, dst_pts),
                TransformType, min_samples=min_samples,
                residual_threshold=drmax, max_trials=10000)
            n_inliers = np.sum(inliers)
            inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
            inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
            src_pts_ransac = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            dst_pts_ransac = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            #non_nan_inds = ~np.isnan(src_pts_ransac) * ~np.isnan(dst_pts_ransac)
            #src_pts_ransac = src_pts_ransac[non_nan_inds]
            #dst_pts_ransac = dst_pts_ransac[non_nan_inds]
            kpts = [src_pts_ransac, dst_pts_ransac]
            # find shift parameters
            transform_matrix = model.params
            iteration = len(src_pts)- len(src_pts_ransac)
            error_abs_mean = np.mean(np.abs(estimate_kpts_transform_error(src_pts_ransac, dst_pts_ransac, transform_matrix)))
        except:
            transform_matrix = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            kpts = [[], []]
            error_abs_mean = np.nan
            iteration = np.nan
    if save_matches:
        fnm_matches = fnm_2.replace('_kpdes.bin', '_matches.bin')
        pickle.dump(kpts, open(fnm_matches, 'wb'))
    else:
        fnm_matches = ''
    return transform_matrix, fnm_matches, kpts, error_abs_mean, iteration


def determine_transformations_dataset(fnms, DASK_client, **kwargs):
    use_DASK = kwargs.get("use_DASK", False)
    params_s4 = []
    for j, fnm in enumerate(fnms[:-1]):
        fname1 = fnms[j]
        fname2 = fnms[j+1]
        params_s4.append([fname1, fname2, kwargs])
    if use_DASK:
        print('Using DASK distributed')
        futures4 = DASK_client.map(determine_transformations_files, params_s4)
        #determine_transformations_files returns (transform_matrix, fnm_matches, kpts, iteration)
        results_s4 = DASK_client.gather(futures4)
    else:
        print('Using Local Computation')
        results_s4 = []
        for param_s4 in tqdm(params_s4, desc = 'Extracting Transformation Parameters: '):
            results_s4.append(determine_transformations_files(param_s4))
    return results_s4



def transform_and_save_single_frame(params_tss):
    '''
    Transforms a single EM frame with known transformation parameters and saves the new image into TIF file

    Parameters:
    params_tss : list
    params_tss = fname_orig, fname_transformed, tr_matrix, tr_args
        fname_orig : str
            Filename (full path) of the original image 
        fname_transformed : str
            Filename (full path) of the transformed image
        tr_matrix : 2d array
            Transformation matrix
    tr_args : list
        tr_args = ImgB_fraction, xsz, ysz, xi, xa, yi, ya, int_order, invert_data, perfrom_transformation, save_asI8, data_min_glob, data_max_glob, ftype, dtp


    Returns:
    fname_transformed : str
        Filename of the transformed image
    '''
    fname_orig, fname_transformed, tr_matrix, tr_args = params_tss
    ImgB_fraction, xsz, ysz, xi, xa, yi, ya, int_order, invert_data, perfrom_transformation, save_asI8, data_min_glob, data_max_glob, ftype, dtp = tr_args
    EMimage_padded = np.zeros((ysz, xsz), dtype=dtp)
    EMframe = FIBSEM_frame(fname_orig, ftype=ftype)

    if ImgB_fraction < 1e-5:
        EMimage = EMframe.RawImageA
    else:
        EMimage = EMframe.RawImageA * (1.0 - ImgB_fraction) + EMframe.RawImageB * ImgB_fraction

    if invert_data:
        if EMframe.EightBit==0:
            EMimage_padded[yi:ya, xi:xa]  = np.negative(EMimage)
        else:
            EMimage_padded[yi:ya, xi:xa]  =  uint8(255) - EMimage    
    else:
        EMimage_padded[yi:ya, xi:xa]  = EMimage
    if perfrom_transformation:
        transf = ProjectiveTransform(matrix = tr_matrix)
        EMimage_transformed = warp(EMimage_padded, transf, order = int_order, preserve_range=True)
    else:
        EMimage_transformed = EMimage_padded
    if save_asI8:
        EMimage_transformed_clipped = np.clip(((EMimage_transformed - data_min_glob)/(data_max_glob - data_min_glob)*255.0), 0, 255)
    else:
        EMimage_transformed_clipped = EMimage_transformed
    tiff.imsave(fname_transformed, EMimage_transformed_clipped.astype(dtp))
    return fname_transformed


def build_filename(fname, **kwargs):
    ftype = kwargs.get("ftype", 0)
    threshold_min = kwargs.get("threshold_min", 1e-3)
    threshold_max = kwargs.get("threshold_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    kp_max_num = kwargs.get("kp_max_num", -1)
    save_res_png  = kwargs.get("save_res_png", True)

    save_asI8 =  kwargs.get("save_asI8", False) 
    zbin_2x =  kwargs.get("zbin_2x", True)                  # If true, the data will be converted to I8 using global MIN and MAX values determined in the Step 1
    preserve_scales =  kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params =  kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    subtract_linear_fit =  kwargs.get("subtract_linear_fit", [True, True])   # If True, the linear slope will be subtracted from the cumulative shifts.
    pad_edges =  kwargs.get("pad_edges", True)
    suffix =  kwargs.get("suffix", '')

    frame = FIBSEM_frame(fname, ftype=ftype)
    dformat_read = 'I8' if frame.EightBit else 'I16'
    save_asI8_save = save_asI8 or frame.EightBit==1

    if save_asI8_save:
        dtp = int8
        dformat_save = 'I8'
        mrc_mode = 0
        if zbin_2x:
            fnm_reg = 'Registered_I8_zbin2x.mrc'
        else:
            fnm_reg = 'Registered_I8.mrc'
    else:
        dtp = int16
        dformat_save = 'I16'
        mrc_mode = 1
        if zbin_2x:
            fnm_reg = 'Registered_I16_zbin2x.mrc'
        else:
            fnm_reg = 'Registered_I16.mrc'
            
    fnm_reg = fnm_reg.replace('.mrc', ('_' + TransformType.__name__ + '_' + solver + '.mrc'))

    fnm_reg = fnm_reg.replace('.mrc', '_drmax{:.1f}.mrc'.format(drmax))
  
    if preserve_scales:
        fnm_reg = fnm_reg.replace('.mrc', '_const_scls_'+fit_params[0]+'.mrc')

    if np.any(subtract_linear_fit):
        fnm_reg = fnm_reg.replace('.mrc', '_shift_subtr.mrc')
    
    if pad_edges:
        fnm_reg = fnm_reg.replace('.mrc', '_padded.mrc')

    if len(suffix)>0:
        fnm_reg = fnm_reg.replace('.mrc', '_' + suffix + '.mrc')
    return fnm_reg, mrc_mode, dtp

def find_fit(tr_matr_cum, fit_params):
    fit_method = fit_params[0]
    if fit_method == 'SG':  # perform Savitsky-Golay fitting with parameters
        ws, porder = fit_params[1:3]         # window size 701, polynomial order 3
        s00_fit = savgol_filter(tr_matr_cum[:, 0, 0].astype(double), ws, porder)
        s01_fit = savgol_filter(tr_matr_cum[:, 0, 1].astype(double), ws, porder)
        s10_fit = savgol_filter(tr_matr_cum[:, 1, 0].astype(double), ws, porder)
        s11_fit = savgol_filter(tr_matr_cum[:, 1, 1].astype(double), ws, porder)
    else:
        fr = np.arange(0, len(tr_matr_cum), dtype=np.double)
        if fit_method == 'PF':  # perform polynomial fitting with parameters
            porder = fit_params[1]         # polynomial order
            s00_coeffs = np.polyfit(fr, tr_matr_cum[:, 0, 0].astype(double), porder)
            s00_fit = np.polyval(s00_coeffs, fr)
            s01_coeffs = np.polyfit(fr, tr_matr_cum[:, 0, 1].astype(double), porder)
            s01_fit = np.polyval(s01_coeffs, fr)
            s10_coeffs = np.polyfit(fr, tr_matr_cum[:, 1, 0].astype(double), porder)
            s10_fit = np.polyval(s10_coeffs, fr)
            s11_coeffs = np.polyfit(fr, tr_matr_cum[:, 1, 1].astype(double), porder)
            s11_fit = np.polyval(s11_coeffs, fr)
        
        else:   # otherwise perform linear fit with origin point tied to 1 for Sxx and Syy and to 0 for Sxy and Syx
            slp00 = -1.0 * (np.sum(fr)-np.dot(tr_matr_cum[:, 0, 0],fr))/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
            s00_fit = 1.0 + slp00 * fr
            slp11 = -1.0 * (np.sum(fr)-np.dot(tr_matr_cum[:, 1, 1],fr))/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
            s11_fit = 1.0 + slp11 * fr
            slp01 = np.dot(tr_matr_cum[:, 0, 1],fr)/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
            s01_fit = slp01 * fr
            slp10 = np.dot(tr_matr_cum[:, 1, 0],fr)/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
            s10_fit = slp10 * fr
        
    tr_matr_cum_new = tr_matr_cum.copy()
    tr_matr_cum_new[:, 0, 0] = tr_matr_cum[:, 0, 0].astype(double) + 1.0 - s00_fit
    tr_matr_cum_new[:, 0, 1] = tr_matr_cum[:, 0, 1].astype(double) - s01_fit
    tr_matr_cum_new[:, 1, 0] = tr_matr_cum[:, 1, 0].astype(double) - s10_fit
    tr_matr_cum_new[:, 1, 1] = tr_matr_cum[:, 1, 1].astype(double) + 1.0 - s11_fit
    s_fits = [s00_fit, s01_fit, s10_fit, s11_fit]
    return tr_matr_cum_new, s_fits

def process_transf_matrix(transformation_matrix, fnms_matches, npts, error_abs_mean, **kwargs):
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    Sample_ID = kwargs.get("Sample_ID", '')

    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
    kp_max_num = kwargs.get("kp_max_num", -1)
    save_res_png  = kwargs.get("save_res_png", True)

    preserve_scales =  kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params =  kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    subtract_linear_fit =  kwargs.get("subtract_linear_fit", [True, True])   # The linear slopes along X- and Y- directions (respectively) will be subtracted from the cumulative shifts.
    #print("subtract_linear_fit:", subtract_linear_fit)
    pad_edges =  kwargs.get("pad_edges", True)

    tr_matr_cum = transformation_matrix.copy()   
    prev_mt = np.eye(3,3)
    for j, cur_mt in enumerate(tqdm(transformation_matrix, desc='Calculating Original Cummilative Transformation Matrix')):
        prev_mt = np.matmul(cur_mt, prev_mt)
        tr_matr_cum[j] = prev_mt
    # Now insert identity matrix for the zero frame which does not need to be trasformed
    tr_matr_cum = np.insert(tr_matr_cum, 0, np.eye(3,3), axis=0)  
    
    fr = np.arange(0, len(tr_matr_cum), dtype=np.double)
    s00_cum_orig = tr_matr_cum[:, 0, 0].astype(np.double)
    s01_cum_orig = tr_matr_cum[:, 0, 1].astype(np.double)
    s10_cum_orig = tr_matr_cum[:, 1, 0].astype(np.double)
    s11_cum_orig = tr_matr_cum[:, 1, 1].astype(np.double)
    Xshift_cum_orig = tr_matr_cum[:, 0, 2].astype(np.double)
    Yshift_cum_orig = tr_matr_cum[:, 1, 2].astype(np.double)
    
    fs = 14
    lwf = 2
    lwl = 1
    fig5, axs5 = subplots(4,3, figsize=(18, 16), sharex=True)
    fig5.subplots_adjust(left=0.07, bottom=0.03, right=0.99, top=0.95)
    # display the info
    axs5[0,0].axis(False)
    axs5[0,0].text(-0.1, 0.9, Sample_ID, fontsize = fs + 4)
    #axs5[0,0].text(-0.1, 0.73, 'Global Data Range:  Min={:.2f}, Max={:.2f}'.format(data_min_glob, data_max_glob), transform=axs5[0,0].transAxes, fontsize = fs)
    
    if TransformType == RegularizedAffineTransform:
        tstr = ['{:d}'.format(x) for x in targ_vector] 
        otext = 'Reg.Aff.Transf., λ= {:.1e}, t=['.format(l2_matrix[0,0]) + ' '.join(tstr) + '], w/' + solver
    else:
        otext = TransformType.__name__ + ' with ' + solver + ' solver'

    axs5[0,0].text(-0.1, 0.56, otext, transform=axs5[0,0].transAxes, fontsize = fs)

    sbtrfit = ('ON, ' if  subtract_linear_fit[0] else 'OFF, ') + ('ON' if  subtract_linear_fit[1] else 'OFF')
    axs5[0,0].text(-0.1, 0.39, 'drmax={:.1f}, Max # of KeyPts={:d}, Max # of Iter.={:d}'.format(drmax, kp_max_num, max_iter), transform=axs5[0,0].transAxes, fontsize = fs)
    padedges = 'ON' if pad_edges else 'OFF'
    if preserve_scales:
        fit_method = fit_params[0]
        if fit_method == 'LF':
            fit_str = ', Meth: Linear Fit'
            fm_string = 'linear'
        else:
            if fit_method == 'SG':
                fit_str = ', Meth: Sav.-Gol., ' + str(fit_params[1:])
                fm_string = 'Sav.-Gol.'
            else:
                fit_str = ', Meth: Pol.Fit, ord.={:d}'.format(fit_params[1])
                fm_string = 'polyn.'
        preserve_scales_string = 'Pres. Scls: ON' + fit_str
    else:
        preserve_scales_string = 'Preserve Scales: OFF'
    axs5[0,0].text(-0.1, 0.22, preserve_scales_string, transform=axs5[0,0].transAxes, fontsize = fs)
    axs5[0,0].text(-0.1, 0.05, 'Subtract Shift Fit: ' + sbtrfit + ', Pad Edges: ' + padedges, transform=axs5[0,0].transAxes, fontsize = fs)
    # plot number of keypoints
    axs5[0, 1].plot(npts, 'g', linewidth = lwl, label = '# of key-points per frame')
    axs5[0, 1].set_title('# of key-points per frame')
    axs5[0, 1].text(0.03, 0.2, 'Mean # of kpts= {:.0f}   Median # of kpts= {:.0f}'.format(np.mean(npts), np.median(npts)), transform=axs5[0, 1].transAxes, fontsize = fs-1)
    # plot Standard deviations
    axs5[0, 2].plot(error_abs_mean, 'magenta', linewidth = lwl, label = 'Mean Abs Error over keyponts per frame')
    axs5[0, 2].set_title('Mean Abs Error keyponts per frame')  
    axs5[0, 2].text(0.03, 0.2, 'Mean Abs Error= {:.3f}   Median Abs Error= {:.3f}'.format(np.mean(error_abs_mean), np.median(error_abs_mean)), transform=axs5[0, 2].transAxes, fontsize = fs-1)
    
    if preserve_scales:  # in case of ScaleShift Transform WITH scale perservation
        print('Recalculating the transformation matrix for preserved scales')

        tr_matr_cum, s_fits = find_fit(tr_matr_cum, fit_params)
        s00_fit, s01_fit, s10_fit, s11_fit = s_fits
        '''
        det_cum_intermed = (s00_cum_new*s11_cum_new - s01_cum_orig*s10_cum_orig)
        slp_det = -1.0 * (np.sum(fr)-np.dot(det_cum_intermed,fr))/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
        det_fit = 1.0 + slp_det * fr
        tr_matr_cum[:, 0, 0] = tr_matr_cum[:, 0, 0]/det_fit
        tr_matr_cum[:, 0, 1] = tr_matr_cum[:, 0, 1]/det_fit
        tr_matr_cum[:, 1, 0] = tr_matr_cum[:, 1, 0]/det_fit
        tr_matr_cum[:, 1, 1] = tr_matr_cum[:, 1, 1]/det_fit
        '''
        
        txs = np.zeros(len(tr_matr_cum), dtype=float)
        tys = np.zeros(len(tr_matr_cum), dtype=float)
        
        for j, fnm_matches in enumerate(tqdm(fnms_matches, desc='Recalculating the shifts for preserved scales: ')):
            try:
                src_pts, dst_pts = pickle.load(open(fnm_matches, 'rb'))

                txs[j+1] = np.mean(tr_matr_cum[j, 0, 0] * dst_pts[:, 0] + tr_matr_cum[j, 0, 1] * dst_pts[:, 1]
                                   - tr_matr_cum[j+1, 0, 0] * src_pts[:, 0] - tr_matr_cum[j+1, 0, 1] * src_pts[:, 1])
                tys[j+1] = np.mean(tr_matr_cum[j, 1, 1] * dst_pts[:, 1] + tr_matr_cum[j, 1, 0] * dst_pts[:, 0]
                                   - tr_matr_cum[j+1, 1, 1] * src_pts[:, 1] - tr_matr_cum[j+1, 1, 0] * src_pts[:, 0])
            except:
                txs[j+1] = 0.0
                tys[j+1] = 0.0
        txs_cum = np.cumsum(txs)
        tys_cum = np.cumsum(tys)
        tr_matr_cum[:, 0, 2] = txs_cum
        tr_matr_cum[:, 1, 2] = tys_cum

    Xshift_cum = tr_matr_cum[:, 0, 2].copy()
    Yshift_cum = tr_matr_cum[:, 1, 2].copy()

    # Subtract linear trends from offsets
    if subtract_linear_fit[0]:
        fr = np.arange(0, len(Xshift_cum))
        pX = np.polyfit(fr, Xshift_cum, 1)
        Xfit = np.polyval(pX, fr)
        Xshift_residual = Xshift_cum - Xfit
        #Xshift_residual0 = -np.polyval(pX, 0.0)
    else:
        Xshift_residual = Xshift_cum.copy()

    if subtract_linear_fit[1]:
        fr = np.arange(0, len(Yshift_cum))
        pY = np.polyfit(fr, Yshift_cum, 1)
        Yfit = np.polyval(pY, fr)
        Yshift_residual = Yshift_cum - Yfit
        #Yshift_residual0 = -np.polyval(pY, 0.0)
    else:
        Yshift_residual = Yshift_cum.copy()

    # define new cumulative transformation matrix where the offests may have linear slopes subtracted
    tr_matr_cum[:, 0, 2] = Xshift_residual
    tr_matr_cum[:, 1, 2] = Yshift_residual
    
    # plot scales terms
    axs5[1, 0].plot(transformation_matrix[:, 0, 0], 'r', linewidth = lwl, label = 'Sxx frame-to-frame')
    axs5[1, 0].plot(transformation_matrix[:, 1, 1], 'b', linewidth = lwl, label = 'Syy frame-to-frame')
    axs5[1, 0].set_title('Frame-to-Frame Scale Change', fontsize = fs)
    axs5[2, 0].plot(s00_cum_orig, 'r', linewidth = lwl, linestyle='dotted', label = 'Sxx cum.')
    axs5[2, 0].plot(s11_cum_orig, 'b', linewidth = lwl, linestyle='dotted', label = 'Syy cum.')
    if preserve_scales:
        axs5[2, 0].plot(s00_fit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Sxx cum. - '+fm_string+' fit')
        axs5[2, 0].plot(s11_fit, 'cyan', linewidth = lwf, linestyle='dashed', label = 'Syy cum. - '+fm_string+' fit')
    axs5[2, 0].set_title('Cumulative Scale', fontsize = fs)
    yi10,ya10 = axs5[1, 0].get_ylim()
    dy0 = (ya10-yi10)/2.0
    yi20,ya20 = axs5[2, 0].get_ylim()
    if (ya20-yi20)<0.01*dy0:
        axs5[2, 0].set_ylim((yi20-dy0, ya20+dy0))
    axs5[3, 0].plot(tr_matr_cum[:, 0, 0], 'r', linewidth = lwl, label = 'Sxx cum. - residual')
    axs5[3, 0].plot(tr_matr_cum[:, 1, 1], 'b', linewidth = lwl, label = 'Syy cum. - residual')
    axs5[3, 0].set_title('Residual Cumulative Scale', fontsize = fs)
    axs5[3, 0].set_xlabel('Frame', fontsize = fs+1)
    yi30,ya30 = axs5[3, 0].get_ylim()
    if (ya30-yi30)<0.01*dy0:
        axs5[3, 0].set_ylim((yi30-dy0, ya30+dy0))

    # plot shear terms
    axs5[1, 1].plot(transformation_matrix[:, 0, 1], 'r', linewidth = lwl, label = 'Sxy frame-to-frame')
    axs5[1, 1].plot(transformation_matrix[:, 1, 0], 'b', linewidth = lwl, label = 'Syx frame-to-frame')
    axs5[1, 1].set_title('Frame-to-Frame Shear Change', fontsize = fs)
    axs5[2, 1].plot(s01_cum_orig, 'r', linewidth = lwl, linestyle='dotted', label = 'Sxy cum.')
    axs5[2, 1].plot(s10_cum_orig, 'b', linewidth = lwl, linestyle='dotted', label = 'Syx cum.')
    if preserve_scales:
        axs5[2, 1].plot(s01_fit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Sxy cum. - '+fm_string+' fit')
        axs5[2, 1].plot(s10_fit, 'cyan', linewidth = lwf, linestyle='dashed', label = 'Syx cum. - '+fm_string+' fit')
    axs5[2, 1].set_title('Cumulative Shear', fontsize = fs)
    yi11,ya11 = axs5[1, 1].get_ylim()
    dy1 = (ya11-yi11)/2.0
    yi21,ya21 = axs5[2, 1].get_ylim()
    if (ya21-yi21)<0.01*dy1:
        axs5[2, 1].set_ylim((yi21-dy1, ya21+dy1))
    axs5[3, 1].plot(tr_matr_cum[:, 0, 1], 'r', linewidth = lwl, label = 'Sxy cum. - residual')
    axs5[3, 1].plot(tr_matr_cum[:, 1, 0], 'b', linewidth = lwl, label = 'Syx cum. - residual')
    axs5[3, 1].set_title('Residual Cumulative Shear', fontsize = fs)
    axs5[3, 1].set_xlabel('Frame', fontsize = fs+1)
    yi31,ya31 = axs5[3, 1].get_ylim()
    if (ya31-yi21)<0.01*dy1:
        axs5[3, 1].set_ylim((yi31-dy1, ya31+dy1))

    # plot shifts
    axs5[1, 2].plot(transformation_matrix[:, 0, 2], 'r', linewidth = lwl, label = 'Tx fr.-to-fr.')
    axs5[1, 2].plot(transformation_matrix[:, 1, 2], 'b', linewidth = lwl, label = 'Ty fr.-to-fr.')
    axs5[1, 2].set_title('Frame-to-Frame Shift', fontsize = fs)
    if preserve_scales:
        axs5[2, 2].plot(Xshift_cum_orig, 'r', linewidth = lwl, label = 'Tx cum. - orig.')
        axs5[2, 2].plot(Yshift_cum_orig, 'b', linewidth = lwl, label = 'Ty cum. - orig.')
        axs5[2, 2].plot(Xshift_cum, 'r', linewidth = lwl, linestyle='dotted', label = 'Tx cum. - pres. scales')
        axs5[2, 2].plot(Yshift_cum, 'b', linewidth = lwl, linestyle='dotted', label = 'Ty cum. - pres. scales')
    else:
        axs5[2, 2].plot(Xshift_cum, 'r', linewidth = lwl, linestyle='dotted', label = 'Tx cum.')
        axs5[2, 2].plot(Yshift_cum, 'b', linewidth = lwl, linestyle='dotted', label = 'Ty cum.')
    if subtract_linear_fit[0]:
        axs5[2, 2].plot(Xfit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Tx cum. - lin. fit')
    if subtract_linear_fit[1]:
        axs5[2, 2].plot(Yfit, 'cyan', linewidth = lwf, linestyle='dashed', label = 'Ty cum. - lin. fit')
    axs5[2, 2].set_title('Cumulative Shift', fontsize = fs)
    axs5[3, 2].plot(tr_matr_cum[:, 0, 2], 'r', linewidth = lwl, label = 'Tx cum. - residual')
    axs5[3, 2].plot(tr_matr_cum[:, 1, 2], 'b', linewidth = lwl, label = 'Ty cum. - residual')
    axs5[3, 2].set_title('Residual Cumulative Shift', fontsize = fs)
    axs5[3, 2].set_xlabel('Frame', fontsize = fs+1)

    for ax in axs5.ravel()[1:]:
        ax.grid(True)
        ax.legend(fontsize = fs-1)
    fn = os.path.join(data_dir, fnm_reg)
    fig5.suptitle(fn, fontsize = fs)
    if save_res_png :
        fig5.savefig(fn.replace('.mrc', '_Transform_Summary.png'), dpi=300)
        
    return tr_matr_cum

def determine_pad_offsets(shape, tr_matr, disp_res):
    ysz, xsz = shape
    xmins = np.zeros(len(tr_matr))
    xmaxs = xmins.copy()
    ymins = xmins.copy()
    ymaxs = xmins.copy()
    corners = np.array([[0,0], [0, ysz], [xsz, 0], [xsz, ysz]])
    for j, trm in enumerate(tqdm(tr_matr, desc = 'Determining the pad offsets', disable=(not disp_res))):
        a = (trm[0:2, 0:2] @ corners.T).T + trm[0:2, 2]
        xmins[j] = np.min(a[:, 0])
        xmaxs[j] = np.max(a[:, 0])
        ymins[j] = np.min(a[:, 1])
        ymaxs[j] = np.max(a[:, 1])
    return np.min(xmins), np.max(xmaxs)-xsz, np.min(ymins), np.max(ymaxs)-ysz

# This is a function used for selecting proper threshold and kp_max_num parameters for SIFT processing

def SIFT_evaluation_dataset(fs, **kwargs):
    ftype = kwargs.get("ftype", 0)
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    threshold_min = kwargs.get("threshold_min", 1e-3)
    threshold_max = kwargs.get("threshold_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    kp_max_num = kwargs.get("kp_max_num", -1)
    Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
    BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
    save_res_png  = kwargs.get("save_res_png", True)

    frame = FIBSEM_frame(fs[0], ftype=ftype)
    if ftype == 0:
        if frame.FileVersion > 8 :
            Sample_ID = frame.Sample_ID.strip('\x00')
        else:
            Sample_ID = frame.Notes[0:16]
    else:
        Sample_ID = frame.Sample_ID
    Sample_ID = kwargs.get("Sample_ID", Sample_ID)

    print(Sample_ID)
    
    #if save_res_png :
    #    frame.display_images()

    img = np.ravel(frame.RawImageA)
    fsz=12
    fszl=14
    dmin, dmax = frame.get_image_min_max(image_name = 'RawImageA', thr_min=threshold_min, thr_max=threshold_max, nbins=nbins)
    xi = dmin-(np.abs(dmax-dmin)/10)
    xa = dmax+(np.abs(dmax-dmin)/10)

    fig, axs = subplots(2,2, figsize=(12,8))
    hist, bins, patches = axs[0,0].hist(img, bins = nbins)
    axs[0,0].set_xlim(xi, xa)
    axs[0,0].plot([dmin, dmin], [0, np.max(hist)], 'r', linestyle = '--')
    axs[0,0].plot([dmax, dmax], [0, np.max(hist)], 'g', linestyle = '--')
    axs[0,0].set_ylabel('Count', fontsize = fsz)
    pdf = hist / (frame.XResolution * frame.YResolution)
    cdf = np.cumsum(pdf)
    xCDF = bins[0:-1]+(bins[1]-bins[0])/2.0
    xthr = [xCDF[0], xCDF[-1]]
    ythr_min = [threshold_min, threshold_min]
    y1thr_max = [1-threshold_max, 1-threshold_max]

    axs[1,0].plot(xCDF, cdf, label='CDF')
    axs[1,0].plot(xthr, ythr_min, 'r', label='thr_min={:.5f}'.format(threshold_min))
    axs[1,0].plot([dmin, dmin], [0, 1], 'r', linestyle = '--', label = 'data_min={:.1f}'.format(dmin))
    axs[1,0].plot(xthr, y1thr_max, 'g', label='1.0 - thr_max = {:.5f}'.format(1-threshold_max))
    axs[1,0].plot([dmax, dmax], [0, 1], 'g', linestyle = '--', label = 'data_max={:.1f}'.format(dmax))
    axs[1,0].set_xlabel('Intensity Level', fontsize = fsz)
    axs[1,0].set_ylabel('CDF', fontsize = fsz)
    axs[1,0].set_xlim(xi, xa)
    axs[1,0].legend(loc='center', fontsize=fsz)
    axs[0,0].set_title('Data Min and Max with thr_min={:.0e},  thr_max={:.0e}'.format(threshold_min, threshold_max), fontsize = fsz)

    minmax = []
    for f in fs:
        minmax.append(FIBSEM_frame(f, ftype=ftype).get_image_min_max(image_name = 'RawImageA', thr_min=threshold_min, thr_max=threshold_max, nbins=nbins))
    dmin = np.min(np.array(minmax))
    dmax = np.max(np.array(minmax))
    #print('data range: ', dmin, dmax)
    
    t0 = time.time()
    params1 = [fs[0], dmin, dmax, kwargs]
    fnm_1 = extract_keypoints_descr_files(params1)
    params2 = [fs[1], dmin, dmax, kwargs]
    fnm_2 = extract_keypoints_descr_files(params2)
    
    params_dsf = [fnm_1, fnm_2, kwargs]
    transform_matrix, fnm_matches, kpts, error_abs_mean, iteration = determine_transformations_files(params_dsf)
    n_matches = len(kpts[0])
    
    src_pts_filtered, dst_pts_filtered = kpts
    
    src_pts_transformed = src_pts_filtered @ transform_matrix[0:2, 0:2].T + transform_matrix[0:2, 2]
    xshifts = (dst_pts_filtered - src_pts_transformed)[:,0]
    yshifts = (dst_pts_filtered - src_pts_transformed)[:,1]
    
    t1 = time.time()
    comp_time = (t1-t0)
    #print('Time to compute: {:.1f}sec'.format(comp_time))

    axx = axs[0,1]
    hst = axx.hist(xshifts, bins=64)
    axx.set_xlabel('SIFT: X Error (pixels)')
    axx.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(xshifts)), transform=axx.transAxes, fontsize=fsz)
    axx.text(0.05, 0.8, 'median={:.3f}'.format(np.median(xshifts)), transform=axx.transAxes, fontsize=fsz)
    axx.set_title('data range: {:.1f} ÷ {:.1f}'.format(dmin, dmax), fontsize=fsz)
    axy = axs[1,1]
    hst = axy.hist(yshifts, bins=64)
    axy.set_xlabel('SIFT: Y Error (pixels)')
    axy.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(yshifts)), transform=axy.transAxes, fontsize=fsz)
    axy.text(0.05, 0.8, 'median={:.3f}'.format(np.median(yshifts)), transform=axy.transAxes, fontsize=fsz)
    axt=axx  # print Transformation Matrix data over axx plot
    axt.text(0.65, 0.8, 'Transf. Matrix:', transform=axt.transAxes, fontsize=fsz)
    axt.text(0.55, 0.7, '{:.4f} {:.4f} {:.4f}'.format(transform_matrix[0,0], transform_matrix[0,1], transform_matrix[0,2]), transform=axt.transAxes, fontsize=fsz-1)
    axt.text(0.55, 0.6, '{:.4f} {:.4f} {:.4f}'.format(transform_matrix[1,0], transform_matrix[1,1], transform_matrix[1,2]), transform=axt.transAxes, fontsize=fsz-1)
    
    for ax in ravel(axs):
        ax.grid(True)
    
    fig.suptitle(Sample_ID + ',  thr_min={:.0e}, thr_max={:.0e}, kp_max_num={:d}, comp.time={:.1f}sec'.format(threshold_min, threshold_max, kp_max_num, comp_time), fontsize=fszl)

    if TransformType == RegularizedAffineTransform:
        tstr = ['{:d}'.format(x) for x in targ_vector] 
        otext =  TransformType.__name__ + ', λ= {:.1e}, t=['.format(l2_matrix[0,0]) + ', '.join(tstr) + '], ' + solver + ', #of matches={:d}'.format(n_matches)
    else:
        otext = TransformType.__name__ + ', ' + solver + ', #of matches={:d}'.format(n_matches)

    axs[0,0].text(0.01, 1.14, otext, fontsize=fszl, transform=axs[0,0].transAxes)        
    if save_res_png :
        png_name = os.path.splitext(fs[0])[0] + '_SIFT_eval_'+TransformType.__name__ + '_' + solver +'_thr_min{:.5f}_thr_max{:.5f}.png'.format(threshold_min, threshold_max) 
        fig.savefig(png_name, dpi=300)
            
    xfsz = np.int(7 * frame.XResolution / np.max([frame.XResolution, frame.YResolution]))+1
    yfsz = np.int(7 * frame.YResolution / np.max([frame.XResolution, frame.YResolution]))+2
    fig2, ax = subplots(1,1, figsize=(xfsz,yfsz))
    fig2.subplots_adjust(left=0.0, bottom=0.25*(1-frame.YResolution/frame.XResolution), right=1.0, top=1.0)
    symsize = 2
    fsize = 12  
    img2 = FIBSEM_frame(fs[-1], ftype=ftype).RawImageA
    ax.imshow(img2, cmap='Greys', vmin=dmin, vmax=dmax)
    ax.axis(False)
    x, y = dst_pts_filtered.T
    M = sqrt(xshifts*xshifts+yshifts*yshifts)
    xs = xshifts
    ys = yshifts

    # the code below is for vector map. vectors have origin coordinates x and y, and vector projections xs and ys.
    vec_field = ax.quiver(x,y,xs,ys,M, scale=40, width =0.003, cmap='jet')
    cbar = fig2.colorbar(vec_field, cmap='jet', pad=0.05, shrink=0.70, orientation = 'horizontal', format="%.1f")
    cbar.set_label('SIFT Shift Amplitude (pix)', fontsize=fsize)

    ax.text(0.01, 1.1-0.13*frame.YResolution/frame.XResolution, Sample_ID + ', thr_min={:.0e}, thr_max={:.0e}, kp_max_num={:d},  #of matches={:d}'.format(threshold_min, threshold_max, kp_max_num, n_matches), fontsize=fsize, transform=ax.transAxes)
            
    if save_res_png :
        fig2_fnm = os.path.join(data_dir, 'SIFT_vmap_'+TransformType.__name__ + '_' + solver +'_thr_min{:.0e}_thr_max{:.0e}_kp_max{:d}.png'.format(threshold_min, threshold_max, kp_max_num))
        fig2.savefig(fig2_fnm, dpi=300)

    return(dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts)


def save_inlens_data(fname):
    tfr = FIBSEM_frame(fname)
    tfr.save_images_tif('A')
    return fname

def calc_data_range_dataset(fls, DASK_client, **kwargs):
    nfrs = len(fls)
    use_DASK = kwargs.get("use_DASK", False)
    ftype = kwargs.get("ftype", 0)
    Sample_ID = kwargs.get("Sample_ID", '')
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    EightBit = kwargs.get("EightBit", 0)
    threshold_min = kwargs.get("threshold_min", 1e-3)
    threshold_max = kwargs.get("threshold_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    sliding_minmax = kwargs.get("sliding_minmax", True)
    save_res_png  = kwargs.get("save_res_png", True)
    fit_params =  kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    if EightBit == 1:
        print('Original data is 8-bit, no need to find Min and Max for 8-bit conversion')
        data_min_glob = uint8(0)
        data_max_glob =  uint8(255)
        data_min_sliding = np.zeros(nfrs, dtype=uint8)
        data_max_sliding = np.zeros(nfrs, dtype=uint8)+ uint8(255)
    else:
        params_s2 = [[fl, kwargs] for fl in fls]

        if use_DASK:
            print('Using DASK distributed')
            futures = DASK_client.map(get_min_max_thresholds_file, params_s2)
            data_minmax_glob = np.array(DASK_client.gather(futures))
        else:
            print('Using Local Computation')
            data_minmax_glob = np.zeros((nfrs, 2))
            for j, param_s2 in enumerate(tqdm(params_s2, desc='Calculating the Global Data Range: ')):
                data_minmax_glob[j, :] = get_min_max_thresholds_file(param_s2)

        #data_min_glob = np.min(data_minmax_glob)
        #data_max_glob = np.max(data_minmax_glob)
        data_min_glob, trash = get_min_max_thresholds(data_minmax_glob[:, 0], threshold_min, threshold_max, nbins) 
        trash, data_max_glob = get_min_max_thresholds(data_minmax_glob[:, 1], threshold_min, threshold_max, nbins)

        data_min_sliding = savgol_filter(data_minmax_glob[:, 0].astype(double), min([fit_params[1], fit_params[1]]), fit_params[2])
        data_max_sliding = savgol_filter(data_minmax_glob[:, 1].astype(double), min([fit_params[1], fit_params[1]]), fit_params[2])

        # if needed, change the global data range
        #data_min_glob = -4300
        #data_max_glob = -1500
        if save_res_png :
            fs = 12
            fig0, ax0 = subplots(1,1,figsize=(6,4))
            fig0.subplots_adjust(left=0.14, bottom=0.11, right=0.99, top=0.94)
            ax0.plot(data_minmax_glob[:, 0], 'b', linewidth=1, label='Frame Minima')
            ax0.plot(data_min_sliding, 'b', linewidth=2, linestyle = 'dotted', label='Sliding Minima')
            ax0.plot(data_minmax_glob[:, 1], 'r', linewidth=1, label='Frame Maxima')
            ax0.plot(data_max_sliding, 'r', linewidth=2, linestyle = 'dotted', label='Sliding Maxima')
            ax0.legend()
            ax0.grid(True)
            ax0.set_xlabel('Frame')
            ax0.set_ylabel('Minima and Maxima Values')
            dxn = (data_max_glob - data_min_glob)*0.1
            ax0.set_ylim((data_min_glob - dxn, data_max_glob+dxn))
            # if needed, display the data in a narrower range
            #ax0.set_ylim((-4500, -1500))
            xminmax = [0, len(data_minmax_glob)]
            y_min = [data_min_glob, data_min_glob]
            y_max = [data_max_glob, data_max_glob]
            ax0.plot(xminmax, y_min, 'b', linestyle = '--')
            ax0.plot(xminmax, y_max, 'r', linestyle = '--')
            ax0.text(len(data_minmax_glob)/20.0, data_min_glob-dxn/1.75, 'data_min_glob={:.1f}'.format(data_min_glob), fontsize = fs-2, c='b')
            ax0.text(len(data_minmax_glob)/20.0, data_max_glob+dxn/2.25, 'data_max_glob={:.1f}'.format(data_max_glob), fontsize = fs-2, c='r')
            ldm = 50
            data_dir_short = data_dir if len(data_dir)<ldm else '... '+ data_dir[-ldm:]  
            fig0.suptitle(Sample_ID + '    ' +  data_dir_short, fontsize = fs-2)
            fig0.savefig(os.path.join(data_dir, fnm_reg.replace('.mrc','_DataBounds.png')), dpi=300)
           
    return [data_min_glob, data_max_glob, data_min_sliding, data_max_sliding]


def transform_frame(frame, tr_matr, **kwargs):
    '''
    Transforms single frame

    Parameters:
        frame : instance of FIBSEM_frame object
        tr_matr : transformation matrix
    **kwargs:
        ImgB_fraction : Image B (ESB image) fraction for image fusion
        pad_edges : boolean
            default is True
        xy_limits : list of 4 indices: [xi, xa, yi, ya]  paddind range
            default is [0, -1, 0, -1]
        perfrom_transformation : boolean
            default is True
        invert_data : boolean
            default is False
        flatten_image : boolean
            default is False
        Image_correction : list of two full-size image corrections

    Returns:
        transformed_img : 2D array

    '''
    ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)         # fusion fraction. In case if Img B is present, the fused image 
                                                           # for each frame will be constructed ImgF = (1.0-ImgB_fraction)*ImgA + ImgB_fraction*ImgB
    if test_frame.DetB == 'None':
        ImgB_fraction=0.0
    pad_edges =  kwargs.get("pad_edges", True)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    int_order = kwargs.get("int_order", 1)                  # The order of interpolation. 1: Bi-linear
    perfrom_transformation =  kwargs.get("perfrom_transformation", True)
    invert_data =  kwargs.get("invert_data", False)
    xy_limits = kwargs.get("xy_limits", [0, -1, 0, -1])
    xi, xa, yi, ya = xy_limits
    flatten_image =  kwargs.get("flatten_image", False)
    Image_correction =  kwargs.get("Image_correction", [np.ones(frame.RawImageA.shape), np.ones(frame.RawImageA.shape)])

    if pad_edges and perfrom_transformation:
        shift_matrix = np.array([[1.0, 0.0, xi],
                                 [0.0, 1.0, yi],
                                 [0.0, 0.0, 1.0]])
        inv_shift_matrix = np.linalg.inv(shift_matrix)
    else:
        xi = 0
        yi = 0
        shift_matrix = np.eye(3,3)
        inv_shift_matrix = np.eye(3,3)

    if ImgB_fraction < 1e-5 or (not hasattr(frame, 'RawImageB')):
        if flatten_image:
            image = (frame.RawImageA.astype(float64) -  frame.Scaling[1,0])*Image_correction[0] + frame.Scaling[1,0]
        else:
            image = frame.RawImageA.astype(float64)

    else:
        if flatten_image:
            ImgA_flattened = (frame.RawImageA.astype(float64) -  frame.Scaling[1,0])*Image_correction[0] + frame.Scaling[1,0]
            ImgB_flattened = (frame.RawImageB.astype(float64) -  frame.Scaling[1,1])*Image_correction[1] + frame.Scaling[1,1]
            image = ImgA_flattened * (1.0 - ImgB_fraction) + ImgB_flattened * ImgB_fraction
        else:
            image = frame.RawImageA * (1.0 - ImgB_fraction) + frame.RawImageB * ImgB_fraction
        
    if invert_data:
        if test_frame.EightBit==0:
            padded_img[yi:ya, xi:xa] = np.negative(image)

        else:
            padded_img[yi:ya, xi:xa]  =  uint8(255) - image   
    else:
        padded_img[yi:ya, xi:xa]  = image

    if perfrom_transformation:
        transf = ProjectiveTransform(matrix = shift_matrix @ (tr_matr @ inv_shift_matrix))
        transformed_img = warp(padded_img, transf, order = int_order,  preserve_range=True)

    else:
        transformed_img = padded_img

    return transformed_img



def transform_and_save_dataset(save_transformed_dataset, frame_inds, fls, tr_matr_cum_residual, data_minmax, npts, error_abs_mean, **kwargs):
    ftype = kwargs.get("ftype", 0)
    data_dir = kwargs.get("data_dir", '')
    test_frame = FIBSEM_frame(fls[0], ftype=ftype)
    ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)         # fusion fraction. In case if Img B is present, the fused image 
                                                           # for each frame will be constructed ImgF = (1.0-ImgB_fraction)*ImgA + ImgB_fraction*ImgB
    if test_frame.DetB == 'None':
        ImgB_fraction=0.0
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    Sample_ID = kwargs.get("Sample_ID", '')
    pad_edges =  kwargs.get("pad_edges", True)
    save_res_png  = kwargs.get("save_res_png", True)
    save_asI8 =  kwargs.get("save_asI8", False)
    dtp = kwargs.get("dtp", int16)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    mrc_mode = kwargs.get("mrc_mode", 0) 
    zbin_2x =  kwargs.get("zbin_2x", True)
    int_order = kwargs.get("int_order", 1)                  # The order of interpolation. 1: Bi-linear
    preserve_scales =  kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params =  kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    subtract_linear_fit =  kwargs.get("subtract_linear_fit", [True, True])   # If True, the linear slope will be subtracted from the cumulative shifts.
    perfrom_transformation =  kwargs.get("perfrom_transformation", True)
    invert_data =  kwargs.get("invert_data", False)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])  #  [top, height, keft, width]
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    disp_res = kwargs.get("disp_res", True)

    dformat_save = 'I8' if mrc_mode==0 else 'I16'
    
    if invert_data:
        if test_frame.EightBit==0:
            data_max_glob, data_min_glob, data_max_sliding, data_min_sliding = np.negative(data_minmax)
        else:
            data_max_glob, data_min_glob, data_max_sliding, data_min_sliding = [ uint8(255) - x for x in data_minmax]
    else:
        data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax

    if pad_edges and perfrom_transformation:
        shape = [test_frame.YResolution, test_frame.XResolution]
        xmn, xmx, ymn, ymx = determine_pad_offsets(shape, tr_matr_cum_residual, disp_res)
        padx = int(xmx - xmn)
        pady = int(ymx - ymn)
        xi = int(np.max([xmx, 0]))
        yi = int(np.max([ymx, 0]))
        # The initial transformation matrices are calculated with no padding.Padding is done prior to transformation
        # so that the transformed images are not clipped.
        # Such padding means shift (by xi and yi values). Therefore the new transformation matrix
        # for padded frames will be (Shift Matrix)x(Transformation Matrix)x(Inverse Shift Matrix)
        # those are calculated below base on the amount of padding calculated above
        shift_matrix = np.array([[1.0, 0.0, xi],
                                 [0.0, 1.0, yi],
                                 [0.0, 0.0, 1.0]])
        inv_shift_matrix = np.linalg.inv(shift_matrix)
    else:
        padx = 0
        pady = 0
        xi = 0
        yi = 0
        shift_matrix = np.eye(3,3)
        inv_shift_matrix = np.eye(3,3)
 
    fpath_reg = os.path.join(data_dir, fnm_reg)
    xsz = test_frame.XResolution + padx
    xa = xi + test_frame.XResolution
    ysz = test_frame.YResolution + pady
    ya = yi + test_frame.YResolution
    nfrs = len(fls)

    xi_eval = xi + evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = xa
    yi_eval = yi + evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ya

    frame0_img = np.zeros((ysz, xsz))
    frame1_img = frame0_img.copy()
    frame_img = frame0_img.copy()
    prev_frame_img = frame0_img.copy()

    image_nsad = np.zeros((len(frame_inds)-1), dtype=float)
    image_ncc = np.zeros((len(frame_inds)-1), dtype=float)
    image_mi = np.zeros((len(frame_inds)-1), dtype=float)

    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0

    if zbin_2x:
        if save_transformed_dataset:
            print('Saving the registered and 2x z-binned stack into the file: ', fpath_reg)
            # Make a new, empty memory-mapped MRC file
            mrc = mrcfile.new_mmap(fpath_reg, shape=(len(frame_inds)//2, ysz, xsz), mrc_mode=mrc_mode, overwrite=True)
            # mode 0 -> int8
            # mode 1 -> int16:
            mrc.voxel_size = np.round(test_frame.PixelSize)*0.001
            tq_desc = 'Saving into ' + dformat_save + ' MRC File'
        else:
            tq_desc = 'Processing frames'
        ind = np.arange(0,len(frame_inds)-1,2)
        for j in tqdm(ind, desc = tq_desc, disable=(not disp_res)):
            if ImgB_fraction < 1e-5:
                j0image = FIBSEM_frame(fls[frame_inds[j]], ftype=ftype).RawImageA.astype(float64)
                j1image = FIBSEM_frame(fls[frame_inds[j+1]], ftype=ftype).RawImageA.astype(float64)
            else:
                j0frame = FIBSEM_frame(fls[frame_inds[j]], ftype=ftype)
                j0image = j0frame.RawImageA * (1.0 - ImgB_fraction) + j0frame.RawImageB * ImgB_fraction
                j1frame = FIBSEM_frame(fls[frame_inds[j+1]], ftype=ftype)
                j1image = j1frame.RawImageA * (1.0 - ImgB_fraction) + j1frame.RawImageB * ImgB_fraction

            if invert_data:
                if test_frame.EightBit==0:
                    frame0_img[yi:ya, xi:xa] = np.negative(j0image)
                    frame1_img[yi:ya, xi:xa] = np.negative(j1image)
                else:
                    frame0_img[yi:ya, xi:xa]  =  uint8(255) - j0image
                    frame1_img[yi:ya, xi:xa]  =  uint8(255) - j1image    
            else:
                frame0_img[yi:ya, xi:xa]  = j0image
                frame1_img[yi:ya, xi:xa]  = j1image

            if perfrom_transformation:
                transf0 = ProjectiveTransform(matrix = shift_matrix @ (tr_matr_cum_residual[frame_inds[j]] @ inv_shift_matrix))
                frame0_img_reg = warp(frame0_img, transf0, order = int_order,  preserve_range=True)
                transf1 = ProjectiveTransform(matrix = shift_matrix @ (tr_matr_cum_residual[frame_inds[j+1]] @ inv_shift_matrix))
                frame1_img_reg = warp(frame1_img, transf1, order = int_order, preserve_range=True)
            else:
                frame0_img_reg = frame0_img.copy()
                frame1_img_reg = frame1_img.copy()
        
            if (mrc_mode==0 and test_frame.EightBit==0):
                curr_img = np.clip((((frame0_img_reg - data_min_glob)/2.0 + (frame1_img_reg - data_min_glob)/2.0)/(data_max_glob - data_min_glob)*255.0), 0, 255)
            else:
                curr_img = frame0_img_reg/2.0 + frame1_img_reg/2.0
            if save_transformed_dataset:
                mrc.data[j//2,:,:] = np.flip(curr_img.astype(dtp), axis=0)
            if sliding_evaluation_box:
                xi_eval = start_evaluation_box[2] + dx_eval*j//2//nfrs
                yi_eval = start_evaluation_box[0] + dy_eval*j//2//nfrs
                if start_evaluation_box[3] > 0:
                    xa_eval = xi_eval + start_evaluation_box[3]
                else:
                    xa_eval = xsz
                if start_evaluation_box[1] > 0:
                    ya_eval = yi_eval + start_evaluation_box[1]
                else:
                    ya_eval = ysz
            I1c = cp.array(frame0_img_reg[yi_eval:ya_eval, xi_eval:xa_eval])
            if j>0:
                I2c = cp.array(prev_frame_img[yi_eval:ya_eval, xi_eval:xa_eval])
                #image_nsad[j-1] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0 - data_min_glob))
                image_nsad[j-1] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0))
                image_ncc[j-1] = Two_Image_NCC_SNR(frame0_img_reg[yi_eval:ya_eval, xi_eval:xa_eval], prev_frame_img[yi_eval:ya_eval, xi_eval:xa_eval])[0]
                image_mi[j-1] = cp.asnumpy(mutual_information_2d_cp(I1c.ravel(), I2c.ravel(), sigma=1.0, bin=2048, normalized=True))
            I2c = cp.array(frame1_img_reg[yi_eval:ya_eval, xi_eval:xa_eval])
            image_nsad[j] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0 - data_min_glob))
            image_ncc[j] = Two_Image_NCC_SNR(frame0_img_reg[yi_eval:ya_eval, xi_eval:xa_eval], frame1_img_reg[yi_eval:ya_eval, xi_eval:xa_eval])[0]
            image_mi[j] = cp.asnumpy(mutual_information_2d_cp(I1c.ravel(), I2c.ravel(), sigma=1.0, bin=2048, normalized=True))
            prev_frame_img = frame1_img_reg
            del I1c, I2c
    else:
        # Execute this frame if 2x z-binning is NOT selected
        if save_transformed_dataset:
            print('Saving the registered stack into the file: ', fpath_reg)
            # Make a new, empty memory-mapped MRC file
            mrc = mrcfile.new_mmap(fpath_reg, shape=(len(frame_inds), ysz, xsz), mrc_mode=mrc_mode, overwrite=True)
            # mode 0 -> int8
            # mode 1 -> int16:
            mrc.voxel_size = np.round(test_frame.PixelSize)*0.001
            tq_desc = 'Saving into ' + dformat_save + ' MRC File'
        else:
            tq_desc = 'Processing frames'
        ind = np.arange(0,len(frame_inds))
        for j in tqdm(ind, desc = tq_desc, disable=(not disp_res)):
            if ImgB_fraction < 1e-5:
                j0image = FIBSEM_frame(fls[frame_inds[j]], ftype=ftype).RawImageA.astype(float64)
            else:
                j0frame = FIBSEM_frame(fls[frame_inds[j]], ftype=ftype)
                j0image = j0frame.RawImageA * (1.0 - ImgB_fraction) + j0frame.RawImageB * ImgB_fraction

            if invert_data:
                if test_frame.EightBit==0:
                    frame_img[yi:ya, xi:xa]  = np.negative(j0image)
                else:
                    frame_img[yi:ya, xi:xa]  =  uint8(255) - j0image    
            else:
                frame_img[yi:ya, xi:xa]  = j0image
            if perfrom_transformation:
                transf = ProjectiveTransform(matrix = shift_matrix @ (tr_matr_cum_residual[frame_inds[j]] @ inv_shift_matrix))
                frame_img_reg = warp(frame_img, transf, order = int_order, preserve_range=True)
            else:
                frame_img_reg = frame_img.copy()
            if (mrc_mode==0 and test_frame.EightBit==0):
                curr_img = np.clip(((frame_img_reg - data_min_glob)/(data_max_glob - data_min_glob)*255.0), 0, 255)
            else:
                curr_img = frame_img_reg
            if save_transformed_dataset:
                mrc.data[j,:,:] = np.flip(curr_img.astype(dtp), axis=0)
            if j>0:
                if sliding_evaluation_box:
                    xi_eval = start_evaluation_box[2] + dx_eval*j//nfrs
                    yi_eval = start_evaluation_box[0] + dy_eval*j//nfrs
                    if start_evaluation_box[3] > 0:
                        xa_eval = xi_eval + start_evaluation_box[3]
                    else:
                        xa_eval = xsz
                    if start_evaluation_box[1] > 0:
                        ya_eval = yi_eval + start_evaluation_box[1]
                    else:
                        ya_eval = ysz
                I1c = cp.array(frame_img_reg[yi_eval:ya_eval, xi_eval:xa_eval])
                I2c = cp.array(prev_frame_img[yi_eval:ya_eval, xi_eval:xa_eval])
                #image_nsad[j-1] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0 - data_min_glob))
                image_nsad[j-1] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0))
                #print('cp results: ',cp.mean(cp.abs(I1c-I2c)), cp.mean(I1c/2.0 + I2c/2.0))
                image_ncc[j-1] = Two_Image_NCC_SNR(prev_frame_img[yi_eval:ya_eval, xi_eval:xa_eval], frame_img_reg[yi_eval:ya_eval, xi_eval:xa_eval])[0]
                image_mi[j-1] = cp.asnumpy(mutual_information_2d_cp(I1c.ravel(), I2c.ravel(), sigma=1.0, bin=2048, normalized=True))
                del I1c, I2c
            prev_frame_img = frame_img_reg
    if save_transformed_dataset:
        mrc.close()

    # Generate a figure with analysis of registration quality - Image NSAD, NCC, NMI's vs. frames.
    #image_nsad = image_nsad[1:-1]
    nsads = [np.mean(image_nsad), np.median(image_nsad), np.std(image_nsad)] 
    #image_ncc = image_ncc[1:-1]
    nccs = [np.mean(image_ncc), np.median(image_ncc), np.std(image_ncc)]
    nmis = [np.mean(image_mi), np.median(image_mi), np.std(image_mi)]

    Pres_scales = 'ON' if  preserve_scales else 'OFF'
    Subtr_Lin_Fit = ('ON, ' if  subtract_linear_fit[0] else 'OFF, ') + ('ON' if  subtract_linear_fit[1] else 'OFF')
    Pad_Edg = 'ON' if  pad_edges else 'OFF'

    if disp_res:
        fs=12
        lwl=1
        if preserve_scales:
            fit_method = fit_params[0]
            if fit_method == 'LF':
                fit_str = ', Meth: Linear Fit'
                fm_string = 'linear'
            else:
                if fit_method == 'SG':
                    fit_str = ', Meth: Sav.-Gol., ' + str(fit_params[1:])
                    fm_string = 'Sav.-Gol.'
                else:
                    fit_str = ', Meth: ' + fit_method + ', ' + str(fit_params[1:])
                    fm_string = fit_method
            preserve_scales_string = 'Preserve Scales: ON' + fit_str
        else:
            preserve_scales_string = 'Preserve Scales: OFF'

        cond_str = TransformType.__name__ + ' with ' + solver + ', drmax={:.1f},  '.format(drmax) + preserve_scales_string + '.    Subtract Linear Fit:' + Subtr_Lin_Fit + ',  Pad Edges:'+Pad_Edg

        fig4, axs4 = subplots(2,2, figsize=(12, 8), sharex=True)
        fig4.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.92, wspace=0.18, hspace=0.04)

        axs4[0,0].plot(frame_inds[:-1], error_abs_mean[frame_inds[:-1]], 'magenta', linewidth=lwl)
        axs4[0,0].text(0.02, 0.04, 'Mean Abs Error= {:.3f}   Median Abs Error= {:.3f}'.format(np.mean(error_abs_mean), np.median(error_abs_mean)), transform=axs4[0,0].transAxes, fontsize = fs-1)
        axs4[0,0].set_ylabel('Mean Abs. Error (KeyPts)')

        axs4[1,0].plot(frame_inds[:-1], image_nsad, 'r', linewidth=lwl)
        axs4[1,0].set_ylabel('Normalized Sum of Abs. Diff')
        axs4[1,0].text(0.02, 0.04, 'NSAD mean = {:.3f}   NSAD median = {:.3f}  NSAD STD = {:.3f}'.format(nsads[0], nsads[1], nsads[2]), transform=axs4[1,0].transAxes, fontsize = fs-1)
        axs4[1,0].set_xlabel('Frame #')

        axs4[0,1].plot(frame_inds[:-1], image_ncc, 'b', linewidth=lwl)
        axs4[0,1].set_ylabel('Normalized Cross-Correlation')
        axs4[0,1].grid(True)
        axs4[0,1].text(0.02, 0.04, 'NCC mean = {:.3f}   NCC median = {:.3f}  NCC STD = {:.3f}'.format(nccs[0], nccs[1], nccs[2]), transform=axs4[0,1].transAxes, fontsize = fs-1)

        axs4[1,1].plot(frame_inds[:-1], image_mi, 'g', linewidth=lwl)
        axs4[1,1].set_ylabel('Normalized Mutual Information')
        axs4[1,1].set_xlabel('Frame #')
        axs4[1,1].grid(True)
        axs4[1,1].text(0.02, 0.04, 'NMI mean = {:.3f}   NMI median = {:.3f}  NMI STD = {:.3f}'.format(nmis[0], nmis[1], nmis[2]), transform=axs4[1,1].transAxes, fontsize = fs-1)

        for ax in axs4.ravel():
            ax.grid(True)

        if save_transformed_dataset:
            fig4.suptitle(os.path.join(data_dir, fnm_reg), fontsize = fs-2)
        if perfrom_transformation:
            axs4[0,0].text(-0.1, 1.04, Sample_ID + '    ' +  cond_str, transform=axs4[0,0].transAxes)
        else:
            axs4[0,0].text(0.0, 1.04, Sample_ID, transform=axs4[0,0].transAxes)
        if save_res_png :
            fig4.savefig(os.path.join(data_dir, fnm_reg.replace('.mrc','_RegistrationQuality.png')), dpi=300)

    registration_summary_fnm = os.path.join(data_dir, fnm_reg).replace('.mrc', '_RegistrationQuality.csv')
    columns=['Frame', 'Npts', 'Mean Abs Error', 'Image NSAD', 'Image NCC', 'Image MI']
    reg_summary = pd.DataFrame(np.vstack((frame_inds[:-1], npts[frame_inds[:-1]], error_abs_mean[frame_inds[:-1]], image_nsad, image_ncc, image_mi)).T, columns = columns, index = None)
    reg_summary.to_csv(registration_summary_fnm, index = None)
    
    return reg_summary


def transform_and_save_dataset_DASK(DASK_client, save_transformed_dataset, indices_to_transform, fls, tr_matr_cum_residual, data_minmax, npts, error_abs_mean, **kwargs):
    use_DASK = kwargs.get("use_DASK", False)
    ftype = kwargs.get("ftype", 0)
    data_dir = kwargs.get("data_dir", '')
    test_frame = FIBSEM_frame(fls[0], ftype=ftype)
    ImgB_fraction = kwargs.get("ImgB_fraction", 0)         # fusion fraction. In case if Img B is present, the fused image 
                                                           # for each frame will be constructed ImgF = (1.0-ImgB_fraction)*ImgA + ImgB_fraction*ImgB
    if test_frame.DetB == 'None':
        ImgB_fraction=0.0
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    Sample_ID = kwargs.get("Sample_ID", '')
    pad_edges =  kwargs.get("pad_edges", True)
    save_res_png  = kwargs.get("save_res_png", True)
    save_asI8 =  kwargs.get("save_asI8", False)
    dtp = kwargs.get("dtp", int16)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    mrc_mode = kwargs.get("mrc_mode", 0) 
    zbin_2x =  kwargs.get("zbin_2x", True)
    int_order = kwargs.get("int_order", 1)                  # The order of interpolation. 1: Bi-linear
    preserve_scales =  kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params =  kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    subtract_linear_fit =  kwargs.get("subtract_linear_fit", [True, True])   # If True, the linear slope will be subtracted from the cumulative shifts.
    perfrom_transformation =  kwargs.get("perfrom_transformation", True)
    invert_data =  kwargs.get("invert_data", False)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])  # [top, height, keft, width]
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    disp_res = kwargs.get("disp_res", True)

    dformat_save = 'I8' if mrc_mode==0 else 'I16'
    
    if invert_data:
        if test_frame.EightBit==0:
            data_max_glob, data_min_glob, data_max_sliding, data_min_sliding = np.negative(data_minmax)
        else:
            data_max_glob, data_min_glob, data_max_sliding, data_min_sliding = [ uint8(255) - x for x in data_minmax]
    else:
        data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax

    if pad_edges and perfrom_transformation:
        shape = [test_frame.YResolution, test_frame.XResolution]
        xmn, xmx, ymn, ymx = determine_pad_offsets(shape, tr_matr_cum_residual, disp_res)
        padx = np.int16(xmx - xmn)
        pady = np.int16(ymx - ymn)
        xi = np.int16(np.max([xmx, 0]))
        yi = np.int16(np.max([ymx, 0]))
        # The initial transformation matrices are calculated with no padding.Padding is done prior to transformation
        # so that the transformed images are not clipped.
        # Such padding means shift (by xi and yi values). Therefore the new transformation matrix
        # for padded frames will be (Shift Matrix)x(Transformation Matrix)x(Inverse Shift Matrix)
        # those are calculated below base on the amount of padding calculated above
        shift_matrix = np.array([[1.0, 0.0, xi],
                                 [0.0, 1.0, yi],
                                 [0.0, 0.0, 1.0]])
        inv_shift_matrix = np.linalg.inv(shift_matrix)
    else:
        padx = 0
        pady = 0
        xi = 0
        yi = 0
        shift_matrix = np.eye(3,3)
        inv_shift_matrix = np.eye(3,3)
 
    fpath_reg = os.path.join(data_dir, fnm_reg)
    xsz = test_frame.XResolution + padx
    xa = xi + test_frame.XResolution
    ysz = test_frame.YResolution + pady
    ya = yi + test_frame.YResolution

    xi_eval = xi + evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = xa
    yi_eval = yi + evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ya

    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0
    
    tr_args = [ImgB_fraction, xsz, ysz, xi, xa, yi, ya, int_order, invert_data, perfrom_transformation, save_asI8, data_min_glob, data_max_glob, ftype, dtp]
    fnames_orig = [fls[j] for j in indices_to_transform]
    fnames_new = [os.path.splitext(fname_orig)[0] + '_transformed.tif' for fname_orig in fnames_orig]
    params_s7 = [[fnames_orig[j], fnames_new[j], shift_matrix @ (tr_matr_cum_residual[j] @ inv_shift_matrix), tr_args] for j in indices_to_transform]

    if use_DASK:
        print('Using DASK distributed')
        futures = DASK_client.map(transform_and_save_single_frame, params_s7)
        transformed_filenames = np.array(DASK_client.gather(futures))
    else:
        print('Using Local Computation')
        transformed_filenames = []
        for j, param_s7 in enumerate(tqdm(params_s7, desc= 'Performing Single Frame Transformations ')):
            transformed_filenames.append(transform_and_save_single_frame(param_s7))

    frame0_img = np.zeros((ysz, xsz))
    frame1_img = frame0_img.copy()
    frame_img = frame0_img.copy()
    prev_frame_img = frame0_img.copy()

    image_nsad = np.zeros((len(indices_to_transform)-1), dtype=float)
    image_ncc = np.zeros((len(indices_to_transform)-1), dtype=float)
    image_mi = np.zeros((len(indices_to_transform)-1), dtype=float)

    if zbin_2x:
        # Execute this frame if 2x z-binning is selected
        print('Saving the registered and 2x z-binned stack into the file: ', fpath_reg)
        # Make a new, empty memory-mapped MRC file
        if save_transformed_dataset:
            mrc = mrcfile.new_mmap(fpath_reg, shape=(len(indices_to_transform)//2, ysz, xsz), mrc_mode=mrc_mode, overwrite=True)
            # mode 0 -> int8
            # mode 1 -> int16:
            mrc.voxel_size = np.round(test_frame.PixelSize)*0.001
        ind = np.arange(0,len(transformed_filenames)-1,2)
        for j in tqdm(ind, desc = 'Saving into ' + dformat_save + ' MRC File'):
            frame0_img_reg = tiff.imread(transformed_filenames[j])
            frame1_img_reg = tiff.imread(transformed_filenames[j+1])            
            os.remove(transformed_filenames[j])
            os.remove(transformed_filenames[j+1])
    
            if (mrc_mode==0 and test_frame.EightBit==0):
                curr_img = np.clip((frame0_img_reg/2.0 + frame1_img_reg/2.0), 0, 255)
            else:
                curr_img = frame0_img_reg/2.0 + frame1_img_reg/2.0
            if save_transformed_dataset:
                mrc.data[j//2,:,:] = np.flip(curr_img.astype(dtp), axis=0)

            if sliding_evaluation_box:
                xi_eval = start_evaluation_box[2] + dx_eval*j//2//nfrs
                yi_eval = start_evaluation_box[0] + dy_eval*j//2//nfrs
                if start_evaluation_box[3] > 0:
                    xa_eval = xi_eval + start_evaluation_box[3]
                else:
                    xa_eval = xsz
                if start_evaluation_box[1] > 0:
                    ya_eval = yi_eval + start_evaluation_box[1]
                else:
                    ya_eval = ysz
            I1c = cp.array(frame0_img_reg.astype(float)[yi_eval:ya_eval, xi_eval:xa_eval])
            if j>0:
                I2c = cp.array(prev_frame_img[yi_eval:ya_eval, xi_eval:xa_eval])
                #image_nsad[j-1] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0 - data_min_glob))
                image_nsad[j-1] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0))
                image_ncc[j-1] = Two_Image_NCC_SNR(frame0_img_reg.astype(float)[yi_eval:ya_eval, xi_eval:xa_eval], prev_frame_img[yi_eval:ya_eval, xi_eval:xa_eval])[0]
                image_mi[j-1] = cp.asnumpy(mutual_information_2d_cp(I1c.ravel(), I2c.ravel(), sigma=1.0, bin=2048, normalized=True))
            I2c = cp.array(frame1_img_reg.astype(float)[yi_eval:ya_eval, xi_eval:xa_eval])
            image_nsad[j] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0 - data_min_glob))
            image_ncc[j] = Two_Image_NCC_SNR(frame0_img_reg.astype(float)[yi_eval:ya_eval, xi_eval:xa_eval], frame1_img_reg.astype(float)[yi_eval:ya_eval, xi_eval:xa_eval])[0]
            image_mi[j] = cp.asnumpy(mutual_information_2d_cp(I1c.ravel(), I2c.ravel(), sigma=1.0, bin=2048, normalized=True))
            prev_frame_img = frame1_img_reg.astype(float)
            del I1c, I2c
    else:
        # Execute this frame if 2x z-binning is NOT selected
        print('Saving the registered stack into the file: ', fpath_reg)
        # Make a new, empty memory-mapped MRC file
        if save_transformed_dataset:
            mrc = mrcfile.new_mmap(fpath_reg, shape=(len(indices_to_transform), ysz, xsz), mrc_mode=mrc_mode, overwrite=True)
            # mode 0 -> int8
            # mode 1 -> int16:
            mrc.voxel_size = np.round(test_frame.PixelSize)*0.001
        ind = np.arange(0,len(transformed_filenames))
        for j, transformed_filename in enumerate(tqdm(transformed_filenames, desc = 'Saving into ' + dformat_save + ' MRC File')):
            frame_img_reg = tiff.imread(transformed_filename)
            os.remove(transformed_filename)
            if save_transformed_dataset:
                mrc.data[j,:,:] = np.flip(curr_img.astype(dtp), axis=0)
            if j>0:
                if sliding_evaluation_box:
                    xi_eval = start_evaluation_box[2] + dx_eval*j//nfrs
                    yi_eval = start_evaluation_box[0] + dy_eval*j//nfrs
                    if start_evaluation_box[3] > 0:
                        xa_eval = xi_eval + start_evaluation_box[3]
                    else:
                        xa_eval = xsz
                    if start_evaluation_box[1] > 0:
                        ya_eval = yi_eval + start_evaluation_box[1]
                    else:
                        ya_eval = ysz
                I1c = cp.array(frame_img_reg.astype(float)[yi_eval:ya_eval, xi_eval:xa_eval])
                I2c = cp.array(prev_frame_img[yi_eval:ya_eval, xi_eval:xa_eval])
                #image_nsad[j-1] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0 - data_min_glob))
                image_nsad[j-1] =  cp.asnumpy(cp.mean(cp.abs(I1c-I2c))/cp.mean(I1c/2.0 + I2c/2.0))
                #print('cp results: ',cp.mean(cp.abs(I1c-I2c)), cp.mean(I1c/2.0 + I2c/2.0))
                image_ncc[j-1] = Two_Image_NCC_SNR(prev_frame_img[yi_eval:ya_eval, xi_eval:xa_eval], frame_img_reg.astype(float)[yi_eval:ya_eval, xi_eval:xa_eval])[0]
                image_mi[j-1] = cp.asnumpy(mutual_information_2d_cp(I1c.ravel(), I2c.ravel(), sigma=1.0, bin=2048, normalized=True))
                del I1c, I2c
            prev_frame_img = frame_img_reg.astype(float)
    if save_transformed_dataset:
        mrc.close()

    # Generate a figure with analysis of registration quality - Image NSAD, NCC, NMI's vs. frames.
    #image_nsad = image_nsad[1:-1]
    nsads = [np.mean(image_nsad), np.median(image_nsad), np.std(image_nsad)] 
    #image_ncc = image_ncc[1:-1]
    nccs = [np.mean(image_ncc), np.median(image_ncc), np.std(image_ncc)]
    nmis = [np.mean(image_mi), np.median(image_mi), np.std(image_mi)]

    Pres_scales = 'ON' if  preserve_scales else 'OFF'
    Subtr_Lin_Fit = ('ON, ' if  subtract_linear_fit[0] else 'OFF, ') + ('ON' if  subtract_linear_fit[1] else 'OFF')
    Pad_Edg = 'ON' if  pad_edges else 'OFF'

    fs=12
    lwl=1
    if preserve_scales:
        fit_method = fit_params[0]
        if fit_method == 'LF':
            fit_str = ', Meth: Linear Fit'
            fm_string = 'linear'
        else:
            if fit_method == 'SG':
                fit_str = ', Meth: Sav.-Gol., ' + str(fit_params[1:])
                fm_string = 'Sav.-Gol.'
            else:
                fit_str = ', Meth: ' + fit_method + ', ' + str(fit_params[1:])
                fm_string = fit_method
        preserve_scales_string = 'Preserve Scales: ON' + fit_str
    else:
        preserve_scales_string = 'Preserve Scales: OFF'

    cond_str = TransformType.__name__ + ' with ' + solver + ', drmax={:.1f},  '.format(drmax) + preserve_scales_string + '.    Subtract Linear Fit:' + Subtr_Lin_Fit + ',  Pad Edges:'+Pad_Edg

    fig4, axs4 = subplots(2,2, figsize=(12, 8), sharex=True)
    fig4.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.92, wspace=0.18, hspace=0.04)

    axs4[0,0].plot(error_abs_mean[indices_to_transform[:-1]], 'magenta', linewidth=lwl)
    #axs4[0,0].text(0.02, 0.12, cond_str, transform=axs4[0,0].transAxes)
    axs4[0,0].text(0.02, 0.04, 'Mean Abs Error= {:.3f}   Median Abs Error= {:.3f}'.format(np.mean(error_abs_mean), np.median(error_abs_mean)), transform=axs4[0,0].transAxes, fontsize = fs-1)
    axs4[0,0].set_ylabel('Mean Abs. Error (KeyPts)')

    axs4[1,0].plot(image_nsad, 'r', linewidth=lwl)
    axs4[1,0].set_ylabel('Normalized Sum of Abs. Diff')
    #axs4[1,0].text(0.02, 0.12, cond_str, transform=axs4[0,1].transAxes)
    axs4[1,0].text(0.02, 0.04, 'NSAD mean = {:.3f}   NSAD median = {:.3f}  NSAD STD = {:.3f}'.format(nsads[0], nsads[1], nsads[2]), transform=axs4[1,0].transAxes, fontsize = fs-1)
    axs4[1,0].set_xlabel('Frame #')

    axs4[0,1].plot(image_ncc, 'b', linewidth=lwl)
    axs4[0,1].set_ylabel('Normalized Cross-Correlation')
    axs4[0,1].grid(True)
    #axs4[0,1].legend(loc='upper left')
    #axs4[0,1].text(0.02, 0.12, cond_str, transform=axs4[1,0].transAxes)
    axs4[0,1].text(0.02, 0.04, 'NCC mean = {:.3f}   NCC median = {:.3f}  NCC STD = {:.3f}'.format(nccs[0], nccs[1], nccs[2]), transform=axs4[0,1].transAxes, fontsize = fs-1)

    axs4[1,1].plot(image_mi, 'g', linewidth=lwl)
    axs4[1,1].set_ylabel('Normalized Mutual Information')
    axs4[1,1].set_xlabel('Frame #')
    axs4[1,1].grid(True)
    #axs4[1,1].legend(loc='upper left')
    #axs4[1,1].text(0.02, 0.12, cond_str, transform=axs4[1,1].transAxes)
    axs4[1,1].text(0.02, 0.04, 'NMI mean = {:.3f}   NMI median = {:.3f}  NMI STD = {:.3f}'.format(nmis[0], nmis[1], nmis[2]), transform=axs4[1,1].transAxes, fontsize = fs-1)

    for ax in axs4.ravel():
        ax.grid(True)
        #ax.legend(loc='upper left')

    fig4.suptitle(os.path.join(data_dir, fnm_reg), fontsize = fs-2)
    if perfrom_transformation:
        axs4[0,0].text(-0.1, 1.04, Sample_ID + '    ' +  cond_str, transform=axs4[0,0].transAxes)
    else:
        axs4[0,0].text(0.0, 1.04, Sample_ID, transform=axs4[0,0].transAxes)
    if save_res_png :
        fig4.savefig(os.path.join(data_dir, fnm_reg.replace('.mrc','_RegistrationQuality.png')), dpi=300)

    registration_summary_fnm = os.path.join(data_dir, fnm_reg).replace('.mrc', '_RegistrationQuality.csv')
    columns=['Npts', 'Mean Abs Error', 'Image NSAD', 'Image NCC', 'Image MI']
    reg_summary = pd.DataFrame(np.vstack((npts[indices_to_transform[:-1]], error_abs_mean[indices_to_transform[:-1]], image_nsad, image_ncc, image_mi)).T, columns = columns, index = None)
    reg_summary.to_csv(registration_summary_fnm, index = None)
    
    return reg_summary


def check_for_nomatch_frames_dataset(fls, fnms, fnms_matches,
                                     transformation_matrix,
                                     error_abs_mean, npts,
                                     thr_npt, **kwargs):
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')

    inds_zeros = [np.squeeze(np.argwhere(np.array(npts) < thr_npt ))]
    print('Frames with no matches to the next frame:  ', np.array(inds_zeros))
    frames_to_remove = []
    for ind0 in inds_zeros:
        if ind0 < (len(fls)-2) and npts[ind0+1] < thr_npt:
            frames_to_remove.append(ind0+1)
            print('Frame to remove: {:d} : '.format(ind0+1) + ', File: ' + fls[ind0+1])
            frame_to_remove  = FIBSEM_frame(fls[ind0+1], ftype=ftype)
            frame_to_remove.save_snapshot(dpi=300)
    print('Frames to remove:  ', frames_to_remove)

    if len(frames_to_remove) == 0:
        print('No frames selected for removal')
    else:
        # create copies of the original arrays
        fnms_orig = fnms.copy()
        fls_orig = fls.copy()
        error_abs_mean_orig = error_abs_mean.copy()
        tr_matrix_orig = transformation_matrix.copy()

        # go through the frames to be removed and remove the frames from the list and then re-calculate the shift for new neighbours.
        for j,fr in enumerate(tqdm(frames_to_remove, desc = 'Removing frames and finding shifts for new sequential frames')):
            frj = fr-j # to account for the fact that every time we remove a frame the array shrinks and indicis reset
            print('Removing the frame {:d}'.format(frj))
            print(fls[frj])
            fls = np.delete(fls, frj)
            fnms = np.delete(fnms, frj)
            fnms_matches = np.delete(fnms_matches, frj)
            error_abs_mean = np.delete(error_abs_mean, frj)
            transformation_matrix = np.delete(transformation_matrix, frj, axis = 0)
            npts = np.delete(npts, frj, axis = 0)
            fname1 = fnms[frj-1]
            fname2 = fnms[frj]
            new_step4_res = determine_transformations_files([fname1, fname2, kwargs])
            npts[frj-1] = np.array(len(new_step4_res[2][0]))
            error_abs_mean[frj-1] = new_step4_res[3]
            transformation_matrix[frj-1] = np.array(new_step4_res[0])
        print('Mean Number of Keypoints :', np.mean(npts).astype(int))
    return frames_to_remove, fls, fnms, fnms_matches, error_abs_mean, npts, transformation_matrix



class FIBSEM_dataset: 
    """
    A class representing a FIB-SEM data set
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    Contains the info/settings on the FIB-SEM dataset and the procedures that can be performed on it.

    Attributes
    ----------
    fls : array of str
        filenames for the individual data frames in the set
    data_dir : str
        data direcory (path)
    Sample_ID : str
            Sample ID
    ftype : int
        file type (0 - Shan Xu's .dat, 1 - tif)
    fnm_reg : str
        filename for the final registed dataset
    use_DASK : boolean
        use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
    threshold_min : float
        CDF threshold for determining the minimum data value
    threshold_max : float
        CDF threshold for determining the maximum data value
    nbins : int
        number of histogram bins for building the PDF and CDF
    sliding_minmax : boolean
        if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
        if False - same data_min_glob and data_max_glob will be used for all files
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
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    save_asI8 : boolean
        If True, the data will be converted to I8 using data_min_glob and data_min_glob values determined by calc_data_range method
    zbin_2x : boolean
        If True, the data will be binned 2x in z-direction (z-milling direction) when saving the final result.
    preserve_scales : boolean
        If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
    fit_params : list
        Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
        Other options are:
            ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
            ['PF', 2]  - use polynomial fit (in this case of order 2)
    int_order : int
        The order of interpolation (when transforming the data).
            The order has to be in the range 0-5:
                0: Nearest-neighbor
                1: Bi-linear (default)
                2: Bi-quadratic
                3: Bi-cubic
                4: Bi-quartic
                5: Bi-quintic
    subtract_linear_fit : [boolean, boolean]
        List of two Boolean values for two directions: X- and Y-.
        If True, the linear slopes along X- and Y- directions (respectively)
        will be subtracted from the cumulative shifts.
        This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
    pad_edges : boolean
        If True, the data will be padded before transformation to avoid clipping.
    ImgB_fraction : float
            fractional ratio of Image B to be used for constructing the fuksed image:
            ImageFused = ImageA * (1.0-ImgB_fraction) + ImageB * ImgB_fraction
    evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.

    Methods
    -------
    SIFT_evaluation(eval_fls = [], **kwargs):
        Evaluate SIFT settings and perfromance of few test frames (eval_fls).

    convert_raw_data_to_tif_files(sDASK_client = '', **kwargs):
        Convert binary ".dat" files into ".tif" files

    calc_data_range(DASK_client, **kwargs):
        Calculate Min and Max range for I8 conversion of the data (open CV SIFT requires I8)

    extract_keypoints(DASK_client, **kwargs):
        Extract Key-Points and Descriptors

    determine_transformations(DASK_client, **kwargs):
        Determine transformation matrices for sequential frame pairs

    process_transformation_matrix(**kwargs):
        Calculate cumulative transformation matrix

    save_parameters(**kwargs):
        Save transformation attributes and parameters (including transformation matrices)

    check_for_nomatch_frames(thr_npt, **kwargs):
        Check for frames with low number of Key-Point matches,m exclude them and re-calculate the cumulative transformation matrix

    transform_and_save(**kwargs):
        Transform the frames using the cumulative transformation matrix and save the data set into .mrc file

    show_eval_box(**kwargs):
        Show the box used for evaluating the registration quality

    estimate_SNRs(**kwargs):
        Estimate SNRs in Image A and Image B based on single-image SNR calculation.

    evaluate_ImgB_fractions(ImgB_fractions, frame_inds, **kwargs):
        Calculate NCC and SNR vs Image B fraction over a set of frames.
    """

    def __init__(self, fls, data_dir, **kwargs):
        """
        Initializes an instance of  FIBSEM_dataset object. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com

        Parameters
        ----------
        fls : array of str
            filenames for the individual data frames in the set
        data_dir : str
            data direcory (path)

        kwargs
        ---------
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        Sample_ID : str
                Sample ID
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        sliding_minmax : boolean
            if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
            if False - same data_min_glob and data_max_glob will be used for all files
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
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        save_asI8 : boolean
            If True, the data will be converted to I8 using data_min_glob and data_min_glob values determined by calc_data_range method
        zbin_2x : boolean
            If True, the data will be binned 2x in z-direction (z-milling direction) when saving the final result.
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        int_order : int
            The order of interpolation (when transforming the data).
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        """
        self.fls = fls
        self.fnms = [os.path.splitext(fl)[0] + '_kpdes.bin' for fl in fls]
        self.nfrs = len(fls)
        print('Total Number of frames: ', self.nfrs)
        self.data_dir = data_dir
        self.ftype = kwargs.get("ftype", 0) # ftype=0 - Shan Xu's binary format  ftype=1 - tif files
        test_frame = FIBSEM_frame(fls[0], ftype=self.ftype)
        self.DetA = test_frame.DetA
        self.DetB = test_frame.DetB
        self.ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)
        if self.DetB == 'None':
            ImgB_fraction = 0.0
        self.Sample_ID = kwargs.get("Sample_ID", '')
        self.EightBit = kwargs.get("EightBit", 1)
        self.use_DASK = kwargs.get("use_DASK", True)
        self.threshold_min = kwargs.get("threshold_min", 1e-3)
        self.threshold_max = kwargs.get("threshold_max", 1e-3)
        self.nbins = kwargs.get("nbins", 256)
        self.sliding_minmax = kwargs.get("sliding_minmax", True)
        self.TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
        l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
        l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
        l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
        l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
        self.l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
        self.targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
        self.solver = kwargs.get("solver", 'RANSAC')
        self.drmax = kwargs.get("drmax", 2.0)
        self.max_iter = kwargs.get("max_iter", 1000)
        self.BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        self.save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
        self.kp_max_num = kwargs.get("kp_max_num", -1)
        self.save_res_png  = kwargs.get("save_res_png", True)

        self.save_asI8 =  kwargs.get("save_asI8", False) 
        self.zbin_2x =  kwargs.get("zbin_2x", True)                 # If true, the data will be converted to I8 using global MIN and MAX values determined in the Step 1
        self.preserve_scales =  kwargs.get("preserve_scales", True) # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
        self.fit_params =  kwargs.get("fit_params", False)          # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                                    # window size 701, polynomial order 3

        self.int_order = kwargs.get("int_order", False)             #     The order of interpolation. The order has to be in the range 0-5:
                                                                    #    - 0: Nearest-neighbor
                                                                    #    - 1: Bi-linear (default)
                                                                    #    - 2: Bi-quadratic
                                                                    #    - 3: Bi-cubic
                                                                    #    - 4: Bi-quartic
                                                                    #    - 5: Bi-quintic
        self.subtract_linear_fit =  kwargs.get("subtract_linear_fit", [True, True])   # If True, the linear slope will be subtracted from the cumulative shifts.
        self.pad_edges =  kwargs.get("pad_edges", True)
        self.fnm_reg, self.mrc_mode, self.dtp = build_filename(fls[0], **kwargs)
        print('Registered data will be saved into: ', self.fnm_reg)


        kwargs.update({'mrc_mode' : self.mrc_mode, 'data_dir' : self.data_dir, 'fnm_reg' : self.fnm_reg, 'dtp' : self.dtp})

        if kwargs.get("recall_parameters", False):
            dump_filename = kwargs.get("dump_filename", '')
            try:
                dump_data = pickle.load(open(dump_filename, 'rb'))
                dump_loaded = True
            except Exception as ex1:
                dump_loaded = False
                print('Failed to open Parameter dump filename: ', dump_filename)
                print(ex1.message)

            if dump_loaded:
                try:
                    for key in tqdm(dump_data, desc='Recalling the data set parameters'):
                        setattr(self, key, dump_data[key])
                except Exception as ex2:
                    print('Parameter dump filename: ', dump_filename)
                    print('Failed to restore the object parameters')
                    print(ex2.message)
            
 

    def SIFT_evaluation(self, eval_fls = [], **kwargs):
        '''
        Evaluate SIFT settings and perfromance of few test frames (eval_fls).
        
        Parameters:
        eval_fls : array of str
            filenames for the data frames to be used for SIFT evaluation
        
        kwargs
        ---------
        data_dir : str
            data direcory (path)
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        fnm_reg : str
            filename for the final registed dataset
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        
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
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
    
        Returns:
        dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts
        '''
        if len(eval_fls) == 0:
            eval_fls = [self.fls[self.nfrs//2], self.fls[self.nfrs//2+1]]
        data_dir = kwargs.get("data_dir", self.data_dir)
        ftype = kwargs.get("ftype", self.ftype)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        threshold_min = kwargs.get("threshold_min", self.threshold_min)
        threshold_max = kwargs.get("threshold_max", self.threshold_max)
        nbins = kwargs.get("nbins", self.nbins)
        TransformType = kwargs.get("TransformType", self.TransformType)
        l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
        targ_vector = kwargs.get("targ_vector", self.targ_vector)
        solver = kwargs.get("solver", self.solver)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)
        Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        save_matches = kwargs.get("save_matches", self.save_matches)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)

        
        dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts = SIFT_evaluation_dataset(eval_fls,
                                ftype = ftype,
                                Sample_ID = Sample_ID,
                                data_dir = data_dir,
                                fnm_reg = fnm_reg,
                                threshold_min = threshold_min,
                                threshold_max = threshold_max,
                                nbins = nbins,
                                TransformType = TransformType, 
                                l2_matrix = l2_matrix,
                                targ_vector = targ_vector,
                                solver = solver,
                                drmax = drmax,
                                max_iter = max_iter,
                                kp_max_num = kp_max_num,
                                Lowe_Ratio_Threshold = Lowe_Ratio_Threshold,
                                BFMatcher = BFMatcher,
                                save_matches = save_matches,
                                save_res_png  = save_res_png )
        src_pts_filtered, dst_pts_filtered = kpts
        print('Transformation Matrix determined using '+ TransformType.__name__ +' using ' + solver + ' solver')
        print(transform_matrix)
        print('{:d} keypoint matches were detected with {:.1f} pixel outlier threshold'.format(n_matches, drmax))
        print('Number of iterations: {:d}'.format(iteration))
        return dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts


    def convert_raw_data_to_tif_files(self, DASK_client = '', **kwargs):
        '''
        Convert binary ".dat" files into ".tif" files.
        
        Parameters:
        DASK_client : instance of the DASK client object
        
        kwargs
        ---------
            use_DASK : boolean
                use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        '''
        use_DASK = kwargs.get("use_DASK", self.use_DASK)
        if self.ftype ==0 :
            print('Step 2a: Creating "*InLens.tif" files using DASK distributed')
            t00 = time.time()
            if use_DASK:
                try:
                    futures = DASK_client.map(save_inlens_data, self.fls)
                    fls_new = np.array(DASK_client.gather(futures))
                except:
                    fls_new = []
                    for fl in tqdm(self.fls, desc = 'Converting .dat data files into .tif format'):
                            fls_new.append(save_inlens_data(fl))
            else:
                fls_new = []
                for fl in tqdm(self.fls, desc = 'Converting .dat data files into .tif format'):
                    fls_new.append(save_inlens_data(fl))

            t01 = time.time()
            print('Step 2a: Elapsed time: {:.2f} seconds'.format(t01 - t00))
            print('Step 2a: Quick check if all files were converted: ', np.array_equal(self.fls, fls_new))
        else:
            print('Step 2a: data is already in TIF format')


    def calc_data_range(self, DASK_client, **kwargs):
        '''
        Calculate Min and Max range for I8 conversion of the data (open CV SIFT requires I8).
        
        Parameters:
        DASK_client : instance of the DASK client object
        
        kwargs
        ---------
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        Sample_ID : str
            Sample ID
        EightBit : int
            0 - 16-bit data, 1: 8-bit data
        fnm_reg : str
            filename for the final registed dataset
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        sliding_minmax : boolean
            if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
            if False - same data_min_glob and data_max_glob will be used for all files
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
    
        Returns:
        data_minmax : list of 4 parameters
            data_min_glob : float   
                min data value for I8 conversion (open CV SIFT requires I8)
            data_max_glob : float   
                max data value for I8 conversion (open CV SIFT requires I8)
            data_min_sliding : float array
                min data values (one per file) for I8 conversion
            data_max_sliding : float array
                max data values (one per file) for I8 conversion
        '''
        use_DASK = kwargs.get("use_DASK", self.use_DASK)
        ftype = kwargs.get("ftype", self.ftype)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        data_dir = self.data_dir
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        EightBit = kwargs.get("EightBit", self.EightBit)
        threshold_min = kwargs.get("threshold_min", self.threshold_min)
        threshold_max = kwargs.get("threshold_max", self.threshold_max)
        nbins = kwargs.get("nbins", self.nbins)
        sliding_minmax = kwargs.get("sliding_minmax", self.sliding_minmax)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        fit_params = kwargs.get("fit_params", self.fit_params)
        self.data_minmax = calc_data_range_dataset(self.fls, DASK_client,
                                use_DASK = use_DASK,
                                ftype = ftype,
                                Sample_ID = Sample_ID,
                                data_dir = data_dir,
                                fnm_reg = fnm_reg,
                                EightBit = EightBit,
                                threshold_min = threshold_min,
                                threshold_max = threshold_max,
                                nbins = nbins,
                                sliding_minmax = sliding_minmax,
                                save_res_png  = save_res_png ,
                                fit_params = fit_params)
        return self.data_minmax


    def extract_keypoints(self, DASK_client, **kwargs):
        '''
        Extract Key-Points and Descriptors

        Parameters:
        DASK_client : instance of the DASK client object
        
        kwargs
        ---------
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        EightBit : int
            0 - 16-bit data, 1: 8-bit data
        fnm_reg : str
            filename for the final registed dataset
        threshold_min : float
            CDF threshold for determining the minimum data value
        threshold_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        sliding_minmax : boolean
            if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
            if False - same data_min_glob and data_max_glob will be used for all files
        data_minmax : list of 4 parameters
            data_min_glob : float   
                min data value for I8 conversion (open CV SIFT requires I8)
            data_max_glob : float   
                max data value for I8 conversion (open CV SIFT requires I8)
            data_min_sliding : float array
                min data values (one per file) for I8 conversion
            data_max_sliding : float array
                max data values (one per file) for I8 conversion
        kp_max_num : int
            Max number of key-points to be matched.
            Key-points in every frame are indexed (in descending order) by the strength of the response.
            Only kp_max_num is kept for further processing.
            Set this value to -1 if you want to keep ALL keypoints (may take forever to process!)
    
        Returns:
        fnms : array of str
            filenames for binary files kontaining Key-Point and Descriptors for each frame
        '''
        if len(self.fls) == 0:
            print('Data set not defined, perform initialization first')
            fnms = []
        else:  
            use_DASK = kwargs.get("use_DASK", self.use_DASK)
            ftype = kwargs.get("ftype", self.ftype)
            data_dir = self.data_dir
            fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
            threshold_min = kwargs.get("threshold_min", self.threshold_min)
            threshold_max = kwargs.get("threshold_max", self.threshold_max)
            nbins = kwargs.get("nbins", self.nbins)
            sliding_minmax = kwargs.get("sliding_minmax", self.sliding_minmax)
            data_minmax = kwargs.get("data_minmax", self.data_minmax)
            kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)
            data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax
            kpt_kwargs = {'ftype' : ftype, 'threshold_min' : threshold_min, 'threshold_max' : threshold_max, 'nbins' : nbins, 'kp_max_num' : kp_max_num}

            if sliding_minmax:
                params_s3 = [[dts3[0], dts3[1], dts3[2], kpt_kwargs] for dts3 in zip(self.fls, data_min_sliding, data_max_sliding)]
            else:
                params_s3 = [[fl, data_min_glob, data_max_glob, kpt_kwargs] for fl in self.fls]        
            if use_DASK:
                print('Using DASK distributed')
                futures_s3 = DASK_client.map(extract_keypoints_descr_files, params_s3)
                fnms = DASK_client.gather(futures_s3)
            else:
                print('Using Local Computation')
                fnms = []
                for j, param_s3 in enumerate(tqdm(params_s3, desc='Extracting Key Points and Descriptors: ')):
                    fnms.append(extract_keypoints_descr_files(param_s3))

            self.fnms = fnms
        return fnms


    def determine_transformations(self, DASK_client, **kwargs):
        '''
        Determine transformation matrices for sequential frame pairs

        Parameters:
        DASK_client : instance of the DASK client object
        
        kwargs
        ---------
        use_DASK : boolean
            use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
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
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        Lowe_Ratio_Threshold : float
            threshold for Lowe's Ratio Test
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check

    
        Returns:
        results_s4 : array of lists containing the reults:
            results_s4 = [transformation_matrix, fnm_matches, npt, error_abs_mean]
            transformation_matrix : 2D float array
                transformation matrix for each sequential frame pair
            fnm_matches : str
                filename containing the matches used to determin the transformation for the par of frames
            npts : int
                number of matches
            error_abs_mean : float
                mean abs error of registration for all matched Key-Points
        '''
        if len(self.fnms) == 0:
            print('No data on individual key-point data files, peform key-point search')
            results_s4 = []
        else:
            use_DASK = kwargs.get("use_DASK", self.use_DASK)
            ftype = kwargs.get("ftype", self.ftype)
            TransformType = kwargs.get("TransformType", self.TransformType)
            l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
            targ_vector = kwargs.get("targ_vector", self.targ_vector)
            solver = kwargs.get("solver", self.solver)
            drmax = kwargs.get("drmax", self.drmax)
            max_iter = kwargs.get("max_iter", self.max_iter)
            kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)
            Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
            BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
            save_matches = kwargs.get("save_matches", self.save_matches)
            save_res_png  = kwargs.get("save_res_png", self.save_res_png )
            dt_kwargs = {'ftype' : ftype,
                            'TransformType' : TransformType,
                            'l2_matrix' : l2_matrix,
                            'targ_vector': targ_vector, 
                            'solver' : solver,
                            'drmax' : drmax,
                            'max_iter' : max_iter,
                            'BFMatcher' : BFMatcher,
                            'save_matches' : save_matches,
                            'kp_max_num' : kp_max_num,
                            'Lowe_Ratio_Threshold' : Lowe_Ratio_Threshold}

            params_s4 = []
            for j, fnm in enumerate(self.fnms[:-1]):
                fname1 = self.fnms[j]
                fname2 = self.fnms[j+1]
                params_s4.append([fname1, fname2, dt_kwargs])
            if use_DASK:
                print('Using DASK distributed')
                futures4 = DASK_client.map(determine_transformations_files, params_s4)
                #determine_transformations_files returns (transform_matrix, fnm_matches, kpts, iteration)
                results_s4 = DASK_client.gather(futures4)
            else:
                print('Using Local Computation')
                results_s4 = []
                for param_s4 in tqdm(params_s4, desc = 'Extracting Transformation Parameters: '):
                    results_s4.append(determine_transformations_files(param_s4))
            #determine_transformations_files returns (transform_matrix, fnm_matches, kpts, errors, iteration)
            self.transformation_matrix = np.nan_to_num(np.array([result[0] for result in results_s4]))
            self.fnms_matches = [result[1] for result in results_s4]
            self.error_abs_mean = np.nan_to_num(np.array([result[3] for result in results_s4]))
            self.npts = np.nan_to_num(np.array([len(result[2][0])  for result in results_s4]))
            print('Mean Number of Keypoints :', np.mean(self.npts).astype(np.int16))
        return results_s4


    def process_transformation_matrix(self, **kwargs):
        '''
        Calculate cumulative transformation matrix
        
        kwargs
        ---------
        data_dir : str
            data direcory (path)
        fnm_reg : str
            filename for the final registed dataset
        Sample_ID : str
            Sample ID
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
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
    
        Returns:
        tr_matr_cum_residual : list of 2D arrays of float
            Cumulative transformation matrices
        '''
        if len(self.transformation_matrix) == 0:
            print('No data on individual key-point matches, peform key-point search / matching first')
            self.tr_matr_cum_residual = []
        else:
            data_dir = kwargs.get("data_dir", self.data_dir)
            fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
            TransformType = kwargs.get("TransformType", self.TransformType)
            Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
            l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
            targ_vector = kwargs.get("targ_vector", self.targ_vector)
            solver = kwargs.get("solver", self.solver)
            drmax = kwargs.get("drmax", self.drmax)
            max_iter = kwargs.get("max_iter", self.max_iter)
            BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
            save_matches = kwargs.get("save_matches", self.save_matches)
            kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)
            save_res_png  = kwargs.get("save_res_png", self.save_res_png )
            preserve_scales =  kwargs.get("preserve_scales", self.preserve_scales)
            fit_params =  kwargs.get("fit_params", self.fit_params)
            subtract_linear_fit =  kwargs.get("subtract_linear_fit", self.subtract_linear_fit)
            pad_edges =  kwargs.get("pad_edges", self.pad_edges)

            TM_kwargs = {'fnm_reg' : fnm_reg,
                            'data_dir' : data_dir,
                            'TransformType' : TransformType,
                            'Sample_ID' : Sample_ID,
                            'l2_matrix' : l2_matrix,
                            'targ_vector': targ_vector, 
                            'solver' : solver,
                            'drmax' : drmax,
                            'max_iter' : max_iter,
                            'BFMatcher' : BFMatcher,
                            'save_matches' : save_matches,
                            'kp_max_num' : kp_max_num,
                            'save_res_png ' : save_res_png ,
                            'preserve_scales' : preserve_scales,
                            'fit_params' : fit_params,
                            'subtract_linear_fit' : subtract_linear_fit,
                            'pad_edges' : pad_edges}
            self.tr_matr_cum_residual = process_transf_matrix(self.transformation_matrix,
                                             self.fnms_matches,
                                             self.npts,
                                             self.error_abs_mean,
                                             **TM_kwargs)
        return self.tr_matr_cum_residual

    def save_parameters(self, **kwargs):
        '''
        Save transformation attributes and parameters (including transformation matrices).

        kwargs:
        -------
        dump_file : string
            String containing the name of the binary dump for saving all attributes of the current istance of the FIBSEM_dataset object.


        Returns:
        dump_file : string
        '''
        default_dump_file = os.path.join(self.data_dir, self.fnm_reg.replace('.mrc', '_params.bin'))
        dump_file = kwargs.get("dump_file", default_dump_file)

        pickle.dump(self.__dict__, open(dump_file, 'wb'))

        npts_fnm = dump_file.replace('_params.bin', '_Npts_Errs_data.csv')
        Tr_matrix_xls_fnm = dump_file.replace('_params.bin', '_Transform_Matrix_data.csv')
        
        # Save the keypoint statistics into a CSV file
        columns=['Npts', 'Mean Abs Error']
        npdt = pd.DataFrame(np.vstack((self.npts, self.error_abs_mean)).T, columns = columns, index = None)
        npdt.to_csv(npts_fnm, index = None)
        
        # Save the X-Y shift data and keypoint statistics into a CSV file
        columns=['T00 (Sxx)', 'T01 (Sxy)', 'T02 (Tx)',  
                 'T10 (Syx)', 'T11 (Syy)', 'T12 (Ty)', 
                 'T20 (0.0)', 'T21 (0.0)', 'T22 (1.0)']
        tr_mx_dt = pd.DataFrame(self.transformation_matrix.reshape((len(self.transformation_matrix), 9)), columns = columns, index = None)
        tr_mx_dt.to_csv(Tr_matrix_xls_fnm, index = None)
        return dump_file

    def check_for_nomatch_frames(self, thr_npt, **kwargs):
        '''
        Calculate cumulative transformation matrix

        Parameters:
        -----------
        thr_npt : int
            minimum number of matches. If the pair has less than this - it is reported as "suspicious" and is excluded.

        kwargs
        ---------
        data_dir : str
            data direcory (path)
        fnm_reg : str
            filename for the final registed dataset
        Sample_ID : str
            Sample ID
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
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.

        Returns:
        tr_matr_cum_residual : list of 2D arrays of float
            Cumulative transformation matrices
        '''
        self.thr_npt = thr_npt
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        TransformType = kwargs.get("TransformType", self.TransformType)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
        targ_vector = kwargs.get("targ_vector", self.targ_vector)
        solver = kwargs.get("solver", self.solver)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        save_matches = kwargs.get("save_matches", self.save_matches)
        kp_max_num = kwargs.get("kp_max_num", self.kp_max_num)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        preserve_scales =  kwargs.get("preserve_scales", self.preserve_scales)
        fit_params =  kwargs.get("fit_params", self.fit_params)
        subtract_linear_fit =  kwargs.get("subtract_linear_fit", self.subtract_linear_fit)
        pad_edges =  kwargs.get("pad_edges", self.pad_edges)
  
        res_nomatch_check = check_for_nomatch_frames_dataset(self.fls, self.fnms, self.fnms_matches,
                                     self.transformation_matrix, self.error_abs_mean, self.npts,
                                     thr_npt,
                                     data_dir = self.data_dir, fnm_reg = self.fnm_reg)
        frames_to_remove, self.fls, self.fnms, self.fnms_matches, self.error_abs_mean, self.npts, self.transformation_matrix = res_nomatch_check

        if len(frames_to_remove) > 0:
            TM_kwargs = {'fnm_reg' : fnm_reg,
                            'data_dir' : data_dir,
                            'TransformType' : TransformType,
                            'Sample_ID' : Sample_ID,
                            'l2_matrix' : l2_matrix,
                            'targ_vector': targ_vector, 
                            'solver' : solver,
                            'drmax' : drmax,
                            'max_iter' : max_iter,
                            'BFMatcher' : BFMatcher,
                            'save_matches' : save_matches,
                            'kp_max_num' : kp_max_num,
                            'save_res_png ' : save_res_png ,
                            'preserve_scales' : preserve_scales,
                            'fit_params' : fit_params,
                            'subtract_linear_fit' : subtract_linear_fit,
                            'pad_edges' : pad_edges}
            self.tr_matr_cum_residual = process_transf_matrix(self.transformation_matrix,
                                             self.fnms_matches,
                                             self.npts,
                                             self.error_abs_mean,
                                             **TM_kwargs)
        return self.tr_matr_cum_residual


    def transform_and_save(self, save_transformed_dataset=True, frame_inds=np.array((-1)), **kwargs):
        '''
        Transform the frames using the cumulative transformation matrix and save the data set into .mrc file
        save_transformed_dataset : boolean
            If true, the transformed data set will be saved into MRC file
        frame_inds : int array
            Array of frame indecis. If not set or set to np.array((-1)), all frames will be transformed
        
        kwargs
        ---------
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        data_dir : str
            data direcory (path)
        fnm_reg : str
            filename for the final registed dataset
        ImgB_fraction : float
            fractional ratio of Image B to be used for constructing the fuksed image:
            ImageFused = ImageA * (1.0-ImgB_fraction) + ImageB * ImgB_fraction
        Sample_ID : str
            Sample ID
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
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        perfrom_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed.
        invert_data : boolean
            If True - the data is inverted.
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
        start_evaluation_box : list of 4 int
            see above
        stop_evaluation_box : list of 4 int
            see above
        
        Returns:
        reg_summary : pandas DataFrame
            reg_summary = pd.DataFrame(np.vstack((npts, error_abs_mean, image_nsad, image_ncc, image_mi)
        '''
        if (frame_inds == np.array((-1))).all():
            frame_inds = np.arange(len(self.fls))

        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        ImgB_fraction = kwargs.get("ImgB_fraction", self.ImgB_fraction)
        if self.DetB == 'None':
            ImgB_fraction = 0.0
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        pad_edges =  kwargs.get("pad_edges", self.pad_edges)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        save_asI8 =  kwargs.get("save_asI8", self.save_asI8)
        dtp = kwargs.get("dtp", self.dtp)
        TransformType = kwargs.get("TransformType", self.TransformType)
        solver = kwargs.get("solver", self.solver)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        mrc_mode = kwargs.get("mrc_mode", self.mrc_mode) 
        zbin_2x =  kwargs.get("zbin_2x", self.zbin_2x)
        int_order = kwargs.get("int_order", self.int_order) 
        preserve_scales =  kwargs.get("preserve_scales", self.preserve_scales)
        fit_params =  kwargs.get("fit_params", self.fit_params)
        subtract_linear_fit =  kwargs.get("subtract_linear_fit", self.subtract_linear_fit)

        perfrom_transformation =  kwargs.get("perfrom_transformation", True)  and hasattr(self, 'tr_matr_cum_residual')
        invert_data =  kwargs.get("invert_data", False)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        disp_res  = kwargs.get("disp_res", True )
        
        save_kwargs = {'fnm_reg' : fnm_reg,
                            'ftype' : ftype,
                            'data_dir' : data_dir,
                            'Sample_ID' : Sample_ID,
                            'pad_edges' : pad_edges,
                            'ImgB_fraction' : ImgB_fraction,
                            'save_res_png ' : save_res_png ,
                            'save_asI8' : save_asI8,
                            'dtp' : dtp,
                            'TransformType' : TransformType,
                            'solver' : solver,
                            'drmax' : drmax,
                            'max_iter' : max_iter,
                            'BFMatcher' : BFMatcher,
                            'mrc_mode' : mrc_mode,
                            'zbin_2x' : zbin_2x,
                            'int_order' : int_order,                        
                            'preserve_scales' : preserve_scales,
                            'fit_params' : fit_params,
                            'subtract_linear_fit' : subtract_linear_fit,
                            'perfrom_transformation' : perfrom_transformation,
                            'invert_data' : invert_data,
                            'evaluation_box' : evaluation_box,
                            'sliding_evaluation_box' : sliding_evaluation_box,
                            'start_evaluation_box' : start_evaluation_box,
                            'stop_evaluation_box' : stop_evaluation_box,
                            'disp_res' : disp_res}

        if perfrom_transformation and hasattr(self, 'tr_matr_cum_residual'):
            reg_summary = transform_and_save_dataset(save_transformed_dataset, frame_inds,
                self.fls, self.tr_matr_cum_residual, self.data_minmax, self.npts, self.error_abs_mean, **save_kwargs)
        else:
            error_abs_mean = np.zeros(len(self.fls)-1)
            npts = np.zeros(len(self.fls)-1)
            tr_matr_cum_residual = ''
            reg_summary = transform_and_save_dataset(save_transformed_dataset, frame_inds,
                self.fls, tr_matr_cum_residual, self.data_minmax, npts, error_abs_mean, **save_kwargs)

        return reg_summary


    def transform_and_save_DASK(self, DASK_client, frame_inds = np.array((-1)), save_transformed_dataset=True, **kwargs):
        '''
        Transform the frames using the cumulative transformation matrix and save the data set into .mrc file
        
        Parameters:
        DASK_client : instance of the DASK client object
        frame_inds : int array
            Array of frame indecis. If not set or set to np.array((-1)), all frames will be transformed
        save_transformed_dataset : boolean
            If true, the transformed data set will be saved into MRC file

        kwargs
        ---------
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        data_dir : str
            data direcory (path)
        fnm_reg : str
            filename for the final registed dataset
        ImgB_fraction : float
            fractional ratio of Image B to be used for constructing the fuksed image:
            ImageFused = ImageA * (1.0-ImgB_fraction) + ImageB * ImgB_fraction
        Sample_ID : str
            Sample ID
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
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg)
        BFMatcher : boolean
            If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        save_matches : boolean
            If True, matches will be saved into individual files
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        perfrom_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed.
        invert_data : boolean
            If True - the data is inverted.
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
        start_evaluation_box : list of 4 int
            see above
        stop_evaluation_box : list of 4 int
            see above

        Returns:
        reg_summary : pandas DataFrame
            reg_summary = pd.DataFrame(np.vstack((npts, error_abs_mean, image_nsad, image_ncc, image_mi)
        '''
        if f(frame_inds == np.array((-1))).all():
            frame_inds = np.arange(len(self.fls))
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        ImgB_fraction = kwargs.get("ImgB_fraction", self.ImgB_fraction)
        if self.DetB == 'None':
            ImgB_fraction = 0.0
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        pad_edges =  kwargs.get("pad_edges", self.pad_edges)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        save_asI8 =  kwargs.get("save_asI8", self.save_asI8)
        dtp = kwargs.get("dtp", self.dtp)
        TransformType = kwargs.get("TransformType", self.TransformType)
        solver = kwargs.get("solver", self.solver)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        mrc_mode = kwargs.get("mrc_mode", self.mrc_mode) 
        zbin_2x =  kwargs.get("zbin_2x", self.zbin_2x)
        int_order = kwargs.get("int_order", self.int_order) 
        preserve_scales =  kwargs.get("preserve_scales", self.preserve_scales)
        fit_params =  kwargs.get("fit_params", self.fit_params)
        subtract_linear_fit =  kwargs.get("subtract_linear_fit", self.subtract_linear_fit)

        perfrom_transformation =  kwargs.get("perfrom_transformation", True)  and hasattr(self, 'tr_matr_cum_residual')
        invert_data =  kwargs.get("invert_data", False)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        disp_res  = kwargs.get("disp_res", True )
        
        save_kwargs = {'fnm_reg' : fnm_reg,
                            'ftype' : ftype,
                            'data_dir' : data_dir,
                            'Sample_ID' : Sample_ID,
                            'pad_edges' : pad_edges,
                            'ImgB_fraction' : ImgB_fraction,
                            'save_res_png ' : save_res_png ,
                            'save_asI8' : save_asI8,
                            'dtp' : dtp,
                            'TransformType' : TransformType,
                            'solver' : solver,
                            'drmax' : drmax,
                            'max_iter' : max_iter,
                            'BFMatcher' : BFMatcher,
                            'mrc_mode' : mrc_mode,
                            'zbin_2x' : zbin_2x,
                            'int_order' : int_order,                        
                            'preserve_scales' : preserve_scales,
                            'fit_params' : fit_params,
                            'subtract_linear_fit' : subtract_linear_fit,
                            'perfrom_transformation' : perfrom_transformation,
                            'invert_data' : invert_data,
                            'evaluation_box' : evaluation_box,
                            'sliding_evaluation_box' : sliding_evaluation_box,
                            'start_evaluation_box' : start_evaluation_box,
                            'stop_evaluation_box' : stop_evaluation_box,
                            'perfrom_transformation' : perfrom_transformation,
                            'invert_data' : invert_data,
                            'disp_res' : disp_res}

        if perfrom_transformation and hasattr(self, 'tr_matr_cum_residual'):
            reg_summary = transform_and_save_dataset_DASK(DASK_client, save_transformed_dataset, frame_inds, self.fls, self.tr_matr_cum_residual, self.data_minmax, self.npts, self.error_abs_mean, **save_kwargs)
        else:
            error_abs_mean = np.zeros(len(self.fls)-1)
            npts = np.zeros(len(self.fls)-1)
            tr_matr_cum_residual = ''
            reg_summary = transform_and_save_dataset_DASK(DASK_client, save_transformed_dataset, frame_inds, self.fls, tr_matr_cum_residual, self.data_minmax, self.npts, error_abs_mean, **save_kwargs)

        return reg_summary


    def show_eval_box(self, **kwargs):
        '''
        Show the box used for evaluating the registration quality

        kwargs
        ---------
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
        start_evaluation_box : list of 4 int
            see above
        stop_evaluation_box : list of 4 int
            see above
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        data_dir : str
            data direcory (path)
        data_minmax : list of 4 parameters
            data_min_glob : float   
                min data value for I8 conversion (open CV SIFT requires I8)
            data_max_glob : float   
                max data value for I8 conversion (open CV SIFT requires I8)
            data_min_sliding : float array
                min data values (one per file) for I8 conversion
            data_max_sliding : float array
                max data values (one per file) for I8 conversion
        fnm_reg : str
            filename for the final registed dataset
        Sample_ID : str
            Sample ID
        int_order : int
            The order of interpolation (when transforming the data).
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        perfrom_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed
        invert_data : boolean
            If True - the data is inverted
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        frame_inds : list of int
            List oif frame indecis to use to display the evaluation box.
            Default are [nfrs//10, nfrs//2, nfrs//10*9]
        '''
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        
        if "data_minmax" in kwargs.keys():
            data_minmax = kwargs.get("data_minmax")
            data_minmax_exists = True
        else:
            if hasattr(self, "data_minmax"):
                data_minmax = self.data_minmax
                data_minmax_exists = True
            else:
                data_minmax_exists = False

        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        int_order = kwargs.get("int_order", self.int_order) 
        perfrom_transformation =  kwargs.get("perfrom_transformation", True) and hasattr(self, 'tr_matr_cum_residual')
        invert_data =  kwargs.get("invert_data", False)
        pad_edges =  kwargs.get("pad_edges", self.pad_edges)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )

        fls = self.fls
        nfrs = len(fls)
        default_indecis = [nfrs//10, nfrs//2, nfrs//10*9]
        frame_inds = kwargs.get("frame_inds", default_indecis)

        test_frame = FIBSEM_frame(fls[0], ftype=ftype)
        
        if data_minmax_exists:
            if invert_data:
                if test_frame.EightBit==0:
                    data_max_glob, data_min_glob, data_max_sliding, data_min_sliding = np.negative(data_minmax)
                else:
                    data_max_glob, data_min_glob, data_max_sliding, data_min_sliding = [ uint8(255) - x for x in data_minmax]
            else:
                data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax


        if pad_edges and perfrom_transformation:
            shape = [test_frame.YResolution, test_frame.XResolution]
            xmn, xmx, ymn, ymx = determine_pad_offsets(shape, self.tr_matr_cum_residual, False)
            padx = np.int16(xmx - xmn)
            pady = np.int16(ymx - ymn)
            xi = np.int16(np.max([xmx, 0]))
            yi = np.int16(np.max([ymx, 0]))
            # The initial transformation matrices are calculated with no padding.Padding is done prior to transformation
            # so that the transformed images are not clipped.
            # Such padding means shift (by xi and yi values). Therefore the new transformation matrix
            # for padded frames will be (Shift Matrix)x(Transformation Matrix)x(Inverse Shift Matrix)
            # those are calculated below base on the amount of padding calculated above
            shift_matrix = np.array([[1.0, 0.0, xi],
                                     [0.0, 1.0, yi],
                                     [0.0, 0.0, 1.0]])
            inv_shift_matrix = np.linalg.inv(shift_matrix)
        else:
            padx = 0
            pady = 0
            xi = 0
            yi = 0
            shift_matrix = np.eye(3,3)
            inv_shift_matrix = np.eye(3,3)
     
        xsz = test_frame.XResolution + padx
        xa = xi + test_frame.XResolution
        ysz = test_frame.YResolution + pady
        ya = yi + test_frame.YResolution

        xi_eval = xi + evaluation_box[2]
        if evaluation_box[3] > 0:
            xa_eval = xi_eval + evaluation_box[3]
        else:
            xa_eval = xa
        yi_eval = yi + evaluation_box[0]
        if evaluation_box[1] > 0:
            ya_eval = yi_eval + evaluation_box[1]
        else:
            ya_eval = ya

        if sliding_evaluation_box:
            dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
            dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
        else:
            dx_eval = 0
            dy_eval = 0

        frame_img = np.zeros((ysz, xsz))
        
        for j in frame_inds:
            if invert_data:
                if test_frame.EightBit==0:
                    frame_img[yi:ya, xi:xa] = np.negative(FIBSEM_frame(fls[j], ftype=ftype).RawImageA)
                else:
                    frame_img[yi:ya, xi:xa]  =  uint8(255) - FIBSEM_frame(fls[j], ftype=ftype).RawImageA
            else:
                frame_img[yi:ya, xi:xa]  = FIBSEM_frame(fls[j], ftype=ftype).RawImageA

            if perfrom_transformation:
                transf = ProjectiveTransform(matrix = shift_matrix @ (self.tr_matr_cum_residual[j] @ inv_shift_matrix))
                frame_img_reg = warp(frame_img, transf, order = int_order, preserve_range=True)
            else:
                frame_img_reg = frame_img.copy()
        
            fig, ax = subplots(1,1, figsize = (10.0, 11.0*ysz/xsz))
            if data_minmax_exists:
                vmin = data_min_sliding[j]
                vmax = data_max_sliding[j]
            else:
                vmin, vmax = get_min_max_thresholds(frame_img_reg)
            ax.imshow(frame_img_reg, cmap='Greys', vmin=vmin, vmax=vmax)
            ax.grid(True, color = "cyan")
            ax.set_title(fls[j])
            if sliding_evaluation_box:
                xi_eval = start_evaluation_box[2] + dx_eval*j//nfrs
                yi_eval = start_evaluation_box[0] + dy_eval*j//nfrs
                if start_evaluation_box[3] > 0:
                    xa_eval = xi_eval + start_evaluation_box[3]
                else:
                    xa_eval = xsz
                if start_evaluation_box[1] > 0:
                    ya_eval = yi_eval + start_evaluation_box[1]
                else:
                    ya_eval = ysz
            rect_patch = patches.Rectangle((xi_eval,yi_eval),abs(xa_eval-xi_eval)-2,abs(ya_eval-yi_eval)-2, linewidth=2.0, edgecolor='yellow',facecolor='none')
            ax.add_patch(rect_patch)
            if save_res_png :
                fig.savefig(os.path.splitext(fls[j])[0]+'_evaluation_box.png', dpi=300)


    def estimate_SNRs(self, **kwargs):
        '''
        Estimate SNRs in Image A and Image B based on single-image SNR calculation.  

        kwargs
        ---------
        frame_inds : list of int
            List oif frame indecis to use to display the evaluation box.
            Default are [nfrs//10, nfrs//2, nfrs//10*9]
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        data_dir : str
            data direcory (path)
        Sample_ID : str
            Sample ID
        ImgB_fraction : float
            Optional fractional weight of Image B to use for constructing the fused image: FusedImage = ImageA*(1.0-ImgB_fraction) + ImageB*ImgB_fraction
            If not provided, the value determined from rSNR ratios will be used.
        invert_data : boolean
            If True - the data is inverted
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        
        '''
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        int_order = kwargs.get("int_order", self.int_order) 
        invert_data =  kwargs.get("invert_data", False)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        ImgB_fraction = kwargs.get("ImgB_fraction", 0.00 )

        fls = self.fls
        nfrs = len(fls)
        default_indecis = [nfrs//10, nfrs//2, nfrs//10*9]
        frame_inds = kwargs.get("frame_inds", default_indecis)

        test_frame = FIBSEM_frame(fls[0], ftype=ftype)

        xi = 0
        yi = 0
        xsz = test_frame.XResolution 
        xa = xi + xsz
        ysz = test_frame.YResolution
        ya = yi + ysz

        xi_eval = xi + evaluation_box[2]
        if evaluation_box[3] > 0:
            xa_eval = xi_eval + evaluation_box[3]
        else:
            xa_eval = xa
        yi_eval = yi + evaluation_box[0]
        if evaluation_box[1] > 0:
            ya_eval = yi_eval + evaluation_box[1]
        else:
            ya_eval = ya

        frame_img = np.zeros((ysz, xsz))
        xSNRAs=[]
        ySNRAs=[]
        rSNRAs=[]
        xSNRBs=[]
        ySNRBs=[]
        rSNRBs=[]

        for j in tqdm(frame_inds, desc='Analyzing Auto-Correlation SNRs '):
            if invert_data:
                if test_frame.EightBit==0:
                    frame_imgA = np.negative(FIBSEM_frame(fls[j], ftype=ftype).RawImageA)
                    if self.DetB != 'None':
                        frame_imgB = np.negative(FIBSEM_frame(fls[j], ftype=ftype).RawImageB)
                else:
                    frame_imgA  =  uint8(255) - FIBSEM_frame(fls[j], ftype=ftype).RawImageA
                    if self.DetB != 'None':
                        frame_imgB  =  uint8(255) - FIBSEM_frame(fls[j], ftype=ftype).RawImageB
                    
            else:
                frame_imgA  = FIBSEM_frame(fls[j], ftype=ftype).RawImageA
                if self.DetB != 'None':
                    frame_imgB  = FIBSEM_frame(fls[j], ftype=ftype).RawImageB

            frame_imgA_eval = frame_imgA[yi_eval:ya_eval, xi_eval:xa_eval]
            ImageA_xSNR, ImageA_ySNR, ImageA_rSNR= Single_Image_SNR(frame_imgA_eval, save_res_png=False, img_label='Image A, frame={:d}'.format(j))
            xSNRAs.append(ImageA_xSNR)
            ySNRAs.append(ImageA_ySNR)
            rSNRAs.append(ImageA_rSNR)
            if self.DetB != 'None':
                frame_imgB_eval = frame_imgB[yi_eval:ya_eval, xi_eval:xa_eval]
                ImageB_xSNR, ImageB_ySNR, ImageB_rSNR = Single_Image_SNR(frame_imgB_eval, save_res_png=False, img_label='Image B, frame={:d}'.format(j))
                xSNRBs.append(ImageB_xSNR)
                ySNRBs.append(ImageB_ySNR)
                rSNRBs.append(ImageB_rSNR)

        fig, ax = subplots(1,1, figsize = (6,4))
        ax.plot(frame_inds, xSNRAs, 'r+', label='Image A x-SNR')
        ax.plot(frame_inds, ySNRAs, 'b+', label='Image A y-SNR')
        ax.plot(frame_inds, rSNRAs, 'g+', label='Image A r-SNR')
        if self.DetB != 'None':
            ax.plot(frame_inds, xSNRBs, 'rx', linestyle='dotted', label='Image B x-SNR')
            ax.plot(frame_inds, ySNRBs, 'bx', linestyle='dotted', label='Image B y-SNR')
            ax.plot(frame_inds, rSNRBs, 'gx', linestyle='dotted', label='Image B r-SNR')
            ImgB_fraction_xSNR = np.mean(np.array(xSNRBs)/(np.array(xSNRAs) + np.array(xSNRBs)))
            ImgB_fraction_ySNR = np.mean(np.array(ySNRBs)/(np.array(ySNRAs) + np.array(ySNRBs)))
            ImgB_fraction_rSNR = np.mean(np.array(rSNRBs)/(np.array(rSNRAs) + np.array(rSNRBs)))
            if ImgB_fraction < 1e-9:
                ImgB_fraction = ImgB_fraction_rSNR
            ax.text(0.1, 0.5, 'ImgB fraction (x-SNR) = {:.4f}'.format(ImgB_fraction_xSNR), color='r', transform=ax.transAxes)
            ax.text(0.1, 0.42, 'ImgB fraction (y-SNR) = {:.4f}'.format(ImgB_fraction_ySNR), color='b', transform=ax.transAxes)
            ax.text(0.1, 0.34, 'ImgB fraction (r-SNR) = {:.4f}'.format(ImgB_fraction_rSNR), color='g', transform=ax.transAxes)

            xSNRFs=[]
            ySNRFs=[]
            rSNRFs=[]
            for j in tqdm(frame_inds, desc='Re-analyzing Auto-Correlation SNRs for fused image'):
                if invert_data:
                    if test_frame.EightBit==0:
                        frame_imgA = np.negative(FIBSEM_frame(fls[j], ftype=ftype).RawImageA)
                        if self.DetB != 'None':
                            frame_imgB = np.negative(FIBSEM_frame(fls[j], ftype=ftype).RawImageB)
                    else:
                        frame_imgA  =  uint8(255) - FIBSEM_frame(fls[j], ftype=ftype).RawImageA
                        if self.DetB != 'None':
                            frame_imgB  =  uint8(255) - FIBSEM_frame(fls[j], ftype=ftype).RawImageB
                        
                else:
                    frame_imgA  = FIBSEM_frame(fls[j], ftype=ftype).RawImageA
                    if self.DetB != 'None':
                        frame_imgB  = FIBSEM_frame(fls[j], ftype=ftype).RawImageB

                frame_imgA_eval = frame_imgA[yi_eval:ya_eval, xi_eval:xa_eval]
                frame_imgB_eval = frame_imgB[yi_eval:ya_eval, xi_eval:xa_eval]

                frame_imgF_eval = frame_imgA_eval * (1.0 - ImgB_fraction) + frame_imgB_eval * ImgB_fraction
                ImageF_xSNR, ImageF_ySNR, ImageF_rSNR = Single_Image_SNR(frame_imgF_eval, save_res_png=False, img_label='Fused, ImB_fr={:.4f}, frame={:d}'.format(ImgB_fraction, j))
                xSNRFs.append(ImageF_xSNR)
                ySNRFs.append(ImageF_ySNR)
                rSNRFs.append(ImageF_rSNR)

            ax.plot(frame_inds, xSNRFs, 'rd', linestyle='dashed', label='Fused Image x-SNR')
            ax.plot(frame_inds, ySNRFs, 'bd', linestyle='dashed', label='Fused Image y-SNR')
            ax.plot(frame_inds, rSNRFs, 'gd', linestyle='dashed', label='Fused Image r-SNR')

        else:
            ImgB_fraction_xSNR = 0.0
            ImgB_fraction_ySNR = 0.0
            ImgB_fraction_rSNR = 0.0
        ax.grid(True)
        ax.legend()
        ax.set_title(Sample_ID + '  ' + data_dir)
        ax.set_xlabel('Frame')
        ax.set_ylabel('SNR')
        if save_res_png :
            fig.savefig(os.path.splitext(fnm_reg)[0]+'SNR_evaluation_mult_frame.png', dpi=300)

        return ImgB_fraction_xSNR, ImgB_fraction_ySNR, ImgB_fraction_rSNR


    def evaluate_ImgB_fractions(self, ImgB_fractions, frame_inds, **kwargs):
        '''
        Calculate NCC and SNR vs Image B fraction over a set of frames.

        ImgB_fractions : list
            List of fractions to estimate the NCC and SNR
        frame_inds : int array
            array of frame indices to perform NCC / SNR evaluation
        
        kwargs
        ---------
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        data_dir : str
            data direcory (path)
        
        Sample_ID : str
            Sample ID
        
        invert_data : boolean
            If True - the data is inverted
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        

        Returns
        SNRimpr_max_position, SNRimpr_max, ImgB_fractions, SNRs
        '''
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        nbr = len(ImgB_fractions)

        br_results = []
        for ImgB_fraction in tqdm(ImgB_fractions, desc='Evaluating Img B fractions'):
            kwargs['ImgB_fraction'] = ImgB_fraction
            kwargs['disp_res'] = False
            kwargs['evaluation_box'] = evaluation_box
            #print('evaluate_ImgB_fractions kwargs', kwargs)
            br_res = self.transform_and_save(save_transformed_dataset=False, frame_inds=frame_inds, **kwargs)
            br_results.append(br_res)

        fig, axs = subplots(3,1, figsize=(6,8))
        fig.subplots_adjust(left=0.12, bottom=0.06, right=0.99, top=0.96, wspace=0.25, hspace=0.24)
        ncc0 = (br_results[0])['Image NCC']
        SNR0 = ncc0 / (1-ncc0)
        SNR_impr = []
        SNRs = []

        for j, (ImgB_fraction, br_result) in enumerate(zip(ImgB_fractions, br_results)):
            my_col = get_cmap("gist_rainbow_r")((nbr-j)/(nbr-1))
            ncc = br_result['Image NCC']
            SNR = ncc / (1.0-ncc)
            axs[0].plot(frame_inds[:-1], SNR, color=my_col, label = 'ImgB fraction = {:.2f}'.format(ImgB_fraction))
            axs[1].plot(frame_inds[:-1], SNR/SNR0, color=my_col, label = 'ImgB fraction = {:.2f}'.format(ImgB_fraction))
            SNR_impr.append(np.mean(SNR/SNR0))
            SNRs.append(np.mean(SNR))

        SNRimpr_max = np.max(SNR_impr)
        SNRimpr_max_ind = np.argmax(SNR_impr)
        ImgB_fraction_max = ImgB_fractions[SNRimpr_max_ind]
        xi = max(0, (SNRimpr_max_ind-3))
        xa = min((SNRimpr_max_ind+3), len(ImgB_fractions))
        ImgB_fr_range = ImgB_fractions[xi : xa]
        SNRimpr_range = SNR_impr[xi : xa]
        popt = np.polyfit(ImgB_fr_range, SNRimpr_range, 2)
        SNRimpr_fit_max_pos = -0.5 * popt[1] / popt[0]
        ImgB_fr_fit = np.linspace(ImgB_fr_range[0], ImgB_fr_range[-1], 21)
        SNRimpr_fit = np.polyval(popt, ImgB_fr_fit)
        if popt[0] < 0 and SNRimpr_fit_max_pos > ImgB_fractions[0] and SNRimpr_fit_max_pos<ImgB_fractions[-1]:
            SNRimpr_max_position = SNRimpr_fit_max_pos
            SNRimpr_max = np.polyval(popt, SNRimpr_max_position)
        else: 
            SNRimpr_max_position = ImgB_fraction_max
        fs=11
        axs[0].grid(True)
        axs[0].set_ylabel('Frame-to-Frame SNR', fontsize=fs)
        axs[0].set_xlabel('Frame', fontsize=fs)  
        axs[0].legend(fontsize=fs-1)
        axs[0].set_title(Sample_ID + '  ' + data_dir, fontsize=fs)
        axs[1].grid(True)
        axs[1].set_ylabel('Frame-to-Frame SNR Improvement', fontsize=fs)
        axs[1].set_xlabel('Frame', fontsize=fs)   
        axs[2].plot(ImgB_fractions, SNR_impr, 'rd', label='Data')
        axs[2].plot(ImgB_fr_fit, SNRimpr_fit, 'b', label='Fit')
        axs[2].plot(SNRimpr_max_position, SNRimpr_max, 'gd', markersize=10, label='Max SNR Improvement')
        axs[2].text(0.1, 0.5, 'Max SNR Improvement={:.3f}'.format(SNRimpr_max), transform=axs[2].transAxes, fontsize=fs)
        axs[2].text(0.1, 0.4, '@ Img B Fraction ={:.3f}'.format(SNRimpr_max_position), transform=axs[2].transAxes, fontsize=fs)
        axs[2].legend(fontsize=fs)
        axs[2].grid(True)
        axs[2].set_ylabel('Mean SNR improvement', fontsize=fs)
        axs[2].set_xlabel('Image B fraction', fontsize=fs)

        if save_res_png :
            fname_image = os.path.join(data_dir, os.path.splitext(fnm_reg)[0]+'_SNR_vs_ImgB_ratio_evaluation.png')
            fig.savefig(fname_image, dpi=300)

        return SNRimpr_max_position, SNRimpr_max, ImgB_fractions, SNRs