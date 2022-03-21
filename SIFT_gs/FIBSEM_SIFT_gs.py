import numpy as np
import matplotlib
from matplotlib import pylab, mlab, pyplot
plt = pyplot
from IPython.core.pylabtools import figsize, getfigs
from pylab import *
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image as PILImage

import os
import time
import glob
import numpy as np
import cupy as cp
import pandas as pd
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


from scipy import ndimage
EPS = np.finfo(float).eps


def get_min_max_thresholds(image, thr_min=1e-3, thr_max=1e-3, nbins=256, disp_res=False):
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
    nmi: float
        the computed similariy measure
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
    nmi: float
        the computed similariy measure
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


class FIBSEM_frame:     # class containing a single FIB-SEM frame data and functions to do simple operations.
    def __init__(self, fname, **kwargs):   
        self.fname = fname
        self.ftype = kwargs.get("ftype", 0) # ftype=0 - Shan Xu's binary format  ftype=1 - tifstack
        self.use_dask_arrays = kwargs.get("use_dask_arrays", False)

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
                
                if self.FileVersion > 8 :
                    self.SampleID = (unpack('25s',self.header[155:180])[0]).decode("utf-8") # Read in Sample ID
                
                self.Notes = (unpack('200s',self.header[180:380])[0]).decode("utf-8")       # Read in notes

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
                    dt = np.dtype(np.int8)
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
                        self.RawImageA = (Raw[:,:,0].astype(float32)*self.ScanRate/self.Scaling[0,0]/self.Scaling[2,0]/self.Scaling[3,0]+self.Scaling[1,0]).astype(int16)
                        if self.AI2 ==2:
                            self.RawImageB = (Raw[:,:,1].astype(float32)*self.ScanRate/self.Scaling[0,1]/self.Scaling[2,1]/self.Scaling[3,1]+self.Scaling[1,1]).astype(int16)
                    elif self.AI2 ==2:
                        self.RawImageB = (Raw[:,:,0].astype(float32)*self.ScanRate/self.Scaling[0,0]/self.Scaling[2,0]/self.Scaling[3,0]+self.Scaling[1,0]).astype(int16)
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
             print('SampleID=', self.SampleID)
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
        fig, axs = subplots(2, 1, figsize=(10,5))
        axs[0].imshow(self.RawImageA, cmap='Greys')
        axs[1].imshow(self.RawImageB, cmap='Greys')
        ttls = ['Detector A: '+self.DetA.strip('\x00'), 'Detector B: '+self.DetB.strip('\x00')]
        for ax, ttl in zip(axs, ttls):
            ax.axis(False)
            ax.set_title(ttl, fontsize=10)
        fig.suptitle(self.fname)
        
    def save_images_jpeg(self, invert=False, images_to_save = 'Both'):
        if images_to_save == 'Both' or imagees_to_save == 'A':
            fname_jpg = self.fname.replace('.dat', '_' + self.DetA.strip('\x00') + '.jpg')
            Img = self.RawImageA_8bit_thresholds()[0]
            if invert:
                Img = 255 - Img
            PILImage.fromarray(Img).save(fname_jpg)

        if images_to_save == 'Both' or imagees_to_save == 'B':
            fname_jpg = self.fname.replace('.dat', '_' + self.DetB.strip('\x00') + '.jpg')
            Img = self.RawImageB_8bit_thresholds()[0]
            if invert:
                Img = 255 - Img
            PILImage.fromarray(Img).save(fname_jpg)


    def save_images(self, images_to_save = 'Both'):
        if images_to_save == 'Both' or imagees_to_save == 'A':
            fnameA = self.fname.replace('.dat', '_' + self.DetA.strip('\x00') + '.tif')
            tiff.imsave(fnameA, self.RawImageA)
        if images_to_save == 'Both' or imagees_to_save == 'B':
            fnameB = self.fname.replace('.dat', '_' + self.DetB.strip('\x00') + '.tif')
            tiff.imsave(fnameB, self.RawImageB)
        
    def get_image_min_max(self, image_name = 'ImageA', thr_min = 1.0e-4, thr_max = 1.0e-3, nbins=256, disp_res = False):
        if image_name == 'ImageA':
            im = self.ImageA
        if image_name == 'ImageB':
            im = self.ImageB
        if image_name == 'RawImageA':
            im = self.RawImageA
        if image_name == 'RawImageB':
            im = self.RawImageB
        return get_min_max_thresholds(im, thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)

    
    def RawImageA_8bit_thresholds(self, thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
        if self.EightBit==1:
            print('8-bit image already - no need to convert')
            dt = self.RawImageA
        else:
            if data_min == data_max:
                data_min, data_max = self.get_image_min_max(image_name ='RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
            dt = ((np.clip(self.RawImageA, data_min, data_max) - data_min)/(data_max-data_min)*255.0).astype(np.uint8)
        return dt, data_min, data_max

    
    def RawImageB_8bit_thresholds(self, thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
        if self.EightBit==1:
            print('8-bit image already - no need to convert')
            dt = self.RawImageB
        else:
            if data_min == data_max:
                data_min, data_max = self.get_image_min_max(image_name ='RawImageB', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
            dt = ((np.clip(self.RawImageB, data_min, data_max) - data_min)/(data_max-data_min)*255.0).astype(np.uint8)
        return dt, data_min, data_max
    
    def save_snapshot(self, display = True, dpi=300, thr_min = 1.0e-3, thr_max = 1.0e-3, nbins=256):
        fig, axs = subplots(3, 1, figsize=(11,8))
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.15, hspace=0.1)
        dminA, dmaxA = self.get_image_min_max(image_name ='RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
        axs[1].imshow(self.RawImageA, cmap='Greys', vmin=dminA, vmax=dmaxA)
        dminB, dmaxB = self.get_image_min_max(image_name ='RawImageB', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
        axs[2].imshow(self.RawImageB, cmap='Greys', vmin=dminB, vmax=dmaxB)
        ttls = [self.Notes.strip('\x00'),
                'Detector A:  '+ self.DetA.strip('\x00') + ',  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}'.format(dminA, dmaxA, thr_min, thr_max) + '    (Brightness: {:.1f}, Contrast: {:.1f})'.format(self.BrightnessA, self.ContrastA),
                'Detector B:  '+ self.DetB.strip('\x00') + ',  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}'.format(dminB, dmaxB, thr_min, thr_max) + '    (Brightness: {:.1f}, Contrast: {:.1f})'.format(self.BrightnessB, self.ContrastB)]
        for ax, ttl in zip(axs, ttls):
            ax.axis(False)
            ax.set_title(ttl, fontsize=10)
        fig.suptitle(self.fname)
        
        if self.FileVersion > 8:
            cell_text = [['Sample ID', '{:s}'.format(self.SampleID.strip('\x00')), '',
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

        fig.savefig(self.fname.replace('.dat', '_snapshot.png'), dpi=dpi)
        if display == False:
            plt.close(fig)

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
    Parameters
    ----------
    points : (N, D) array
        The coordinates of the image points.
    Returns
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

def determine_regularized_affine_transform(src_pts, dst_pts, l2_matrix = None, targ_vector = None):
    """Estimate N-D affine transformation with regularization from a set of corresponding points.
        G.Shtengel 11/2021
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
    """Estimate N-D similarity transformation with or without scaling.
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
    """ScaleShift transformation.
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
    '''ScaleShift transformation.
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
    '''ScaleShift transformation.
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
    """Affine transformation.
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


def kp_to_list(kp):  # converts a keypont object to a list
    x, y = kp.pt
    pt = float(x), float(y)
    angle = float(kp.angle) if kp.angle is not None else None
    size = float(kp.size) if kp.size is not None else None
    response = float(kp.response) if kp.response is not None else None
    class_id = int(kp.class_id) if kp.class_id is not None else None
    octave = int(kp.octave) if kp.octave is not None else None
    return pt, angle, size, response, class_id, octave

def list_to_kp(inp_list):   # # converts a list to a keypont object
    kp = cv2.KeyPoint()
    kp.pt = inp_list[0]
    kp.angle = inp_list[1]
    kp.size = inp_list[2]
    kp.response = inp_list[3]
    kp.octave = inp_list[4]
    kp.class_id = inp_list[5]
    return kp

def get_min_max_thresholds_file(params):
    fl, thr_min, thr_max, nbins = params
    dmin, dmax = FIBSEM_frame(fl).get_image_min_max(image_name = 'RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins)
    return [dmin, dmax]

def extract_keypoints_descr(params):
    fl, dmin, dmax = params
    sift = cv2.xfeatures2d.SIFT_create()
    img, d1, d2 = FIBSEM_frame(fl).RawImageA_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = dmin, data_max = dmax, nbins=256)
    # extract keypoints and descriptors for both images
    kps, dess = sift.detectAndCompute(img, None)
    key_points = [KeyPoint(kp) for kp in kps]
    return key_points, dess

def extract_keypoints_descr_files(params):
    fl, dmin, dmax, kp_max_num = params
    sift = cv2.xfeatures2d.SIFT_create()
    img, d1, d2 = FIBSEM_frame(fl).RawImageA_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = dmin, data_max = dmax, nbins=256)
    # extract keypoints and descriptors for both images
    kps, dess = sift.detectAndCompute(img, None)
    if kp_max_num != -1 and (len(kps) > kp_max_num):
        kp_ind = np.argsort([-kp.response for kp in kps])[0:kp_max_num]
        kps = np.array(kps)[kp_ind]
        dess = np.array(dess)[kp_ind]
    #key_points = [KeyPoint(kp) for kp in kps]
    key_points = [kp_to_list(kp) for kp in kps]
    kpd = [key_points, dess]
    fnm = fl.replace('.dat', '_kpdes.bin')
    pickle.dump(kpd, open(fnm, 'wb')) # converts array to binary and writes to output
    #pickle.dump(dess, open(fnm, 'wb')) # converts array to binary and writes to output
    return fnm

def estimate_kpts_transform_error(src_pts, dst_pts, transform_matrix):
    """ image transformation matrix in a form:
        A = [[a0  a1   a2]
             [b0   b1  b2]
             [0   0    1]]
     Thransofrmation is supposed to be in a form:
     Xnew = a0 * Xoriginal + a1 * Yoriginal + a2
     Ynew = b0 * Xoriginal + b1 * Yoriginal + b2
     source and destination points are pairs of coordinates (2xN array)
     errors are estimated as norm(dest_pts - A*src_pts) so that least square regression can be performed
    """
    src_pts_transformed = src_pts @ transform_matrix[0:2, 0:2].T + transform_matrix[0:2, 2]
    return np.linalg.norm(dst_pts - src_pts_transformed, ord=2, axis=1)


def determine_transformation_matrix(src_pts, dst_pts, TransformType, dr_max = 2, max_iter = 100):
    '''    G.Shtengel, 09/2021
    Determine the transformation matrix in a form:
            A = [[a0  a1   a2]
                 [b0   b1  b2]
                 [0   0    1]]
    based on the given source and destination points using linear regression such that the error is minimized for 
    sum(dst_pts - A*src_pts).
    
    For each matched pair of keypoins the error is calculated as err[j] = dst_pts[j] - A*src_pts[j]
    The iterative procedure throws away the matched keypoint pair with worst error on every iteration
    untill the worst error falls below dr_max or the max number of iterations is reached.
    '''
    transform_matrix = np.eye(3,3)
    iteration = 1
    max_error = dr_max * 2.0
    errors = []
    while iteration <= max_iter and max_error > dr_max:
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
        #print('Iteration {:d}, max_error={:.2f} '.format(iteration, max_error), (iteration <= max_iter), (max_error > dr_max))
        iteration +=1
    kpts = [src_pts, dst_pts]
    error_abs_mean = np.mean(np.abs(np.delete(errs, ind, axis=0)))
    return transform_matrix, kpts, error_abs_mean, iteration


def determine_transformations_files(params_dsf):
    '''    G.Shtengel, 09/2021
    this is a faster version of the procedure - it loads the keypoints and matches for each frame from files.
    params_dsf = fnm_1, fnm_2, TransformType, BF_Matcher, solver, dr_max, max_iter, save_matches
    where
    fnm_1 - keypoints for the first image (source)
    fnm_2 - keypoints for the first image (destination)
    TransformType - transformation type to be used (ShiftTransform, XScaleShiftTransform, ScaleShiftTransform, AffineTransform)
    BF_Matcher -  if True - use BF matcher, otherwise use FLANN matcher for keypoint matching
    solver - a string indicating which solver to use:
    'LinReg' will use Linear Regression with iterative "Throwing out the Worst Residual" Heuristic
    'RANSAC' will use RANSAC (Random Sample Consensus) algorithm.
    dr_max - in the case of 'LinReg' - outlier threshold for iterative regression
           - in the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
    max_iter - max number of iterations
    save_matches - if True - save the matched keypoints into a binary dump file
    '''
    
    fnm_1, fnm_2, TransformType, l2_matrix, targ_vector, BF_Matcher, solver, dr_max, max_iter, save_matches = params_dsf

    if TransformType == RegularizedAffineTransform:

        def estimate(self, src, dst):
            self.params = determine_regularized_affine_transform(src, dst, l2_matrix, targ_vector)
        RegularizedAffineTransform.estimate = estimate

    kpp1s, des1 = pickle.load(open(fnm_1, 'rb'))
    kpp2s, des2 = pickle.load(open(fnm_2, 'rb'))
    
    kp1 = [list_to_kp(kpp1) for kpp1 in kpp1s]     # this converts a list of lists to a list of keypoint objects to be used by a matcher later
    kp2 = [list_to_kp(kpp2) for kpp2 in kpp2s]     # same for the second frame
    
    # establish matches
    if BF_Matcher:    # if BFMatcher==True - use BF (Brute Force) matcher
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
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)
    
    if solver == 'LinReg':
        # Determine the transformation matrix via iterative liear regression
        transform_matrix, kpts, error_abs_mean, iteration = determine_transformation_matrix(src_pts, dst_pts, TransformType, dr_max = dr_max, max_iter = max_iter)
        n_kpts = len(kpts[0])
    else:  # the other option is solver = 'RANSAC'
        try:
            min_samples = len(src_pts)//20
            model, inliers = ransac((src_pts, dst_pts),
                TransformType, min_samples=min_samples,
                residual_threshold=dr_max, max_trials=10000)
            n_inliers = np.sum(inliers)
            inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
            inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
            src_pts_ransac = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            dst_pts_ransac = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
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



def build_filename(frame0, save_asI8, zbin_2x, TransformType, solver, drmax, preserve_scales, fit_prms, subtract_linear_fit, pad_edges, suffix=''):
    frame = FIBSEM_frame(frame0)
    dformat_read = 'I8' if frame.EightBit else 'I16'
    save_asI8_save = save_asI8 or frame.EightBit==1

    if save_asI8_save:
        dtp = np.int8
        dformat_save = 'I8'
        mrc_mode = 0
        if zbin_2x:
            fnm_reg = 'Registered_I8_zbin2x.mrc'
        else:
            fnm_reg = 'Registered_I8.mrc'
    else:
        dtp = np.int16
        dformat_save = 'I16'
        mrc_mode = 1
        if zbin_2x:
            fnm_reg = 'Registered_I16_zbin2x.mrc'
        else:
            fnm_reg = 'Registered_I16.mrc'
            
    fnm_reg = fnm_reg.replace('.mrc', ('_' + TransformType.__name__ + '_' + solver + '.mrc'))

    fnm_reg = fnm_reg.replace('.mrc', '_drmax{:.1f}.mrc'.format(drmax))
  
    if preserve_scales:
        fnm_reg = fnm_reg.replace('.mrc', '_const_scls_'+fit_prms[0]+'.mrc')

    if subtract_linear_fit:
        fnm_reg = fnm_reg.replace('.mrc', '_shift_subtr.mrc')
    
    if pad_edges:
        fnm_reg = fnm_reg.replace('.mrc', '_padded.mrc')

    if len(suffix)>0:
        fnm_reg = fnm_reg.replace('.mrc', '_' + suffix + '.mrc')
    return fnm_reg, mrc_mode, dtp

def find_fit(tr_matr_cum, fit_prms):
    fit_method = fit_prms[0]
    if fit_method == 'SG':  # perform Savitsky-Golay fitting with parameters
        ws, porder = fit_prms[1:3]         # window size 701, polynomial order 3
        s00_fit = savgol_filter(tr_matr_cum[:, 0, 0].astype(double), ws, porder)
        s01_fit = savgol_filter(tr_matr_cum[:, 0, 1].astype(double), ws, porder)
        s10_fit = savgol_filter(tr_matr_cum[:, 1, 0].astype(double), ws, porder)
        s11_fit = savgol_filter(tr_matr_cum[:, 1, 1].astype(double), ws, porder)
    else:
        fr = np.arange(0, len(tr_matr_cum), dtype=np.double)
        if fit_method == 'PF':  # perform polynomial fitting with parameters
            porder = fit_prms[1]         # polynomial order
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

def process_transf_matrix(transformation_matrix, fnms_matches, npts, error_abs_mean, Sample_ID, data_min_glob, data_max_glob,
                          TransformType, l2_matrix, targ_vector, solver, kp_max_num, dr_max, max_iter, preserve_scales, fit_prms, subtract_linear_fit, pad_edges,
                          data_dir, fnm_reg, save_plot=True, add_suff = ''):

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
    axs5[0,0].text(-0.1, 0.73, 'Global Data Range:  Min={:.2f}, Max={:.2f}'.format(data_min_glob, data_max_glob), transform=axs5[0,0].transAxes, fontsize = fs)
    
    if TransformType == RegularizedAffineTransform:
        tstr = ['{:d}'.format(x) for x in targ_vector] 
        otext = 'Reg.Aff.Transf., λ= {:.1e}, t=['.format(l2_matrix[0,0]) + ' '.join(tstr) + '], w/' + solver
    else:
        otext = TransformType.__name__ + ' with ' + solver + ' solver'

    axs5[0,0].text(-0.1, 0.56, otext, transform=axs5[0,0].transAxes, fontsize = fs)

    sbtrfit = 'ON' if subtract_linear_fit else 'OFF'
    axs5[0,0].text(-0.1, 0.39, 'drmax={:.1f}, Max # of KeyPts={:d}, Max # of Iter.={:d}'.format(dr_max, kp_max_num, max_iter), transform=axs5[0,0].transAxes, fontsize = fs)
    padedges = 'ON' if pad_edges else 'OFF'
    if preserve_scales:
        fit_method = fit_prms[0]
        if fit_method == 'LF':
            fit_str = ', Meth: Linear Fit'
            fm_string = 'linear'
        else:
            if fit_method == 'SG':
                fit_str = ', Meth: Sav.-Gol., ' + str(fit_prms[1:])
                fm_string = 'Sav.-Gol.'
            else:
                fit_str = ', Meth: Pol.Fit, ord.={:d}'.format(fit_prms[1])
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

        tr_matr_cum, s_fits = find_fit(tr_matr_cum, fit_prms)
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


    # Subtract linear trend from offsets
    if subtract_linear_fit:
        fr = np.arange(0, len(Xshift_cum) )
        pX = np.polyfit(fr, Xshift_cum, 1)
        Xfit = np.polyval(pX, fr)
        pY = np.polyfit(fr, Yshift_cum, 1)
        Yfit = np.polyval(pY, fr)
        Xshift_residual = Xshift_cum - Xfit
        Yshift_residual = Yshift_cum - Yfit
        #Xshift_residual0 = -np.polyval(pX, 0.0)
        #Yshift_residual0 = -np.polyval(pY, 0.0)
    else:
        Xshift_residual = Xshift_cum.copy()
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
    if subtract_linear_fit:
        axs5[2, 2].plot(Xfit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Tx cum. - lin. fit')
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
    if save_plot:
        fig5.savefig(fn.replace('.mrc', ('_Transform_Summary'+add_suff+'.png')), dpi=300)
        
    return tr_matr_cum

def determine_pad_offsets(shape, tr_matr):
    ysz, xsz = shape
    xmins = np.zeros(len(tr_matr))
    xmaxs = xmins.copy()
    ymins = xmins.copy()
    ymaxs = xmins.copy()
    corners = np.array([[0,0], [0, ysz], [xsz, 0], [xsz, ysz]])
    for j, trm in enumerate(tqdm(tr_matr, desc = 'Determining the pad offsets')):
        a = (trm[0:2, 0:2] @ corners.T).T + trm[0:2, 2]
        xmins[j] = np.min(a[:, 0])
        xmaxs[j] = np.max(a[:, 0])
        ymins[j] = np.min(a[:, 1])
        ymaxs[j] = np.max(a[:, 1])
    return np.min(xmins), np.max(xmaxs)-xsz, np.min(ymins), np.max(ymaxs)-ysz

def correlation_coefficient(frame1, frame2):
    product = ((frame1 - frame1.mean()) * (frame2 - frame2.mean())).mean()
    stds = frame1.std() * frame2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

# This is a function used for selecting proper threshold and kp_max_num parameters for SIFT processing

def SIFT_evaluation(data_dir, fs, threshold_min, threshold_max, nbins, TransformType, l2_matrix, targ_vector,
    solver, drmax, max_iter, kp_max_num=-1,
    save_image_png=True):

    fnm = os.path.join(data_dir,fs[0])
    frame = FIBSEM_frame(fnm)
    if frame.FileVersion > 8 :
        sampleID = frame.SampleID.strip('\x00')
    else:
        sampleID = frame.Notes[0:16]
    print(sampleID)
    
    #if save_image_png:
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
        minmax.append(FIBSEM_frame(os.path.join(data_dir,f)).get_image_min_max(image_name = 'RawImageA', thr_min=threshold_min, thr_max=threshold_max, nbins=nbins))
    dmin = np.min(np.array(minmax))
    dmax = np.max(np.array(minmax))
    #print('data range: ', dmin, dmax)
    
    t0 = time.time()
    params1 = [fs[0], dmin, dmax, kp_max_num]
    fnm_1 = extract_keypoints_descr_files(params1)
    params2 = [fs[1], dmin, dmax, kp_max_num]
    fnm_2 = extract_keypoints_descr_files(params2)
    
    BF_Matcher = False
    save_matches = False
    params_dsf = [fnm_1, fnm_2, TransformType, l2_matrix, targ_vector, BF_Matcher, solver, drmax, max_iter, save_matches]
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
    
    fig.suptitle(sampleID + ',  thr_min={:.0e}, thr_max={:.0e}, kp_max_num={:d}, comp.time={:.1f}sec'.format(threshold_min, threshold_max, kp_max_num, comp_time), fontsize=fszl)

    if TransformType == RegularizedAffineTransform:
        tstr = ['{:d}'.format(x) for x in targ_vector] 
        otext =  TransformType.__name__ + ', λ= {:.1e}, t=['.format(l2_matrix[0,0]) + ', '.join(tstr) + '], ' + solver + ', #of matches={:d}'.format(n_matches)
    else:
        otext = TransformType.__name__ + ', ' + solver + ', #of matches={:d}'.format(n_matches)

    axs[0,0].text(0.01, 1.14, otext, fontsize=fszl, transform=axs[0,0].transAxes)        
    if save_image_png:
        fig.savefig(fnm.replace('.dat', '_SIFT_eval_'+TransformType.__name__ + '_' + solver +'_thr_min{:.5f}_thr_max{:.5f}.png'.format(threshold_min, threshold_max)), dpi=300)
            
    xfsz = np.int(7 * frame.XResolution / np.max([frame.XResolution, frame.YResolution]))+1
    yfsz = np.int(7 * frame.YResolution / np.max([frame.XResolution, frame.YResolution]))+2
    fig2, ax = subplots(1,1, figsize=(xfsz,yfsz))
    fig2.subplots_adjust(left=0.0, bottom=0.25*(1-frame.YResolution/frame.XResolution), right=1.0, top=1.0)
    #fig2.subplots_adjust(left=0.0, bottom=0.00, right=1.0, top=1.0)
    symsize = 2
    fsize = 12
    img2 = FIBSEM_frame(os.path.join(data_dir,fs[-1])).RawImageA
    ax.imshow(img2, cmap='Greys', vmin=dmin, vmax=dmax)
    ax.axis(False)
    x, y = dst_pts_filtered.T
    M = sqrt(xshifts*xshifts+yshifts*yshifts)
    xs = xshifts
    ys = yshifts
    
    # the code below is for scatter points overlayed over the map - not used here
    #cax = ax.scatter(x, y, s=symsize, c=M, marker='s', cmap="plasma")
    #cbar = fig2.colorbar(cax, ax=ax, pad=0.05, shrink=0.70, orientation = 'horizontal', format="%.1f")
    #cbar.ax.tick_params(labelsize=fsize)
    #cbar.set_label('SIFT Shift Amplitude (pix)', fontsize=fsize)
    
    # the code below is for vector map. vectors have origin coordinates x and y, and vector projections xs and ys.
    vec_field = ax.quiver(x,y,xs,ys,M, scale=40, width =0.003, cmap='jet')
    cbar = fig2.colorbar(vec_field, cmap='jet', pad=0.05, shrink=0.70, orientation = 'horizontal', format="%.1f")
    cbar.set_label('SIFT Shift Amplitude (pix)', fontsize=fsize)

    ax.text(0.01, 1.1-0.13*frame.YResolution/frame.XResolution, sampleID + ', thr_min={:.0e}, thr_max={:.0e}, kp_max_num={:d},  #of matches={:d}'.format(threshold_min, threshold_max, kp_max_num, n_matches), fontsize=fsize, transform=ax.transAxes)
            
    if save_image_png:
        fig2_fnm = os.path.join(data_dir, 'SIFT_vmap_'+TransformType.__name__ + '_' + solver +'_thr_min{:.0e}_thr_max{:.0e}_kp_max{:d}.png'.format(threshold_min, threshold_max, kp_max_num))
        fig2.savefig(fig2_fnm, dpi=300)

    return(dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts)