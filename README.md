# This is a repository for FIB-SEM data processing

## "Register_FIB-SEM_stack_DASK_v0.ipynb" - Python Notebook for perfroming FIB-SEM stack registration (uses SIFT package in OpenCV, DASK package and few other)

In order to run the Python Notebook code, first, install basic Anaconda:
https://www.anaconda.com/products/individual
This notebook uses OpenCV implementation of SIFT. SIFT is part of standard OpenCV releases for version 3.4.1 or earlier. If you have newer version of OpenCV-python installed, SIFT will most likely be not part of it (because of patent issues), and the Python command sift = cv2.xfeatures2d.SIFT_create() will generate error. In this case replace it with a version supporting SIFT using these commands (in anaconda command window):

Uninstall the OpenCV:
>pip uninstall opencv-python
Then install the contrib version of OpenCV:
>pip install opencv-contrib-python

You will also need to have these packages installed:
-	mrcfile
-	skimage
-	dask
-	pickle
-	webbrowser
-	IPython


## Class FIBSEM
The FIB-SEM data stored during the imaging into binary “.dat” files that contain the header (first 1024 bytes) and the FIB-SEM signal, typically from two detectors.
The class FIBSEM_frame initializes an object by reading a FIB-SEM “.dat” file and creating the object with methods performing access to the header information and to the data:
__init__(filename.dat) reads the filename.dat file and unpacks the header and image information.
Example: 
file_name_v8 = 'Merlin-6257_20-02-16_172032_0-0-0.dat'
data_dir_v8 = 'F:\FIB-SEM_SIFT\LID494_ROI5_RawData'
fname_v8 = os.path.join(data_dir_v8,file_name_v8)
frame_v8 = FIBSEM_frame(fname_v8)
The last line creates the object frame_v8 and we can then use that object with following methods.

print_header() prints the information encoded into the first 1024 bytes of the .dat file.

save_snapshot(self, display , dpi, nbins, thr_min, thr_max) creates a snapshot of the frame as shown below, which includes the detector images and some information from the header.

display_images() displays auto-scaled detector images within a python notebook without saving the figure into the file.

save_images_jpeg(invert=False, images_to_save = 'Both') saves autoscaled detector images as a figure in JPG file, (replacing the “.dat” extension with ‘.jpeg’). If the parameter invert is set to False (default), the images will not be inverted – and membranes will be “white”. The parameter images_to_save can be set to 'Both' (default), 'A', or 'B'.

save_images(images_to_save = 'Both') saves the images captured by Detector A and Detector B individually as .tif files and adds the detector names (extracted from the FIB-SEM header) to the filenames. The parameter images_to_save can be set to 'Both' (default), 'A', or 'B'.

get_image_min_max(image_name = 'ImageA', thr_min = 1.0e-4, thr_max = 1.0e-3, nbins=256, disp_res = False) calculates histogram of pixel intensities of a specified image (options are 'ImageA', 'ImageB', 'RawImageA', 'RawImageB',  default is 'ImageA') with number of bins determined by parameter nbins (default = 256) and normalizes it to get the probability distribution function (PDF), from which a cumulative distribution function (CDF) is calculated. Then given the thr_min, thr_max parameters, the minimum and maximum values for the image are found by finding the intensities at which CDF= thr_min and (1- thr_max), respectively.

RawImageA_8bit_thresholds(thr_min, thr_max, data_min, data_max, nbins) and RawImageB_8bit_thresholds (thr_min, thr_max, data_min, data_max, nbins) will first check if the image is already in 8-bit form, if not, it will convert the image to 8-bit form using the minimum (data_min) and maximum(data_max), if those are given, and the thresholds (using the same procedure as explained above), if the data_min and data_max are not provided (or are equal – default values are set to -1 for each).

