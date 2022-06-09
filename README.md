# This is a repository for FIB-SEM data processing

## "Register_FIB-SEM_stack_DASK_v1.ipynb" - Python Notebook for perfroming FIB-SEM stack registration (uses SIFT package in OpenCV, DASK package and few other)

## "Register_FIB-SEM_stack_DASK_v1_AMST.ipynb" - Python Notebook which performs FIB-SEM stack registration of AMST dataset and compares the results with AMST registration [1]
1. J. Hennies et al, "AMST: Alignment to Median Smoothed Template for Focused Ion Beam Scanning Electron Microscopy Image Stacks", Sci. Rep. 10, 2004 (2020).

## "Evaluate_FIB-SEM_MRC_stack_registrations.ipynb" - Python Notebook for evaluating FIB-SEM stack registration (works with stacks saved into MRC files, uses DASK)

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
-	DASK
-	pickle
-	webbrowser
-	IPython
-   cupy (you will need to have CUDA toolkit installed first)


## General Help Functions
    get_spread(data, window=501, porder=3):
        Calculates spread - standard deviation of the (signal - Sav-Gol smoothed signal)
    get_min_max_thresholds(image, thr_min=1e-3, thr_max=1e-3, nbins=256, disp_res=False)
        Determines the data range (min and max) with given fractional thresholds for cumulative distribution.
    radial_profile(data, center)
        Calculates radially average profile of the 2D array (used for FRC and auto-correlation)
    radial_profile_select_angles(data, center, astart = 89, astop = 91, symm=4)
        Calculates radially average profile of the 2D array (used for FRC) within a select range of angles.
    smooth(x, window_len=11, window='hanning')
        smooth the data using a window with requested size.


## Single Frame Image Processing Functions
    Single_Image_SNR(img, **kwargs):
        Estimates SNR based on a single image.
        Calculates SNR of a single image base on auto-correlation analysis after [1].   
        [1] J. T. L. Thong et al, Single-image signal-to-noise ratio estimation. Scanning, 328–336 (2001).
    Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs)
        Analyses the noise statistics in the selected ROI's of the EM data
    Single_Image_Noise_Statistics(img, **kwargs)
        Analyses the noise statistics of the EM data image.
    Perform_2D_fit(img, estimator, **kwargs)
        Bin the image and then perform 2D polynomial (currently only 2D parabolic) fit on the binned image.


## Two-Frame Image Processing Functions
    mutual_information_2d(x, y, sigma=1, bin=256, normalized=False)
        Computes (normalized) mutual information between two 1D variate from a joint histogram.
    mutual_information_2d_cp(x, y, sigma=1, bin=256, normalized=False)
        Computes (normalized) mutual information between two 1D variate from a joint histogram using CUPY package.
    Two_Image_NCC_SNR(img1, img2, **kwargs)
        Estimate normalized cross-correlation and SNR of two images. After:
        [1] J. Frank, L. AI-Ali, Signal-to-noise ratio of electron micrographs obtained by cross correlation. Nature 256, 4 (1975).
        [2] J. Frank, in: Computer Processing of Electron Microscopic Images. Ed. P.W. Hawkes (Springer, Berlin, 1980).
        [3] M. Radermacher, T. Ruiz, On cross-correlations, averages and noise in electron microscopy. Acta Crystallogr. Sect. F Struct. Biol. Commun. 75, 12–18 (2019).
    Two_Image_FSC(img1, img2, **kwargs)
        Perform Fourier Shell Correlation to determine the image resolution, after [1].
        [1] M. van Heela, and M. Schatzb, "Fourier shell correlation threshold criteria," Journal of Structural Biology 151, 250-262 (2005)


## MRC stack evaluation Functions
    analyze_mrc_stack_registration(mrc_filename, DASK_client, **kwargs)
        Read MRC stack and analyze registration - calculate NSAD, NCC, and MI.
    show_eval_box_mrc_stack(mrc_filename, **kwargs)
        Read MRC stack and display the eval box for each frame from the list.


## Helper Functions for Results Presentation
    plot_registrtion_quality_csvs(data_files, labels, **kwargs):
        Read and plot together multiple registration quality summaries.


## class FIBSEM_frame:
    A class representing single FIB-SEM data frame. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com.
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
        number of pixels - frame size in horisontal direction
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
        Analyses the noise statistics in the selected ROI's of the EM data.
    analyze_noise_statistics(**kwargs):
        Analyses the noise statistics of the EM data image.
    analyze_SNR_autocorr(image_name = 'RawImageA', **kwargs):
        Estimates SNR using auto-correlation analysis of a single image.
    show_eval_box(**kwargs):
        Show the box used for evaluating the noise
    determine_field_fattening_parameters(image_name = 'ImageA', **kwargs):
        Perfrom 2D parabolic fit (calls Perform_2D_fit(Img, estimator, **kwargs)) and determine the field-flattening parameters
    flatten_image(image_name = 'ImageA', **kwargs):
        Flatten the image


## class FIBSEM_dataset: 
    A class representing a FIB-SEM data set. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com.
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
            evaluation_box = [left, width, top, height] boundaries of the box used for evaluating the image registration.
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
    def evaluate_ImgB_fractions(ImgB_fractions, frame_inds, **kwargs):
        Calculate NCC and SNR vs Image B fraction over a set of frames.

