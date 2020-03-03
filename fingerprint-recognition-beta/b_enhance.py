# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2020
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Fingerprint Recognition
# 02. Module to enhance fingerprint samples, aiming at further minutiae detection.
# Language: Python 3
# Needed libraries: NumPy (https://numpy.org/), OpenCV (https://opencv.org/),
# SciPy (https://www.scipy.org/) and Scikit-Image (https://scikit-image.org/docs/dev/api/skimage.html).
# Quick install (with PyPI - https://pypi.org/): execute, on command shell (each line at a time):
# "pip3 install numpy";
# "pip3 install opencv-contrib-python==3.4.2.17";
# "pip3 install scipy";
# "pip3 install scikit-image".


import math
import numpy
import cv2
import scipy.ndimage
import skimage.morphology
from skimage.filters import gabor_kernel

#####################################################
# Fingerprint-enhancement configuration parameters. #
#####################################################

# Fingerprint ideal height for useful enhancement.
FINGERPRINT_HEIGHT = 352

# Fingerprint block size, in pixels, for patch-wise image processing.
FINGERPRINT_BLOCK = 16

# Threshold for deciding/computing fingerprint content mask.
FINGERPRINT_MASK_TRSH = 0.25

# Possible ridge orientations: 2 x 16 + 1 = 33 equally distributed angles in the interval [-pi, pi].
RIDGE_ORIENTATION_STEP = numpy.pi / 16
RIDGE_ORIENTATIONS = numpy.arange(-numpy.pi, numpy.pi + RIDGE_ORIENTATION_STEP, RIDGE_ORIENTATION_STEP)

# Wavelength contribution to Gabor filter standard deviation.
WAVELENGTH_RATIO = 0.25

# Gabor filter output threshold to consider a pixel as non-activated.
GABOR_OUTPUT_BIN_TRSH = -0.2


#####################################
# Fingerprints enhancement methods. #
#####################################

# Rotates a given <image> CCW obeying the given <rad_angle>.
# Image content is them cropped to avoid empty borders, bounded by the largest possible inscribed square.
# Return the obtained rotated and cropped image.
def __rotate_and_crop(image, rad_angle):
    h, w = image.shape

    degree_angle = 360.0 - (180.0 * rad_angle / numpy.pi)
    rotated = scipy.ndimage.rotate(image, degree_angle, reshape=False)

    crop_size = int(h / numpy.sqrt(2))
    crop_start = int((h - crop_size) / 2.0)

    rotated = rotated[crop_start: crop_start + crop_size, crop_start: crop_start + crop_size]
    return rotated


# Preprocesses a given <fingerprint> image, before "actual" enhancement.
# The fingerprint goes through:
# 1. Transformation from RGB to grayscale color scheme;
# 2. Resize to a height of <output_height> pixels, keeping original aspect ratio.
# 3. Color histogram normalization.
# Parameters
# <fingerprint>: the fingerprint image to be preprocessed.
# <output_height>: the expect fingerprint image height, after preprocessing.
# <dark_ridges>: TRUE if ridges where captured dark, FALSE if white.
# <view>: TRUE if the preprocessed fingerprint must be shown, FALSE otherwise.
# Returns the obtained preprocess fingerprint, ideally with dark ridges.
def _01_preprocess(fingerprint, output_height, dark_ridges=True, view=False):
    # makes the fingerprint grayscale, if it is still colored
    if len(fingerprint.shape) > 2 and fingerprint.shape[2] > 1:  # more than one channel?
        fingerprint = cv2.cvtColor(fingerprint, cv2.COLOR_BGR2GRAY)

    # resizes the fingerprint to present a height of <output_height> pixels, keeping original aspect ratio
    aspect_ratio = float(fingerprint.shape[0]) / fingerprint.shape[1]
    width = int(round(output_height / aspect_ratio))
    fingerprint = cv2.resize(fingerprint, (width, output_height))

    # makes the fingerprint ridges dark, if it is the case
    if not dark_ridges:
        fingerprint = abs(255 - fingerprint)

    # equalizes the fingerprint grayscale color histogram
    fingerprint = cv2.equalizeHist(fingerprint, fingerprint)

    # shows the obtained fingerprint, if it is the case
    if view:
        cv2.imshow('Preprocessing, press any key.', fingerprint)
        cv2.waitKey(0)

#    print('[INFO] Preprocessed fingerprint.')
    return fingerprint


# Segments a given <fingerprint> image (i.e., generates an image mask whose background pixels are black
# and foreground pixels are white.
# Parameters
# <fingerprint>: the fingerprint image to be segmented.
# <block_size>: pixel neigborhood size used to assess if it contains background or foreground information.
# <std_threshold>: the pixel value standard deviation threshold to consider a pixel block-wise foreground.
# <view>: TRUE if the segmented fingerprint must be shown, FALSE otherwise.
# Returns the masked fingerprint with pixel values normalized (0 mean, unit standard deviation) and
# the computed fingerprint content mask.
def _02_segment(fingerprint, block_size, std_threshold, view=False):
    # fingerprint dimensions
    h, w = fingerprint.shape

    # normalizes the fingerprint pixel values (0 mean, unit standard deviation)
    fingerprint = (fingerprint - numpy.mean(fingerprint)) / numpy.std(fingerprint)

    # prepares the fingerprint content mask
    mask = numpy.zeros((h, w), numpy.uint8)  # completely black mask (no useful pixels)

    # for each pixel neighborhood
    block_step = int(block_size / 2.0)
    for row in range(h):
        for col in range(w):
            # obtains the current block
            block = fingerprint[max(0, row - block_step):min(row + block_step + 1, h),
                    max(0, col - block_step):min(col + block_step + 1, w)]

            # updates the image mask according to the current block standard deviation
            # (if the current pixel stands out, sets it up as being useful content)
            if numpy.std(block) > std_threshold:
                mask[row, col] = 255  # pixel is white

    # re-normalizes the fingerprint, now applying the computed mask
    masked_values = fingerprint[mask > 0]
    fingerprint = (fingerprint - numpy.mean(masked_values)) / numpy.std(masked_values)
    fingerprint = cv2.bitwise_and(fingerprint, fingerprint, mask=mask)

    # shows the obtained result, if it is the case
    if view:
        img = fingerprint.copy()
        img = cv2.normalize(fingerprint, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Segmentation, press any key.', img)
        cv2.waitKey(0)

    # returns the normalized fingerprint and obtained mask
#    print('[INFO] Segmented fingerprint.')
    return fingerprint, mask


# TODO
# Description will be added soon.
def _03_compute_orientations(fingerprint, mask, block_size, view=False):
    # fingerprint dimensions
    h, w = fingerprint.shape

    # fingerprint gradients in x- and y-directions
    y_gradient, x_gradient = numpy.gradient(fingerprint)

    # fingerprint pixel orientations
    orientations = numpy.arctan2(y_gradient, x_gradient)
    orientations = cv2.bitwise_and(orientations, orientations, mask=mask)

    # magnitudes of pixel-wise orientations
    magnitudes = numpy.sqrt(y_gradient ** 2 + x_gradient ** 2)
    magnitudes = cv2.bitwise_and(magnitudes, magnitudes, mask=mask)

    # makes the computed orientation discrete
    discret_orientations = numpy.zeros(orientations.shape, dtype=numpy.float32)
    block_step = int(block_size / 2.0)
    for row in range(h):
        for col in range(w):
            if mask[row, col] > 0:
                # obtains the current block of orientations and magnitudes
                ori_block = orientations[max(0, row - block_step):min(row + block_step + 1, h),
                            max(0, col - block_step):min(col + block_step + 1, w)]
                mag_block = magnitudes[max(0, row - block_step):min(row + block_step + 1, h),
                            max(0, col - block_step):min(col + block_step + 1, w)]

                # orientation histogram of the current block
                useful_magnitudes = numpy.where(mag_block > numpy.mean(mag_block))
                freqs, values = numpy.histogram(ori_block[useful_magnitudes], bins=RIDGE_ORIENTATIONS)

                # computes the current discrete orientation
                best_value = numpy.mean(values[numpy.where(freqs == numpy.max(freqs))])
                orientation_index = int(round(best_value / RIDGE_ORIENTATION_STEP))
                discret_orientations[row, col] = RIDGE_ORIENTATIONS[orientation_index]

    discret_orientations = cv2.bitwise_and(discret_orientations, discret_orientations, mask=mask)

    # shows the fingerprint orientations, if it is the case
    if view:
        # 1st, shows the gradients
        img = x_gradient.copy()
        img = cv2.normalize(x_gradient, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Orientation, x gradient, press any key.', img)
        cv2.waitKey(0)

        img = y_gradient.copy()
        img = cv2.normalize(y_gradient, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('Orientation, y gradient, press any key.', img)
        cv2.waitKey(0)

        # 2nd, shows the orientations
        plot_step = 8  # orientation shown at every 4 pixels

        # colored version of the fingerprint (to show orientations in green)
        img = fingerprint.copy()
        img = cv2.normalize(fingerprint, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # draws the orientations
        mag_enhance = 5.0  # magnitude is drawn 5 times greater
        start_pixel = int(plot_step / 2.0)
        for row in range(start_pixel, h, plot_step):
            for col in range(start_pixel, w, plot_step):
                angle = discret_orientations[row, col]
                magnitude = magnitudes[row, col] * mag_enhance

                if magnitude > 0:
                    delta_x = int(round(math.cos(angle) * magnitude))
                    delta_y = int(round(math.sin(angle) * magnitude))

                    cv2.line(img, (col, row), (col + delta_x, row + delta_y), (0, 255, 0), 1)

        # shows the resulting image
        cv2.imshow('Orientation, press any key.', img)
        cv2.waitKey(0)

    # return orientations and magnitudes
#    print('[INFO] Computed ridge orientations.')
    return discret_orientations, magnitudes


# TODO
# Description will be added soon.
def _04_compute_ridge_frequency(fingerprint, mask, orientations, block_size, view=False):
    # computed ridge frequencies
    frequencies = []

    # fingerprint dimensions
    h, w = fingerprint.shape

    # for each pixel neighborhood
    block_step = int(block_size / 2.0)
    for row in range(h):
        for col in range(w):
            if mask[row, col] > 0:
                # obtains the current block
                block = fingerprint[max(0, row - block_step):min(row + block_step + 1, h),
                        max(0, col - block_step):min(col + block_step + 1, w)]

                # rotates the block, making ridges vertical
                rot_block = __rotate_and_crop(block, -orientations[row, col])

                # shows the rotated block, if it is the case
                if view:
                    img = rot_block.copy()
                    img = cv2.normalize(rot_block, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    cv2.imshow('block', img)
                    cv2.waitKey(120)

                # projects ridges on x-axis, to obtain ridge peaks
                ridge_proj = numpy.sum(rot_block, axis=0)
                ridge_peaks = numpy.zeros(ridge_proj.shape)
                ridge_peaks[numpy.where(ridge_proj > numpy.mean(ridge_proj))] = 1

                # counts the number of ridges and computes the frequency
                ridge_count = 0

                is_ridge = False
                for i in range(len(ridge_peaks)):
                    if ridge_peaks[i] == 1 and not is_ridge:
                        ridge_count = ridge_count + 1
                        is_ridge = True

                    elif ridge_peaks[i] == 0 and is_ridge:
                        ridge_count = ridge_count + 1
                        is_ridge = False

                frequencies.append(0.5 * ridge_count / len(ridge_peaks))

    # returns the average frequency
#    print('[INFO] Computed ridge frequency.')
    if len(frequencies) > 0:
        return numpy.mean(frequencies)
    else:
        return 0


# TODO
# Description will be added soon.
def _05_apply_gabor_filter(fingerprint, mask, orientations, ridge_frequency, std_wavelength_ratio, view=False):
    output = numpy.zeros(fingerprint.shape)

    # fingerprint dimensions
    h, w = fingerprint.shape

    # obtains the needed convolutions of the given fingerprint
    fingerprint_filters = {}

    filter_std = std_wavelength_ratio * 1.0 / ridge_frequency
    for orientation in numpy.unique(orientations):
        kernel = numpy.real(gabor_kernel(ridge_frequency, orientation, sigma_x=filter_std, sigma_y=filter_std))
        fingerprint_filters[orientation] = scipy.ndimage.convolve(fingerprint, kernel)

    # for each pixel, sets its value as the proper filter response, depending on the pixel orientation
    for row in range(h):
        for col in range(w):
            if mask[row, col] > 0:
                key_orientation = orientations[row, col]
                output[row, col] = fingerprint_filters[key_orientation][row, col]

    # binarizes the output
    output = (output < GABOR_OUTPUT_BIN_TRSH).astype(numpy.uint8) * 255
    output = cv2.erode(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    # shows the obtained result, if it is the case
    if view:
        cv2.imshow('Filtering, press any key.', output)
        cv2.waitKey(0)

 #   print('[INFO] Applied Gabor filters.')
    return output


# TODO
def _06_skeletonize(fingerprint, view=False):
    fingerprint = skimage.morphology.skeletonize(fingerprint / 255).astype(numpy.uint8) * 255

    # shows the obtained result, if it is the case
    if view:
        cv2.imshow('Skeletonization, press any key.', fingerprint)
        cv2.waitKey(0)

#    print('[INFO] Skeletonized ridges.')
    return fingerprint


################
# Main method. #
################

# Enhances the content of a given fingerprint image, aiming at further minuatiae detection.
# Parameters
# <fingerprint>: the fingerprint image to be enhanced.
# <dark_ridges>: TRUE if ridges where captured dark, FALSE if white.
# <view>: TRUE if the enhanced fingerprint must be shown, FALSE otherwise.
# Returns the pre-processed and the enhanced fingerprint image, as well as a content mask
# with background pixels black and foreground pixels white.
def enhance(fingerprint, dark_ridges=True, view=False):
    # pre-processes the fingerprint
    pp_fingerprint = _01_preprocess(fingerprint, FINGERPRINT_HEIGHT, dark_ridges, view=view)

    # masks the fingerprint content
    en_fingerprint, mask = _02_segment(pp_fingerprint, FINGERPRINT_BLOCK, FINGERPRINT_MASK_TRSH, view=view)

    # computes the fingerprint orientations
    orientations, magnitudes = _03_compute_orientations(en_fingerprint, mask, FINGERPRINT_BLOCK, view=view)

    # computes the frequency of ridges
    ridge_freq = _04_compute_ridge_frequency(en_fingerprint, mask, orientations, FINGERPRINT_BLOCK)  # ,view=view)

    # applies Gabor filters on the fingerprint
    en_fingerprint = _05_apply_gabor_filter(en_fingerprint, mask, orientations, ridge_freq, WAVELENGTH_RATIO, view=view)

    # skeletonizes the fingerprint
    en_fingerprint = _06_skeletonize(en_fingerprint, view=view)

#    print('[INFO] Enhanced fingerprint.')
    return pp_fingerprint, en_fingerprint, mask
