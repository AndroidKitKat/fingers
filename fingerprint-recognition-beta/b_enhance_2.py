# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2020
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Fingerprint Recognition
# 02. Alternative module to enhance fingerprint samples, aiming at further minutiae detection.
# This module uses the solution available at https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python
# under BSD 2-Clause "Simplified" License.
# Language: Python 3
# Needed libraries: NumPy (https://numpy.org/), OpenCV (https://opencv.org/),
# SciPy (https://www.scipy.org/) and Scikit-Image (https://scikit-image.org/docs/dev/api/skimage.html).
# Quick install (with PyPI - https://pypi.org/): execute, on command shell (each line at a time):
# "pip3 install numpy";
# "pip3 install opencv-contrib-python==3.4.2.17";
# "pip3 install scikit-image".

import sys
import os
import cv2
import numpy
import skimage.morphology

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/third-party/Fingerprint-Enhancement-Python-master/src')
from image_enhance import image_enhance

# TODO
# Description will be added soon.
FINGERPRINT_HEIGHT = 352


# TODO
# Description will be added soon.
def _preprocess(fingerprint, output_height, dark_ridges=True, view=False):
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

    print('[INFO] Preprocessed fingerprint.')
    return fingerprint


# TODO
# Description will be added soon.
def _skeletonize(fingerprint, view=False):
    fingerprint = skimage.morphology.skeletonize(fingerprint / 255).astype(numpy.uint8) * 255

    # shows the obtained result, if it is the case
    if view:
        cv2.imshow('Skeletonization, press any key.', fingerprint)
        cv2.waitKey(0)

    print('[INFO] Skeletonized ridges.')
    return fingerprint


# TODO
# Description will be added soon.
def enhance(fingerprint, dark_ridges=True, view=False):
    # pre-processes the fingerprint
    pp_fingerprint = _preprocess(fingerprint, FINGERPRINT_HEIGHT, dark_ridges, view=view)

    # enhances the fingerprint with the thidr-party solution
    en_fingerprint, mask = image_enhance(pp_fingerprint)
    en_fingerprint = en_fingerprint.astype(numpy.uint8) * 255
    mask = mask.astype(numpy.uint8) * 255

    # skeletonizes the fingerprint
    en_fingerprint = _skeletonize(en_fingerprint, view=view)

    print('[INFO] Enhanced fingerprint.')
    return pp_fingerprint, en_fingerprint, mask
