# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2020
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Fingerprint Recognition
# 01. Acquisition-module stub.
# Language: Python 3
# Needed libraries: OpenCV (https://opencv.org/)
# Quick install (with PyPI - https://pypi.org/):
# execute "pip3 install opencv-contrib-python==3.4.2.17" on command shell.

import cv2
from collections import OrderedDict

FINGER_DICT = {}

# 3. Verification scenario (loading of fingerprint + ID, to verify claimed identity); or
def verify_fingerprint(fingerprint, id):

    pass

# Stub function to acquire a fingerprint sample from a file, given its path.
# Parameters
# file_path: The path to image file containing one fingerprint sample.
# view: TRUE if loaded fingerprint must be shown in a proper window, FALSE otherwise.
def acquire_from_file(file_path, view=False):
    # reads the fingerprint image from the given file path
    # and returns it
    fingerprint = cv2.imread(file_path)

    # call function here 
    if view:
        cv2.imshow('press any key', fingerprint)
        cv2.waitKey(0)

    print('[INFO] Acquired fingerprint from file.')
    return fingerprint

    # show the read fingerprint if it is valid

# TODO
# New functions may be added here, targeting either:
# 1. Solution test execution (loading of various genuine and impostor fingerprint pairs); or
# 2. Enrollment scenario (loading of fingerprint + ID to enroll in a system); or
    
# 4. Recognition scenario (loading of fingerprint, to find identity).
