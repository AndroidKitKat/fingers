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
from d_match import match
from b_enhance import enhance
from c_describe import describe
from collections import OrderedDict
import multiprocessing.pool
import itertools
from statistics import mean

FINGER_DICT = {}

# 3. Verification scenario (loading of fingerprint + ID, to verify claimed identity); or
def verify_fingerprint(id, filename, variable):
    test_print = acquire_from_file(filename, view=False)
    pp_test = enhance(test_print, dark_ridges=False, view=False)
    ridges, bifs = describe(pp_test[1], pp_test[2], variable, view=False)
    # use 3 cores for the three sets of fingerprints
    p = multiprocessing.pool.ThreadPool(3)
    matches = p.starmap(match, [(pp_test[1], ridges, bifs, FINGER_DICT[id][0][0], FINGER_DICT[id][0][1], FINGER_DICT[id][0][2]), (pp_test[1], ridges, bifs, FINGER_DICT[id][1][0], FINGER_DICT[id][1][1], FINGER_DICT[id][1][2]), (pp_test[1], ridges, bifs, FINGER_DICT[id][2][0], FINGER_DICT[id][2][1], FINGER_DICT[id][2][2])])
    print(len(matches[0][0] + matches[0][1]))
    print(len(matches[1][0] + matches[1][1]))
    print(len(matches[2][0] + matches[2][1]))

    m_1 = len(matches[0][0] + matches[0][1])
    m_2 = len(matches[1][0] + matches[1][1])
    m_3 = len(matches[2][0] + matches[2][1])

    if mean([m_1, m_2, m_3]) > 20 and max(m_1, m_2, m_3) > 30:
#        print("\n\n*hacker voice* I'm in\n\n")
        return True
    else:
#        print("\n\nINTRUDER DETECTED\n\n")
        return False
    
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

#    print('[INFO] Acquired fingerprint from file.')
    return fingerprint

    # show the read fingerprint if it is valid
