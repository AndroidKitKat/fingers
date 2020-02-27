# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2020
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Fingerprint Recognition
# TODO: Test script with alternative fingerprint enhancement. Better description will be added soon.
# Language: Python 3

import a_acquire
import b_enhance_2
import c_describe
import d_match

# Test script.
# TODO: Better description will be added soon.

fingerprint_file_path_1 = 'test-data/1.bmp'
fingerprint_file_path_2 = 'test-data/2.png'

fingerprint_1 = a_acquire.acquire_from_file(fingerprint_file_path_1, view=False)
fingerprint_2 = a_acquire.acquire_from_file(fingerprint_file_path_2, view=False)

pp_fingerprint_1, en_fingerprint_1, mask_1 = b_enhance_2.enhance(fingerprint_1, dark_ridges=False, view=False)
pp_fingerprint_2, en_fingerprint_2, mask_2 = b_enhance_2.enhance(fingerprint_2, dark_ridges=False, view=False)

ridge_endings_1, bifurcations_1 = c_describe.describe(en_fingerprint_1, mask_1, view=False)
ridge_endings_2, bifurcations_2 = c_describe.describe(en_fingerprint_2, mask_2, view=False)

matches = d_match.match(en_fingerprint_1, ridge_endings_1, bifurcations_1,
                        en_fingerprint_2, ridge_endings_2, bifurcations_2, view=True)
