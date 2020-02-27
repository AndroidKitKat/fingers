# University of Notre Dame
# Course CSE 40537 / 60537 - Biometrics - Spring 2020
# Instructor: Daniel Moreira (dhenriq1@nd.edu)
# Fingerprint Recognition
# TODO: Test script. Better description will be added soon.
# Language: Python 3

import a_acquire
import b_enhance
import c_describe
import d_match
from pprint import pprint
from os import walk

# Test script.
# TODO: Better description will be added soon.

root_file = 'test-data/fingerprint_dc/'
fingerprints = []

cnt = 0

super_secret_dict = {}

for dir_name, subdir_list, file_list in walk(root_file):
    for fname in file_list:
        print(fname)
        if fname[-4:] == '.bmp':
            fingerprints.append(a_acquire.acquire_from_file(f'{dir_name}/{fname}', view=False))
            super_secret_dict[cnt-1] = fname[:-4]
    cnt += 1
    if cnt > 2:
        break

print('done with loading')
pprint(super_secret_dict)
pp_fingerprints = [b_enhance.enhance(fprint, dark_ridges=False, view=False) for fprint in fingerprints]

p_cnt = 0
curr_id = 0
print('done with enhancement')
for pp_print in pp_fingerprints:
    a_acquire.FINGER_DICT[curr_id] = []
    ridges, bifs = c_describe.describe(pp_print[1], pp_print[2], view=False)
    a_acquire.FINGER_DICT[curr_id].append((pp_print[1], ridges, bifs))
    p_cnt += 1
    if not p_cnt % 3:
        curr_id += 1


pprint(a_acquire.FINGER_DICT)
# pprint(fingerprints)
# pprint(pp_fingerprints)
# pprint(minutia)
matches = d_match.match(a_acquire.FINGER_DICT[0][0][0], a_acquire.FINGER_DICT[0][0][1], a_acquire.FINGER_DICT[0][0][2], a_acquire.FINGER_DICT[0][0][0], a_acquire.FINGER_DICT[0][0][1], a_acquire.FINGER_DICT[0][0][2], view=True)
# matches = d_match.match(en_fingerprint_1, ridge_endings_1, bifurcations_1,
# #                         en_fingerprint_2, ridge_endings_2, bifurcations_2, view=True)
pprint(matches)