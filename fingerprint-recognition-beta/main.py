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
import json
import sys

# Test script.
# TODO: Better description will be added soon.

def usage():
    print(f'''{sys.argv[0]}: [name_finger] [finger_file]''')
    exit(1)


id_print = None

if len(sys.argv) == 3:
    id_print = sys.argv[1]
    finger_file = sys.argv[2]
else:
    usage()

root_file = 'test-data/fingerprint_dc/'
fingerprints = []

data_file = 'dataset.json'
key_file = 'ids.json'

f = open(data_file, 'r')
d = json.loads(f.read())
new_dict = {}
for k, v in d.items():
    a_acquire.FINGER_DICT[int(k)] = v
f.close()

fd = open(key_file, 'r')
d1 = json.loads(fd.read())
id_dict = {}
lookup_dict = {}
for k, v in d1.items():
    id_dict[v[0]] = int(k)
    lookup_dict[int(k)] = v[0]
fd.close()

print(id_print, finger_file)

# pprint(id_dict)
# print('loaded dataset')
# print(id_dict[id_print])

a_acquire.verify_fingerprint(id_dict[id_print], finger_file)
