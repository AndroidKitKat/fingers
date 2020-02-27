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

print(finger_file)
root_file = 'test-data/fingerprint_dc/'
fingerprints = []

cnt = -1

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
for k, v in d1.items():
    id_dict[v[0]] = int(k)
fd.close()

pprint(id_dict)
print('loaded dataset')
print(id_dict[id_print])

a_acquire.verify_fingerprint(id_dict[id_print], finger_file)

# super_secret_dict = {}
# reverse_dict = {}

# for dir_name, subdir_list, file_list in walk(root_file):
#     i = 0
#     for fname in file_list:
#         print(fname)
#         if fname[-4:] == '.bmp':
#             i += 1
#             fingerprints.append(a_acquire.acquire_from_file(f'{dir_name}/{fname}', view=False))
#             super_secret_dict[cnt] = [fname[:-6], i]
#             reverse_dict[fname[:-6]] = cnt
#     cnt += 1
#     # if cnt > 3:
#     #     break

# print('done with loading')
# pprint(super_secret_dict)
# pprint(reverse_dict)
# pp_fingerprints = [b_enhance.enhance(fprint, dark_ridges=False, view=False) for fprint in fingerprints]

# curr_id = 0
# print('done with enhancement')
# p_cnt = float('inf')
# for pp_print in pp_fingerprints:
#     try: 
#         if p_cnt >= super_secret_dict[curr_id][1]:
#             a_acquire.FINGER_DICT[curr_id] = []
#             p_cnt = 0
#             curr_id += 1
#         ridges, bifs = c_describe.describe(pp_print[1], pp_print[2], view=False)
#         a_acquire.FINGER_DICT[curr_id-1].append((pp_print[1].tolist(), ridges.tolist(), bifs.tolist()))
#         p_cnt += 1
#     except:
#         continue

# fd = open('dataset.json', 'w')
# fd.write(json.dumps(a_acquire.FINGER_DICT))
# fd.close()

# fd2 = open('ids.json', 'w')
# fd2.write(json.dumps(super_secret_dict))
# fd2.close()

# pprint(a_acquire.FINGER_DICT)
# pprint(fingerprints)
# pprint(pp_fingerprints)
# pprint(minutia)

# matches = d_match.match(new_dict[1][0][0], new_dict[1][0][1], new_dict[1][0][2], new_dict[1][1][0], new_dict[1][1][1], new_dict[1][1][2], view=False)

# print(len(matches[0]))
# print(len(matches[1]))
# pprint(matches)

# matches = d_match.match(a_acquire.FINGER_DICT[3][0][0], a_acquire.FINGER_DICT[3][0][1], a_acquire.FINGER_DICT[3][0][2], a_acquire.FINGER_DICT[3][1][0], a_acquire.FINGER_DICT[3][1][1], a_acquire.FINGER_DICT[3][1][2], view=True)
# matches = d_match.match(en_fingerprint_1, ridge_endings_1, bifurcations_1,
# #                         en_fingerprint_2, ridge_endings_2, bifurcations_2, view=True)
# pprint(matches)