#!/usr/bin/env python3

import functools
import hashlib
import itertools
import multiprocessing
import os
import string
import sys

# Constants

ALPHABET    = string.ascii_lowercase + string.digits
ARGUMENTS   = sys.argv[1:]
CORES       = 1
HASHES      = 'hashes.txt'
LENGTH      = 1
PREFIX      = ''
hash_set = set()
passwords = []

# Functions

def usage(exit_code=0):
    print('''Usage: {} [-a alphabet -c CORES -l LENGTH -p PATH -s HASHES]
    -a ALPHABET Alphabet to use in permutations
    -c CORES    CPU Cores to use
    -l LENGTH   Length of permutations
    -p PREFIX   Prefix for all permutations
    -s HASHES   Path of hashes file'''.format(os.path.basename(sys.argv[0])))
    sys.exit(exit_code)

def sha1sum(s):
    ''' Generate sha1 digest for given string.

    >>> sha1sum('abc')
    'a9993e364706816aba3e25717850c26c9cd0d89d'

    >>> sha1sum('wake me up inside')
    '5bfb1100e6ef294554c1e99ff35ad11db6d7b67b'

    >>> sha1sum('baby now we got bad blood')
    '9c6d9c069682759c941a6206f08fb013c55a0a6e'
    '''
    # TODO: Implement
    #ez
    return hashlib.md5(s.encode()).hexdigest()

def permutations(length, alphabet=ALPHABET):
    ''' Recursively yield all permutations of alphabet up to provided length.

    >>> list(permutations(1, 'ab'))
    ['a', 'b']

    >>> list(permutations(2, 'ab'))
    ['aa', 'ab', 'ba', 'bb']

    >>> list(permutations(1))       # doctest: +ELLIPSIS
    ['a', 'b', ..., '9']

    >>> list(permutations(2))       # doctest: +ELLIPSIS
    ['aa', 'ab', ..., '99']

    >>> import inspect; inspect.isgeneratorfunction(permutations)
    True
    '''
    # TODO: Implement as generator

    #base case
    if length == 1:
        yield from alphabet

   #recursive
    else:
        for letter in alphabet:
            for subperm in permutations(length - 1, alphabet):
                yield letter + subperm

def smash(hashes, length, alphabet=ALPHABET, prefix=''):
    ''' Return all password permutations of specified length that are in hashes

    >>> smash([sha1sum('ab')], 2)
    ['ab']

    >>> smash([sha1sum('abc')], 2, prefix='a')
    ['abc']

    >>> smash(map(sha1sum, 'abc'), 1, 'abc')
    ['a', 'b', 'c']
    '''
    # TODO: Implement with list or generator comprehensions
    perms = permutations(length, alphabet)
    return [prefix + perm for perm in perms if sha1sum(prefix + perm) in hashes]

# Main Execution

if __name__ == '__main__':
    # Parse command line arguments
    while len(ARGUMENTS) and ARGUMENTS[0].startswith('-') and len(ARGUMENTS[0]) > 1:
        arg = ARGUMENTS.pop(0)
        if arg == '-a':
            ALPHABET = ARGUMENTS[0]
            ARGUMENTS.pop(0)
        elif arg == '-c':
            CORES = ARGUMENTS[0]
            ARGUMENTS.pop(0)
        elif arg == '-l':
            LENGTH = ARGUMENTS[0]
            ARGUMENTS.pop(0)
        elif arg == '-p':
            PREFIX = ARGUMENTS[0]
            ARGUMENTS.pop(0)
        elif arg == '-s':
            HASHES = ARGUMENTS[0]
            ARGUMENTS.pop(0)
        elif arg == '-h':
            usage(0)
        else:
            usage(1)

    # Load hashes set
    for line in open(HASHES):
        hash_set.add(line.strip())

    # Execute smash function
    if int(CORES) > 1 and int(LENGTH) > 1:
        subsmash = functools.partial(smash, hash_set, int(LENGTH) - 1, ALPHABET)
        pool = multiprocessing.Pool(processes = int(CORES))
        prefixes = []
        #loop to generate the prefixes thing
        for letter in ALPHABET:
            prefix = PREFIX + letter
            prefixes.append(prefix)
        #this is magic
        passwords = itertools.chain.from_iterable(pool.imap(subsmash, prefixes))
    else:
        passwords = smash(hash_set, int(LENGTH), ALPHABET, PREFIX)

    # Print passwords
    for password in passwords:
        print(password)
