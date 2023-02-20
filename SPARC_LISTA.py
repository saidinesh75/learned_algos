import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!! (info and warnings are not printed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow.compat.v1 as tf
rng = np.random.RandomState(seed=None)
tf.set_random_seed(1)

import math
import sys
import numpy.linalg as la

import matplotlib.pyplot as plt
import time
from generate_msg_mod_modified import generate_msg_mod_modified


"""
M = Number of columns in a section 
L = Number of section
R = Rate
pnz = sparsity
awgn_var = AWGN channel variance
"""

EbN0_dB = np.array([5])

code_params   = {'P': 15.0,     # Average codeword symbol power constraint
                 'R': 0.5,      # Rate
                 'L': 100,      # Number of sections
                 'M': 32,       # Columns per section
                 'K': 1,
                 'dist':0,
                 }  

P,R,L,M,K,dist = map(code_params.get,['P','R','L','M','K','dist'])


delim = np.zeros([2,L])
delim[0,0] = 0
delim[1,0] = M-1

for i in range(1,L):
    delim[0,i] = delim[1,i-1]+1
    delim[1,i] = delim[1,i-1]+M

for e in range(np.size[EbN0_dB]):
    code_params.update({'EbNo_dB':EbN0_dB[e]})
    Eb_No_linear = np.power(10, np.divide(EbN0_dB[e],10))

    bit_len = int(round(L*np.log2(K*M)))
    logM = int(round(np.log2(M)))
    logK = int(round(np.log2(K)))
    sec_size = logM + logK

