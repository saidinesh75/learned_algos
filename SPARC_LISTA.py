import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!! (info and warnings are not printed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# tf.disable_v2_behavior() #compatibility
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

def is_power_of_2(x):
    return (x > 0) and ((x & (x - 1)) == 0)  # '&' id bitwise AND operation.

def gray2bin(num):
    '''
    Converts gray code (int type) to binary code (int type)
    From https://en.wikipedia.org/wiki/Gray_code
    '''
    mask = num >> 1
    while (mask != 0):
        num  = num ^ mask
        mask = mask >> 1
    return num

def bin_arr_2_int(bin_array):
    '''
    Binary array (numpy.ndarray) to integer
    '''
    # assert bin_array.dtype == 'bool'
    k = bin_array.size
    assert 0 < k < 64 # Ensures non-negative integer output
    # 1 << np.arange(k)[::-1]      generates vector like [16 8 4 2 1]
    return bin_array.dot(1 << np.arange(k)[::-1])

def psk_constel(K):
    '''
    K-PSK constellation symbols
    '''
    assert type(K)==int and K>1 and is_power_of_2(K)

    if K == 2:
        c = np.array([1, -1])
    elif K == 4:
        c = np.array([1+0j, 0+1j, -1+0j, 0-1j])
    else:
        theta = 2*np.pi*np.arange(K)/K
        c     = np.cos(theta) + 1J*np.sin(theta)

    return c

def psk_mod(bin_arr, K):
    '''
    K-PSK modulation (using gray coding).

    bin_arr: boolean numpy.ndarray to modulate. Length of  L * log2(K).
    K      : number of PSK contellations, K>1 and is a power of 2

    Returns
    symbols: Corresponding K-PSK modulation symbols of length L.
             (If K=2 then symbols are real, complex otherwise.)
    '''

    assert type(K)==int and K>1 and is_power_of_2(K)
    # assert bin_arr.dtype == 'bool'

    c    = psk_constel(K)           # Constellation symbols
    logK = int(round(np.log2(K)))
    assert bin_arr.size % logK == 0
    L    = bin_arr.size // logK     # Number of symbols
    if L == 1:
        k = bin_arr.size
        idx     = gray2bin(bin_arr_2_int(bin_arr)) # gray code index
        symbols = c[idx]
    else:
        symbols = np.zeros(L, dtype=c.dtype)
        for l in range(L):
            idx        = gray2bin(bin_arr_2_int(bin_arr[l*logK:(l+1)*logK]))
            symbols[l] = c[idx]

    return symbols

def gen_msg_sparc_lista(cols, code_params,delim,rng):
    P,R,L,M,K,dist = map(code_params.get,['P','R_actual','L','M','K','dist'])
    N = int(L*M)
    bit_len = int(round(L*np.log2(K*M)))
    logM = int(round(np.log2(M)))
    logK = int(round(np.log2(K)))
    sec_size = logM + logK
    
    beta = np.zeros((N,cols),dtype=complex) if K>2 else np.zeros((N,cols))
    beta_val = np.zeros((N,cols),dtype=complex) if K>2 else np.zeros((N,cols))

    for i in range(cols):
        bits_in = rng.randint(2, size=bit_len)
        bits_in_val = rng.randint(2, size=bit_len)

        if K==1 or K==2:
            beta0 = np.zeros(N)    #length of msg_vector = 1000 * 32 = 32000
            beta0_val = np.zeros(N)
        else:
            beta0 = np.zeros(N, dtype=complex)
            beta0_val = np.zeros(N, dtype=complex)

        for l in range(L):
            bits_sec = bits_in[l*sec_size : l*sec_size + logM]  # logM bits used for selection the location of non-zero values
            bits_sec_val = bits_in_val[l*sec_size : l*sec_size + logM]
            assert 0<logM<64

            idx = bits_sec.dot(1 << np.arange(logM)[::-1])
            idx_val = bits_sec_val.dot(1 << np.arange(logM)[::-1])

            if K==1:
                val = 1
            else:
                bits_mod_sec = bits_in[l*sec_size+logM : (l+1)*sec_size] #logK bits used for selection of PSK symbol for the non-zero location
                bits_mod_sec_val = bits_in_val[l*sec_size+logM : (l+1)*sec_size]
                values = psk_mod(bits_mod_sec, K)
                values_val = psk_mod(bits_mod_sec, K)

            beta0[l*int(M) + idx] = values      # will make a 1 at the decimal equivalent in the l-th section
            beta0_val[l*int(M) + idx_val] = values_val

        beta[:,i] = beta0
        beta_val[:,i] = beta0_val

    c = psk_constel(K)    
    return beta,beta_val,c,code_params
    return 

def simple_soft_threshold(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other


EbN0_dB = np.array([5])
cols = 100

# Defining Code_params
code_params   = {'P': 15.0,     # Average codeword symbol power constraint
                 'R': 0.5,      # Rate
                 'L': 100,      # Number of sections
                 'M': 32,       # Columns per section
                 'K': 1,
                 'dist':0,
                 }  
P,R,L,M,K,dist = map(code_params.get,['P','R','L','M','K','dist'])

#...........Columns per section...........................
delim = np.zeros([2,L])
delim[0,0] = 0
delim[1,0] = M-1
for i in range(1,L):
        delim[0,i] = delim[1,i-1]+1
        delim[1,i] = delim[1,i-1]+M

#...........Calculating the length of the codeword..........
bit_len = int(round(L*np.log2(K*M)))
n = int(round(bit_len/R))
R_actual = bit_len / n

#...........Adding the parameters to code_params............
code_params.update({'R_actual':R_actual})
code_params.update({'n':n})

#...........Generating Measurement Matrix...................
A = np.random.normal(size=(n, L*M), scale= np.sqrt(P/L)).astype(np.float32)  #so power of each col = (nP)/L => E[||Ab||^2] = nP
A_ = tf.constant(A,name='A',dtype = tf.float32 )  #Made a tf constant since it need not be trained

#...........Loop for Eb/N0 starts...........................
for e in range(np.size[EbN0_dB]):
    code_params.update({'EbNo_dB':EbN0_dB[e]})
    Eb_No_linear = np.power(10, np.divide(EbN0_dB[e],10))
    beta, beta_val, c,code_params = gen_msg_sparc_lista(cols,code_params,delim,rng)

    #......awgn_var calculation.............................
    Eb = n*P/bit_len
    awgn_var = Eb/Eb_No_linear
    sigma = np.sqrt(awgn_var)
    code_params.update({'awgn_var':awgn_var})
    snr_rx = P/awgn_var
    capacity = 0.5 * np.log2(1 + snr_rx)

    #......Generating the codeword using the measurement matrix
    xgen_ = tf.constant(beta, dtype= tf.float64)  #changed to tf.constant
    ygen_ = tf.matmul(A_,xgen_) + tf.random.normal( (n,cols),stddev=math.sqrt( awgn_var ) )
    y_val = np.matmul(A, beta_val) + np.random.normal( size = (n,cols),scale= math.sqrt(awgn_var)) #tf.sqrt(awgn_var)*rng.randn(n)


    #####...............BUILD LISTA................#####
    T, initial_lambda = 6, 0.1
    # creating placeholders, which will hold dataset later.
    x_ = tf.compat.v1.placeholder( tf.float32,(L*M,None),name='x' )
    y_ = tf.compat.v1.placeholder( tf.float32,(n,None),name='y' )

    eta = simple_soft_threshold #creating an object

    B = A.T / (1.01 * la.norm(A,2)**2) #(This is according to original code, not sure why)

    # All of these below are tf, which means they will be stored as graphs and will run later. 
    # B and S are the trainable parameters
    B_ = tf.Variable(B,dtype=tf.float32,name='B_0') #creating tf.variable to make it a trainable parameter. 
    S_ = tf.Variable(np.identity(L*M) - np.matmul(B, A),dtype=tf.float32,name='S_0' )

    By_ = tf.matmul(B_,y_)  # This will be the input for the shrinkage function in the first iteration

    # creating layers 
    layers = []
    layers.append( ('Linear',By_,None) )   
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    lam0_ = tf.Variable(initial_lambda,name='lam_0')  #to make it learnable

    xhat_ = eta( By_, lam0_) #first itertion and xhat_0 is all 0s
    layers.append( ('LISTA T=1',xhat_, (lam0_,) ) )

    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) ) # using the same lambda
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, lam_ )  # will be stored as graphs which will run during a session later.   
        layers.append( ('LISTA T='+str(t+1),xhat_,(lam_,)) ) # creating layers (xhat_ is an operation)         

    '''
    print statements to check the layers
    print(*layers , sep = "\n")
    print(lam0_)
    for name,xhat_,var_list in layers:
          print(xhat_)
    '''

    #######..........SETUP TRAINING OF LISTA..............########
    trinit,refinements,final_refine = 1e-3, (.5,.1,.01), None
    losses_=[]
    nmse_=[]
    trainers_=[]

    """assert is used when, If the statement is true then it does nothing and continues 
    the execution, but if the statement is False then it stops the execution of the program 
    and throws an error.
    """
    assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    tr_ = tf.Variable(trinit,name='tr',trainable=False) #Learning rate
    training_stages=[]

    nmse_denom_ = tf.nn.l2_loss(x_)  #computes half of l2 loss without the sqrt

    for name,xhat_,var_list in layers:
        loss_  = tf.nn.l2_loss( xhat_ - x_) #original values of x will be stored in the placeholder
        nmse_  = tf.nn.l2_loss( (xhat_ - x_) ) / nmse_denom_ # computing the nmse.
        if var_list is not None:
                train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list) 
                training_stages.append( (name,xhat_,loss_,nmse_,train_,var_list) )
        for fm in refinements:
                train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_)
                training_stages.append( (name+' trainrate=' + str(fm) ,xhat_,loss_,nmse_,train2_,()) )
                
    if final_refine:
        train2_ = tf.train.AdamOptimizer(tr_*final_refine).minimize(loss_)
        training_stages.append( (name+' final refine ' + str(final_refine) ,xhat_,loss_,nmse_,train2_,()) )

    # print(*training_stages , sep = "\n")    
    ivl, maxit, better_wait =10,1000000,3000
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(beta_val), yval=la.norm(y_val) ) )

    state = load_trainable_vars(sess,'LISTA_SPARC_6_R0p5_01.npz') # must load AFTER the initializer

        # must use this same Session to perform all training
        # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done=state.get('done',[])
    log=str(state.get('log',''))

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={y_:y_val,x_:beta_val})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(100*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin()-1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
            #y,x = prob(sess)
            y, x= sess.run( ( ygen_,xgen_ ) ) # generates different y,x for every iteration.
            sess.run(train_,feed_dict={y_:y,x_:x} )
        done = np.append(done,name)

        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,'LISTA_SPARC_L{T}_EN{E}.npz'.format(T=T,E=EbN0_dB[e]),**state)

