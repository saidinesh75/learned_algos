import numpy as np
import tensorflow.compat.v1 as tf
np.random.seed(1)
tf.set_random_seed(1)
# from tools import problems,networks,train
import math
import sys
import numpy.linalg as la
BASE_PATH = "/home/saidinesh/learned_algos/Outputs_learned_algos" # path to save the npz files

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!! (info and warnings are not printed)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
    np.savez(os.path.join(BASE_PATH, filename),**save)

def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        # for k,d in np.load(filename).items():
        for k,d in np.load(os.path.join(BASE_PATH, filename)).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other 

M, N, L, pnz, SNR, kappa = 250, 500, 1000, 0.03, 40, 0
tf.compat.v1.disable_eager_execution()

#create an np matrix A and also a tf tensor (make it constant, since it need not be trained)
A = np.random.normal(size=(M, N), scale=1.0 / math.sqrt(M)).astype(np.float32)
A_ = tf.constant(A,name='A')  #Made a tf constant since it need not be trained

#generate sparse x values drawn from gaussian iid, and non zero locations selected uniformly(with sparsity 0.1).
bernoulli_ = tf.cast(tf.random.uniform((N, L))<pnz, tf.float32) 
xgen_ = bernoulli_ * tf.random.normal((N,L)) 

noise_var = pnz* N/M * math.pow(10., -SNR/10.)
ygen_ = tf.matmul(A_,xgen_) + tf.random.normal( (M,L),stddev=math.sqrt( noise_var ) )

xval = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
yval = np.matmul(A,xval) + np.random.normal(0,math.sqrt( noise_var ),(M,L))

xinit = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
yinit = np.matmul(A,xinit) + np.random.normal(0,math.sqrt( noise_var ),(M,L))

# Specifying the number of layers
T, initial_lambda = 8, 0.1

# creating placeholders, which will hold dataset later.
x_ = tf.compat.v1.placeholder( tf.float32,(N,None),name='x' )
y_ = tf.compat.v1.placeholder( tf.float32,(M,None),name='y' )

eta = simple_soft_threshold                 # creating an object
B = A.T / (1.01 * la.norm(A,2)**2)          # Normalised columns

# All of these below are tf, which means they will be stored as graphs (eager execution disabled) and will run later. 
# B and S are the trainable parameters
B_ = tf.Variable(B,dtype=tf.float32,name='B_0') #creating tf.variable to make it a trainable parameter. 
S_ = tf.Variable(np.identity(N) - np.matmul(B, A),dtype=tf.float32,name='S_0' )

By_ = tf.matmul(B_,y_)  # This will be the input for the shrinkage function in the first iteration

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

for name,xhat_,var_list in layers:
    print(var_list)

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

tf.disable_v2_behavior()                                        # compatibility

nmse_denom_ = tf.nn.l2_loss(x_)                                 # computes half of l2 loss without the sqrt

for name,xhat_,var_list in layers:
    loss_  = tf.nn.l2_loss( xhat_ - x_)                         # original values of x will be stored in the placeholder
    nmse_  = tf.nn.l2_loss( (xhat_ - x_) ) / nmse_denom_        # computing the nmse.
    if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list) 
            training_stages.append( (name,xhat_,loss_,nmse_,train_,var_list) )
    for fm in refinements:
            train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_)
            training_stages.append( (name+' trainrate=' + str(fm) ,xhat_,loss_,nmse_,train2_,()) )
            
if final_refine:
    train2_ = tf.train.AdamOptimizer(tr_*final_refine).minimize(loss_)
    training_stages.append( (name+' final refine ' + str(final_refine) ,xhat_,loss_,nmse_,train2_,()) )

ivl, maxit, better_wait =10,1000000,3000
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(xval), yval=la.norm(yval) ) )

state = load_trainable_vars(sess,'LISTA_L8_M250_N500.npz') # must load AFTER the initializer

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
            nmse = sess.run(nmse_,feed_dict={y_:yval,x_:xval})
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
    save_trainable_vars(sess,'LISTA_L8_M250_N500.npz',**state)  # change at the load funtion too!


####........... TESTING.............####

xtest = ((np.random.uniform( 0,1,(N,1))<pnz) * np.random.normal(0,1,(N,1))).astype(np.float32)
ytest = np.matmul(A,xtest) + np.random.normal(0,math.sqrt( noise_var ),(M,1))

count = 0
for name,xhat_,var_list in layers:
    count = count + 1
    if count != T+1:
        continue
    else:
        out = sess.run(xhat_,feed_dict={y_:ytest,x_:xtest})

print("Done")