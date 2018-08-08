import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

def genCovX(C, n): # helper function to create N(0, C)
    # C is the covariance matrice (assume to be psd)
    # n is number of examples
    A = np.linalg.cholesky(C)
    d, _ = C.shape
    Z = np.random.randn(n, d)
    X = Z.dot(A.T) 
    return X.astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_risk(nrgroups, nirgroups, pergroup, experiment):
    if experiment == "binary_r":
        nrvars = nrgroups * pergroup
        r = np.zeros(nrvars)
        # vary percentage of known features in each group
        for i in range(0, nrvars//pergroup):
            r[(i*pergroup):(i*pergroup+i)] = 1
        r = np.concatenate((r,r))
    elif experiment == "corr":
        rbase = np.zeros(pergroup)
        rbase[:pergroup//2] = 1
        r = np.concatenate([rbase for _ in range(nrgroups)])
    return r

############## experiment 4.3.2 data generation ############
def sweepCov():
    nrgroups = 10
    nirgroups = 0
    pergroup = 30
    n = 5000
    # setup                                                                                                                                                                        
    risk = generate_risk(nrgroups, nirgroups, pergroup, "corr")
    # gen data                                                                                                                                                                     
    correlations = [i/nrgroups for i in range(nrgroups)]
    blocks = []
    for c in correlations:
        base = np.diag(np.ones(pergroup))
        base[base==0] = c
        blocks.append(base)
    C = block_diag(*blocks)
    theta = np.ones(nrgroups*pergroup)
    datagen = lambda: genCovData(C=C, theta=theta,
                                 n=n, signoise=5)
    return datagen, risk

# example usage
'''
d, risk = sweepCov()
X, y = d()
'''

############## experiment 4.3.3 data generation ############
def genCovData(C, theta, n=5000, signoise=5):
    # C is the covariance matrice (assume to be psd)
    noise = np.random.randn(n) * signoise
    X = genCovX(C, n)
    y = (X.dot(theta) + noise > 0).reshape(n,1)
    return X.astype(np.float32), y.astype(np.float32).reshape(y.size,1) 

def sweepBinaryR():
    nrgroups = 11
    nirgroups = nrgroups
    pergroup = 10
    n = 5000
    # setup
    risk = generate_risk(nrgroups, nirgroups, pergroup, "binary_r")
    # gen data
    base = np.diag(np.ones(pergroup))       
    base[base==0] = 0.99
    C = block_diag(*([base]*(nrgroups+nirgroups)))
    theta = np.zeros((nrgroups + nirgroups) * pergroup)
    theta[:nrgroups*pergroup] = 1
    datagen = lambda: genCovData(C=C, theta=theta,
                                 n=n, signoise=15)    
    return datagen, risk

# example usage
'''
d, risk = sweepBinaryR()
X, y = d()
'''

############## experiment 4.3.4 data generation ############
def genDiffTheta(n=1000): # bernoulli so noise also on y
    ng = np.random.poisson(10)
    w = np.random.normal(0,10,ng)
    nd = np.random.poisson(20,ng)
    d = nd.sum() # feature dimension
    risk = np.random.randint(0,2,d)
    # linearly distribute w_i to theta_i according to risk_i
    theta = risk.copy().astype(np.float32)
    istart = 0
    for ndi in nd:
        iend = istart+ndi
        theta[istart:iend] /= risk[istart:iend].sum()
        istart = iend
    theta *= np.repeat(w, nd)
    InGrpCov = 0.99
    blocks = []
    for ndi in nd:
        block = np.diag(np.ones(ndi))
        block[block==0] = InGrpCov
        blocks.append(block)
    CovM = block_diag(*blocks)
    def _datagen():
        X = genCovX(C=CovM, n=n)
        y = sigmoid(X.dot(theta))
        for i in range(n):
            y[i] = np.random.binomial(1,y[i]) # bernoulli
        return X.astype(np.float32), y.astype(np.float32).reshape(y.size,1) 
        
    return _datagen, (theta, risk, nd, CovM)

# example usage
'''
d, (_, risk, _, _) = genDiffTheta()
X, y = d()
'''
