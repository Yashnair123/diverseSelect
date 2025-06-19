import numpy as np


def dgp(n, noise, setting):
    dgps = [dgp_random_hypo1,\
           dgp_random_hypo2]

    return dgps[setting](n,noise)

def dgp_random_hypo1(n,noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = 2*cos(X_1) + gaussian noise. X's are Gaussian.
    """    
    gaussians = np.random.normal(size=5*n).reshape(n,5)
    Xs = gaussians

    Ys = 2*np.cos(Xs[:,0])+noise*np.random.normal(size=n)

    return Xs,Ys


def dgp_random_hypo2(n,noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = 3.5 - ||X||_2^2 + gaussian noise. X's are sampled according to
    Gaussian mixture.
    """
    Xs = np.random.normal(size=5*n).reshape(n,5)
    Ys = 3.5-(np.linalg.norm(Xs,axis=1)**2)+noise*np.random.normal(size=n)

    return Xs,Ys