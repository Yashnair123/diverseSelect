import numpy as np


def dgp(n,noise):
    """
    This function generates a dataset of Xs, mus, and Y's directly (instead of just X's and Y's). 
    Y = mus + gaussian noise (simple because so as to analytically calculate true likelihood). 
    mus's are Gaussian and so are X's (which we diversify on)
    """    
    gaussians = np.random.normal(size=5*n).reshape(n,5)
    Xs = gaussians
    
    mus = np.sum(Xs, axis=1)/np.sqrt(5)
    Ys = mus + np.random.normal(scale=noise, size=n)

    return Xs,Ys