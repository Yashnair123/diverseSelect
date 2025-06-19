import numpy as np


def dgp(n, noise, setting):
    dgps = [dgp_random_hypo1,\
           dgp_random_hypo2]

    return dgps[setting](n,noise)

def dgp_random_hypo1(n,noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = X1*X2 + gaussian noise. X's are sampled according to
    Gaussian mixture.
    """
    means = np.array([
        [1,-1,1.],
        [.75,4,2.],
        [-2,-1.5,1.],
        [1.5,2,1.5],
        [-5,3,2.]
    ])
    
    prob = np.array([0.2,0.4/3,0.4/3,0.4/3,0.4])
    zs = np.random.choice(5, size=n,p=prob).astype(int)
    
    gaussians = np.random.normal(size=3*n).reshape(n,3)
    Xs = means[zs]+0.5*gaussians

    Ys = Xs[:,0]*Xs[:,1]+Xs[:,2]+noise*np.random.normal(size=n)

    return Xs,Ys


def dgp_random_hypo2(n,noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = ||X||_2^2-3.5 + gaussian noise. X's are sampled according to
    Gaussian mixture.
    """
    Xs = np.random.normal(size=5*n).reshape(n,5)
    Ys = (np.linalg.norm(Xs,axis=1)**2)-3.5+noise*np.random.normal(size=n)

    return Xs,Ys