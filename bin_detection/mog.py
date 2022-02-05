'''
: Try to implement a expectation maximisation on a mixture of gaussians for classification 
'''

from ctypes import sizeof
from random import random
import numpy as np
import cv2,os,sys
import pickle
from scipy.stats import multivariate_normal as mnorm
from math import pi
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('x', help = 'Starting point from which to use the dataset to train EM', type = int)
parser.add_argument('mode',help='Enter 1 to train from beginning or 2 to continue training previously saved parameters', type = int)
args = parser.parse_args()
x = args.x
mode = args.mode
assert isinstance(x , int) and x > 0
assert mode in [1,2]

class MixtureOfGaussians(): 
    def __init__(self, X): 
        '''
        initialise with data 
        '''
        self.X_pos, self.X_neg  = X[0][x:x + 10000], X[1][x:x + 40000]
        self.X_pos = self.X_pos/255
        self.X_neg = self.X_neg/255
        print(f"Number of samples in positive data : {len(X[0])}")
        print(f"Number of samples in negative data : {len(X[1])}")
        self.num_clusters = 3
        
        if mode == 1: 
            self.mu_1 = np.random.random([3,3])
            self.mu_0 = np.random.random([3,3])
            self.prior_1 = np.random.random(3)
            self.prior_0 = np.random.random(3)
            self.prior_1 /=np.sum(self.prior_1)
            self.prior_0 /= np.sum(self.prior_0)
            self.cov_1 = np.asarray([(np.random.randint(1,3,3)),(np.random.randint(1,3,3)), (np.random.randint(1,3,3))])
            self.cov_0 = np.asarray([(np.random.randint(1,3,3)),(np.random.randint(1,3,3)), (np.random.randint(1,3,3))])
            self.cov_1[self.cov_1 < 0.01] = 0.01
            self.cov_0[self.cov_0 < 0.01] = 0.01
        
        elif mode == 2:
            with open('mog_yuv_3.pkl' , 'rb') as f: 
                params = pickle.load(f)
            self.mu_1, self.cov_1, self.mu_0, self.cov_0, self.prior_1, self.prior_0 = params[0], params[1], params[2], params[3], params[4], params[5]
        
        
    
    def train_EM(self, x, prior_old, mu_old, cov_old): 
        '''
        : Obtain the optimal parameters of the Mixture of Gaussians using Expectation Maximisation
        : 
        '''

        for iters in tqdm(range(1000)): 
            N = len(x)
            h = np.zeros((N,self.num_clusters), dtype = np.float32)
            steps = 100
            K = N//steps
            for cluster in range(self.num_clusters):
                for i in range(K):
                    data = x[steps * i: steps * (i + 1), :]
                    h[steps * i: steps * (i + 1),cluster] = self.gaussian(data ,mu_old[cluster],np.diag(cov_old[cluster])) * prior_old[cluster]
                
                h[steps * K : N, cluster] = self.gaussian(x[steps*K : N], mu_old[cluster],np.diag(cov_old[cluster])) * prior_old[cluster]
            
            h = np.divide(h , np.sum(h,axis = 1, keepdims = True))
            prior_new = np.sum(h,axis = 0)/N
            mu_new = np.divide(h.T @ x, np.sum(h,axis = 0).T)
            cov_temp = np.zeros((self.num_clusters, 3))

            for cluster in range(self.num_clusters): 
                cov_temp[cluster, :] = np.divide(h[:,cluster].T @ ((x - mu_new[cluster])*((x - mu_new[cluster]))), np.sum(h[:,cluster]))
            
            cov_new = np.asarray([(cov_temp[0,:]), (cov_temp[1,:]), (cov_temp[2,:])])
            cov_new[cov_new < 0.01] = 0.01
            print(f"Difference in norm value : {np.linalg.norm(mu_new - mu_old)}")
            if np.linalg.norm(mu_new - mu_old) < 1e-3: 
                break

            prior_old, mu_old, cov_old  = prior_new, mu_new, cov_new
        
        return prior_new, mu_new, cov_new

    def get_parameters(self): 
        '''
        : get the trained parameters
        '''
        prior_1, mu_1, cov_1 = self.train_EM(self.X_pos, self.prior_1, self.mu_1, self.cov_1)
        prior_0, mu_0, cov_0 = self.train_EM(self.X_neg, self.prior_0, self.mu_0, self.cov_0)
        params = [mu_1, cov_1, mu_0, cov_0, prior_1, prior_0]
        return params

    def gaussian(self, x, mu, cov): 
        '''
        : Calculates the gaussian pdf value for given input
        '''
        d = np.shape(x)[1]
        mu = mu[None, :]
        dr = 1/(np.sqrt((2* pi)**(d) * np.linalg.det(cov)))
        nr = (np.exp(-np.diag((x - mu)@(np.linalg.inv(cov))@((x - mu).T) / 2)))
        return nr * dr

    
if __name__ == '__main__':
    
    with open('full_yuv_data.pkl', 'rb') as f: 
        X = pickle.load(f)
        print(f"data size : {(X[0].shape)}")
    mog = MixtureOfGaussians(X)
    params = mog.get_parameters()

    for parameter in params : 
        print(f"Shape of parameter {parameter.shape}")

    with open('mog_yuv_3.pkl', 'wb') as f: 
        pickle.dump(params, f)