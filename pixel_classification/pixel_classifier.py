'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
from pixel_classification.generate_rgb_data import read_pixels
from pixel_classification import pixel_classifier
import pickle
from glob import glob
import os,sys
folder_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder_path, 'parameters.pkl')
class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    with open(model_path, 'rb') as f:
      params = pickle.load(f)
      
    self.mu_1, self.cov_1, self.mu_2, self.cov_2, self.mu_3, self.cov_3 = params[0],params[1],params[2], params[3], params[4], params[5]
    self.prior_1, self.prior_2, self.prior_3 = params[6], params[7], params[8]
       
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Just a random classifier for now
    # Replace this with your own approach 
    # y = 1 + np.random.randint(3, size=X.shape[0])

    y = np.empty(len(X))
    
    for i in range(len(X)): 
      
      red_log_likelihood = self.gaussian_posterior_likelihood(X[i], self.mu_1, self.cov_1, self.prior_1)
      green_log_likelihood = self.gaussian_posterior_likelihood(X[i], self.mu_2, self.cov_2, self.prior_2)
      blue_log_likelihood = self.gaussian_posterior_likelihood(X[i], self.mu_3, self.cov_3, self.prior_3)
      likelihood = [red_log_likelihood, green_log_likelihood, blue_log_likelihood]
      y[i] = 1 + np.argmin(likelihood)
    
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

  def gaussian_posterior_likelihood(self, X, mu, cov, prior): 
        '''
        : Compute the log of the probability of class given X
        : output: log(P(X|class)) + log(P(class)) = (X - mu)^T * cov^(-1) * (X - mu) + log(det(cov)) - 2*log(P(class))
        '''
        return ((X - mu).T.dot(np.linalg.inv(cov)).dot(X - mu)) + np.log(np.linalg.det(cov)) - 2 * np.log(prior)

