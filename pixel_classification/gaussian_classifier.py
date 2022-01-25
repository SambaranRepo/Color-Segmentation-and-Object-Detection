'''
: Code to train a gaussian classifier
'''
import numpy as np
from generate_rgb_data import read_pixels
import pickle
from glob import glob

class GaussianClassifier(): 
    def __init__(self, red, green, blue): 
        self.red, self.green, self.blue = red, green, blue
        self.total = len(self.red) + len(self.green) + len(self.blue)

    def estimate_prior(self, x, total): 
        '''
        : Estimate class prior probability as P(class) = #samples in class / total length of dataset 
        '''
        return len(x)/total
  
    def estimate_mean(self, x): 
        '''
        : MLE of mean
        : mu = 1/N * sum(x)
        '''
        return np.mean(x, axis = 0)

    def estimate_covariance(self, x, mu): 
        '''
        : MLE for covariance
        : sigma = 1/N * sum(x - mu)(x - mu)^T
        '''
        return np.cov(x.T)

    def gaussian_classifier(self): 
        '''
        : Estimate the parameters of a gaussian classifier seperately for red, green and blue. 
        : Get the images which are red, blue and green. 
        : Compute prior class probability for red, blue and green i.e. P(B), P(G), P(R) as P(X = i) = #samples of i / total samples
        : Compute the mean and covariance of class conditional probabilities
        : mean and covariance of the distribution P(X | class) = G(X | mu_class, sigma_class) where X is the RGB pixel values and class is the color using MLE. 
        : We will have three sets of parameters :  mu_r, sigma_r), (mu_g, sigma_g), (mu_b, sigma_b). (RGB) = (1,2,3)
        '''
        mu_1 = self.estimate_mean(self.red)
        mu_2 = self.estimate_mean(self.green)
        mu_3 = self.estimate_mean(self.blue)
        cov_1 = self.estimate_covariance(self.red, mu_1)
        cov_2 = self.estimate_covariance(self.green, mu_2)
        cov_3 = self.estimate_covariance(self.blue, mu_3)
        prior_1 = self.estimate_prior(self.red, self.total)
        prior_2 = self.estimate_prior(self.green, self.total)
        prior_3 = self.estimate_prior(self.blue, self.total)
        params = [mu_1, cov_1, mu_2, cov_2, mu_3, cov_3, prior_1, prior_2, prior_3]
        with open('parameters.pkl', 'wb') as f: 
            pickle.dump(params,f)
        return mu_1, cov_1, mu_2, cov_2, mu_3, cov_3, prior_1, prior_2, prior_3

    def gaussian_posterior_likelihood(self, X, mu, cov, prior): 
        '''
        : Compute the log of the probability of class given X
        : output: log(P(X|class)) + log(P(class)) = (X - mu)^T * cov^(-1) * (X - mu) + log(det(cov)) - 2*log(P(class))
        '''
        return ((X - mu).T.dot(np.linalg.inv(cov)).dot(X - mu)) + np.log(np.linalg.det(cov)) - 2 * np.log(prior)

if __name__ == '__main__':
  folder = glob('pixel_classification/data/training')[0]
  X1 = read_pixels(folder+'/red', verbose = False)
  X2 = read_pixels(folder+'/green')
  X3 = read_pixels(folder+'/blue')
  y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
  X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))

  red = X[np.where(y == 1)]
  green = X[np.where(y==2)]
  blue = X[np.where(y==3)]

  gaussian = GaussianClassifier(red, green, blue)
  print(gaussian.gaussian_classifier())

  