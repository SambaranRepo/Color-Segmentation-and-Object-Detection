'''
: Code to train a gaussian classifier
'''
import numpy as np
import pickle
from glob import glob

class GaussianClassifier(): 
    def __init__(self, X_pos, X_neg): 
        '''
        : Intitialise the class. 
        : Get the training data pixels. 
        '''
        self.X_pos = X_pos
        self.X_neg = X_neg
        self.total = len(self.X_pos) + len(self.X_neg)

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
        : Get the pixels which are classified as either blue or not blue. 
        : Compute prior class probability for blue and not blue class i.e. P(B), P(NB)= #samples of i / total samples
        : Compute the mean and covariance of class conditional probabilities
        : mean and covariance of the distribution P(X | class) = G(X | mu_class, sigma_class) where X is the pixel values and class is the color using MLE. 
        : We will have two sets of parameters :  (mu_b, sigma_b), (mu_nb, sigma_nb). y = {1,0}
        '''
        mu_1 = self.estimate_mean(self.X_pos)
        mu_0 = self.estimate_mean(self.X_neg)
        
        cov_1 = self.estimate_covariance(self.X_pos, mu_1)
        cov_0 = self.estimate_covariance(self.X_neg, mu_0)
        
        prior_1 = self.estimate_prior(self.X_pos, self.total)
        prior_0 = self.estimate_prior(self.X_neg, self.total)

        params = [mu_1, cov_1, mu_0, cov_0, prior_1, prior_0]
        
        return params
  
if __name__ == '__main__':
    
    with open('full_ycrcb_data.pkl', 'rb') as f: 
        X = pickle.load(f)
        X_pos, X_neg = X[0], X[1]
        print(f"Number of samples in blue recycle bin : {X_pos.shape}")
        print(f"Number of samples in non blue recycle bin: {X_neg.shape}")
    bin_detection_classifier = GaussianClassifier(X_pos, X_neg)
    params = bin_detection_classifier.gaussian_classifier()
    
    

    # with open('bin_detection_rgb.pkl', 'wb') as f: 
    #         pickle.dump(params,f)

    with open('bin_detection_ycrcb.pkl', 'wb') as f: 
            pickle.dump(params,f)
            
    # with open('bin_detection_hsv.pkl', 'wb') as f: 
    #         pickle.dump(params,f)