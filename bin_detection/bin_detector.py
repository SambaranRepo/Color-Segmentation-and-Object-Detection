'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import os,cv2
import skimage
from skimage.measure import label, regionprops
import pickle
from math import pi
from glob import glob
from matplotlib import pyplot as plt
import sys,os

# color_space, mode = 'rgb', 2  Best with autograder and no morphology
color_space, mode = 'yuv', 2
folder_path = os.path.dirname(os.path.abspath(__file__))

if color_space == 'rgb': 
	if mode == 1:
		model_path = os.path.join(folder_path, 'bin_detection_rgb.pkl')
	else: 
		model_path = os.path.join(folder_path, 'mog_rgb.pkl')
elif color_space == 'yuv':
	if mode == 1:
		model_path = os.path.join(folder_path, 'bin_detection_ycrcb.pkl')
	else: 
		model_path = os.path.join(folder_path, 'mog_yuv.pkl')

class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		with open(model_path, 'rb') as f: 
			params = pickle.load(f)
		
		self.mu_1, self.cov_1, self.mu_0, self.cov_0, self.prior_1, self.prior_0 = params[0], params[1], params[2], params[3], params[4], params[5]
		
		if mode == 1:
			self.mu_1 = self.mu_1[:,None].T
			self.mu_0 = self.mu_0[:,None].T

		if mode == 2:
			self.p1 = 0.43
			self.p0 = 1- self.p1

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE

		if color_space == 'rgb':
			if mode == 1:
				gamma = 1
				invGamma = 1.0 / gamma
				correction_factor = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
				img = cv2.LUT(img,correction_factor) 
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			elif mode == 2:
				gamma = 1
				invGamma = 1.0 / gamma
				correction_factor = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
				img = cv2.LUT(img,correction_factor) 
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = img / 255
			
		elif color_space == 'yuv':
			if mode == 1:
				gamma = 1
				invGamma = 1.0 / gamma
				correction_factor = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
				img = cv2.LUT(img,correction_factor) 
				img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
			elif mode == 2:
				gamma = 1
				invGamma = 1.0 / gamma
				correction_factor = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
				img = cv2.LUT(img,correction_factor) 
				img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
				img = img / 255
			
		X = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
		mask_img = np.zeros(img.shape[0]*img.shape[1], dtype = np.uint8)

		step = 100
		K = len(X)//step
		if mode == 1:
			for i in range(K): 
				bin_likelihood = self.gaussian_posterior_likelihood(X[step*i : step*(i + 1) + 1], self.mu_1, self.cov_1, self.prior_1)
				non_bin_likelihood = self.gaussian_posterior_likelihood(X[step*i : step*(i + 1) + 1], self.mu_0, self.cov_0, self.prior_0)
				mask_img[step*i : step*(i + 1) + 1]  = bin_likelihood < non_bin_likelihood
				
			bin_likelihood = self.gaussian_posterior_likelihood(X[step*K : len(X)], self.mu_1, self.cov_1, self.prior_1)
			non_bin_likelihood = self.gaussian_posterior_likelihood(X[step*K : len(X)], self.mu_0, self.cov_0, self.prior_0)
			mask_img[step*K : len(X)]  = bin_likelihood < non_bin_likelihood
		
		elif mode == 2:
			for i in (range(K)): 
				bin_likelihood = self.mog_prob(X[step*i : step*(i + 1) + 1],  self.prior_1,self.mu_1, self.cov_1,self.p1)
				non_bin_likelihood = self.mog_prob(X[step*i : step*(i + 1) + 1], self.prior_0, self.mu_0, self.cov_0,self.p0)
				mask_img[step*i : step*(i + 1) + 1]  = bin_likelihood > non_bin_likelihood
				
			bin_likelihood = self.mog_prob(X[step*K : len(X)], self.prior_1, self.mu_1, self.cov_1,self.p1)
			non_bin_likelihood = self.mog_prob(X[step*K : len(X)], self.prior_0, self.mu_0, self.cov_0,self.p0)
			mask_img[step*K : len(X)]  = bin_likelihood > non_bin_likelihood

		
		
		mask_img = mask_img.reshape(img.shape[0],img.shape[1])
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - masked image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 

		# YOUR CODE BEFORE THIS LINE
		################################################################
		mask = img
		x_max, y_max = mask.shape[0], mask.shape[1]	
			
		mask *= 255
		
		# kernel = np.ones((13,13), np.uint8)
		# erode = cv2.erode(mask, kernel, iterations = 1)
		# dilation = cv2.dilate(erode, kernel[:5,:5], iterations = 3)
		blurred = cv2.GaussianBlur(mask, (3,3),0)
		ret, thresh = cv2.threshold(blurred, 127, 255,0)

		boxes = []
		similarity = []
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		for cnt in contours:
			approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
			x,y,w,h = cv2.boundingRect(cnt)
			area_ratio = cv2.contourArea(cnt)/(y_max*x_max)
			similarity1 = 100 - np.absolute((h/w)-1.5)*100
			if 0.82<= h/w <=2 and area_ratio > 0.006:
				similarity1 = 100 - np.absolute((h/w)-1.5)*100
				
				boxes.append([x,y,x + w,y + h])
				
				similarity.append(similarity1)
		boxes.sort()
		return boxes

	def gaussian_posterior_likelihood(self, X, mu, cov, prior): 
		'''
		: Compute the log of the probability of class given X
		: output: log(P(X|class)) + log(P(class)) = (X - mu)^T * cov^(-1) * (X - mu) + log(det(cov)) - 2*log(P(class))
		'''
		return np.diag(((X - mu).dot(np.linalg.inv(cov)).dot((X - mu).T))) + np.log(np.linalg.det(cov)) - 2 * np.log(prior)

	def mog_prob(self, x, prior, mu, cov,p): 
		'''
		: Compute the mixture of gaussian probility density function for given input
		: p(x | yi) = (p(x | z1 )* pi1 + p(x | z2) * pi2 + p(x | z3)*pi3)
		: p(yi,x) = p(x | yi) * p(yi)
		: p(yi) is the prior class probability of class i which equals number of samples of i / total samples
		'''
		prob = np.zeros((len(x)))
		for i in range(3): 
			prob = np.add(prob, self.gaussian(x, mu[i], np.diag(cov[i])) * prior[i])
		return prob*p
	
	def gaussian(self, x, mu, cov): 
		'''
		'''
		d = np.shape(x)[1]
		dr = 1/(np.sqrt((2* pi)**(d) * np.linalg.det(cov)))
		nr = (np.exp(-np.diag((x - mu)@(np.linalg.inv(cov))@((x - mu).T) / 2)))
		return nr * dr