'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


from __future__ import division

from generate_rgb_data import read_pixels
from pixel_classifier import PixelClassifier
from glob import glob
import numpy as np

if __name__ == '__main__':
  # test the classifier
  
  folder = glob('pixel_classification/data/validation/red')[0]
  
  X = read_pixels(folder)
  myPixelClassifier = PixelClassifier()
  y = np.asarray(myPixelClassifier.classify(X))
  y_true = np.asarray([1]*len(y))
  print(f"Precision : {np.mean(y == y_true):.4f}")

  
