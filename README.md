## About
* We build a Gaussian Discriminant Model and a Mixture of Gaussian model to segment an image into blue and non-blue regions. 
* A detector module then takes the segmented image as input, applies some morphology and smoothening operations on the segmented image. After that 
we apply open-cv findContour and boundingRect methods to get contours along the blue areas in the image. Combined with some shape statistics, we filter 
these regions into possible blue recycling bin regions and return the bounding boxes around these bins.

## File Details
Files are structured into two directories :- 
1. bin_detection : This directory contains all code files used to generate training data from the images using roipoly in the rgb and yuv space, 
train the Gaussian Discriminant Model and the Mixture of Gaussians Model, and get the bounding boxes to the blue recycling in the image.

<pre>
bin_detection/
├── bin_detection_rgb.pkl
├── bin_detection_ycrcb.pkl
├── bin_detector.py
├── gaussian_classifier.py
├── generate_color_data.py
├── generate_ycrcb_data.py
├── __init__.py
├── mog.py
├── mog_rgb.pkl
├── mog_yuv.pkl
├── requirements.txt
├── roipoly
│   ├── __init__.py
│   ├── roipoly.py
│   └── version.py
├── test_bin_detector.py
└── test_roipoly.py
</pre>

###  a. generate_color_data.py: 
This script is used to generate rgb pixels using roipoly from the given images. We draw two regions of interest : a positive region containing blue 
recycling bin pixels and a negative region containing blue pixels not belonging to a recycling bin and other colored objects. 
### b. generate_ycrcb_data.py: 
This script is the same as generate_color_data.py, except that this generates the pixels in YUV space. 
### c. gaussian_classifier.py : 
This script is used to train a single gaussian discriminant model on the positive and negative class examples. Usage :
 <pre>
 $python3 bin_detection/gaussian_classifier.py
 </pre>
### d. mog.py
This script is used to train a mixture of gaussian model on the positive and negative class examples. Since expectation maximisation is a complex 
algorithm, the model is trained on a minibatch once at a time. There are hence two modes of using this script : to train a model from the scratch(Mode 1) or to train a saved model using a new minibatch of data(Mode 2). Additionally, we need to give it an input x mentioning from which point we are picking up the data to train the model. Usage : 
<pre>
$python3 bin_detection/mog.py x mode
</pre>
where x > 0 and is an integer mentioning the start of the datapoint and mode = {1,2} mentioning which mode to train. 

### e. bin_detector.py: 
This is the main script that is called to get the segmented image and the bounding box of the blue bins in the given image. This code defines a class that has the methods segment_image and get_bounding_boxes that are used to get the masked image and the bounding boxes around the blue bins respectively.The code takes 2 inputs as arguments : the color space and the mode. Color space can either be rgb or yuv, mode is 1 for using a single gaussian discriminant model, and 2 for using a mixture of gaussian model. Usage : 
<pre>
$python3 bin_detection/bin_detector.py rgb 1
$python3 bin_detection/bin_detector.py rgb 2
$python3 bin_detection/bin_detector.py yuv 1
$python3 bin_detection/bin_detector.py yuv 2
</pre>

2. pixel_classification : This directory contains all code files to classify given pixel as a red, green or blue pixel. 

<pre>
pixel_classification/
├── gaussian_classifier.py
├── generate_rgb_data.py
├── __init__.py
├── parameters.pkl
├── pixel_classifier.py
├── requirements.txt
└── test_pixel_classifier.py
</pre>

### a. gaussian_classifier.py
This script is used to train a single gaussian discriminant model on the red, green and blue class examples. Usage : 
<pre>
$python3 pixel_classifier/gaussian_classifier.py
</pre>



## Technical Report
* [Sambaran Ghosal. "Color Segementation and Object Detection" Feb 2022](report/ColorSegmentationAndBinDetection.pdf)

## Results

### Case 1:
<p float="left">
  <img src="images/mask/0061.eps" width="49%" />
  <img src="images/bin/0061.eps" width="49%" /> 
</p>

### Case 2:
<p float="left">
  <img src="images/mask/0062.eps" width="49%" />
  <img src="images/bin/0062.eps" width="49%" /> 
</p>

### Case 3:
<p float="left">
  <img src="images/mask/0063.eps" width="49%" />
  <img src="images/bin/0063.eps" width="49%" /> 
</p>

### Case 4:
<p float="left">
  <img src="images/mask/0064.eps" width="49%" />
  <img src="images/bin/0064.eps" width="49%" /> 
</p>

### Case 5:
<p float="left">
  <img src="images/mask/0065.eps" width="49%" />
  <img src="images/bin/0065.eps" width="49%" /> 
</p>

### Case 6:
<p float="left">
  <img src="images/mask/0066.eps" width="49%" />
  <img src="images/bin/0066.eps" width="49%" /> 
</p>

### Case 7:
<p float="left">
  <img src="images/mask/0067.eps" width="49%" />
  <img src="images/bin/0067.eps" width="49%" /> 
</p>

### Case 8:
<p float="left">
  <img src="images/mask/0068.eps" width="49%" />
  <img src="images/bin/0068.eps" width="49%" /> 
</p>

### Case 9:
<p float="left">
  <img src="images/mask/0069.eps" width="49%" />
  <img src="images/bin/0069.eps" width="49%" /> 
</p>

### Case 10:
<p float="left">
  <img src="images/mask/0070.eps" width="49%" />
  <img src="images/bin/0070.eps" width="49%" /> 
</p>