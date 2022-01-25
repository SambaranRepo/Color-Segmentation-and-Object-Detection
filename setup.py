from setuptools import setup, find_packages
setup(
    name='example',
    version='0.1.0',
    packages=find_packages(include=['bin_detection', 'pixel_classification', 'bin_detection.*', 'pixel_classification.*', 'tests', 'tests.*'])
)