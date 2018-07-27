#!/usr/bin/python

from setuptools import find_packages, setup

REQUIRED_PACKAGES = ['pip>=18.0', 'requests>=2.18.0', 'setuptools>=34.0.0', 'pip', 'numpy', 'termcolor', 'scikit-learn', 'scipy', 'tensorflow']

if __name__ == '__main__':
    setup(
        name='enron_poi_classifier',
        version='1.0',
        author='Nathan Karasch',
        author_email='nate.karasch@zirous.com',
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(),
        description='A machine learning model for predicting Persons of Interest in the Enron scandal.'
    )
