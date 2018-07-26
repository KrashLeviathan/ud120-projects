#!/usr/bin/python

from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(
        name='trainer',
        version='1.0',
        author='Nathan Karasch',
        author_email='nate.karasch@zirous.com',
        packages=find_packages(),
        description='A machine learning model for predicting Persons of Interest in the Enron scandal.'
    )
