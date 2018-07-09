#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clfKernel = 'rbf'
clfC = 10000.0
print "\nkernel = ", clfKernel
print "C      = ", clfC
clf = SVC(kernel=clfKernel, C=clfC)

# Limits the training set size
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

print clf.score(features_test, labels_test), "\n"

pred = clf.predict(features_test)
# print "10:", pred[10]
# print "26:", pred[26]
# print "50:", pred[50]

chrisCount = 0
for p in pred:
    chrisCount += p

print "Number of Chris emails predicted:", chrisCount

#########################################################


