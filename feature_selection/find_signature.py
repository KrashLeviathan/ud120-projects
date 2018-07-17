#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
# words_file = "word_data_overfit.pkl"
# authors_file = "email_authors_overfit.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )

################################################################################
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(input='content', stop_words='english', sublinear_tf=True, max_df=0.5)
# transformed_data = tfidf.fit_transform(word_data)
# feature_names = tfidf.get_feature_names()
# print "Number of feature names:", len(feature_names)
# print "Number they got:", 38757
# print "feature_names[34597]:", feature_names[34597]
#
# print "feature 32134:", feature_names[32134]
################################################################################


### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
print
print features_test
features_test  = vectorizer.transform(features_test).toarray()
print
print features_test

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]


### your code goes here
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

print "Score:", clf.score(features_test, labels_test)
for i, f in enumerate(clf.feature_importances_):
    if f > 0.2: print i, ':', f, ':', vectorizer.get_feature_names()[i]

# The answer they were looking for was 33614: 0.765
