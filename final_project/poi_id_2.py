#!/usr/bin/python

import sys
import pickle
import pandas
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from numpy import random

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from termcolor import colored

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
from sklearn import svm, naive_bayes, linear_model, tree, ensemble, neighbors, semi_supervised, neural_network, discriminant_analysis
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import utils

import warnings
warnings.filterwarnings("ignore")


param_grid = {
    "ExtraTreeClassifier": {
        "criterion": ['gini', 'entropy'],
        "max_depth": [5, 10, 25, 50, 100, 250, 500, 1000, None],
        "min_impurity_decrease": [0.0, 0.1, 0.2, 0.5]
    },
    "LinearSVC": {
        # "loss": ['hinge', 'squared_hinge'],
        "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "tol": [0.01, 0.001, 0.0001, 0.00001]
    }
}

score_color_thresholds = {
    'accuracy': [0.8, 0.6],
    'other': [0.3, 0.2]
}

def train(algorithm, training_feature_data, training_target_data):
    algo_name = str(algorithm).split('(')[0]
    # scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    # print algorithm.get_params().keys()
    model = GridSearchCV(algorithm, param_grid=param_grid[algo_name], cv=5, scoring='precision')
    model.fit(training_feature_data, training_target_data)
    print "Best Params:", model.best_params_, "\n"
    return model

def score(model, features_test, labels_test):
    predictions = model.predict(features_test)
    results = {}
    results["accuracy"]  = metrics.accuracy_score(predictions, labels_test)
    results["f1_score"]  = metrics.f1_score(predictions, labels_test)
    results["recall"]    = metrics.recall_score(predictions, labels_test)
    results["precision"] = metrics.precision_score(predictions, labels_test)
    return results

def conditional_color(score, score_type='other'):
    th = score_color_thresholds[score_type]
    text = str(round(score * 100, 2))
    if score > th[0]:
        return colored(text, 'green')
    elif score > th[1]:
        return colored(text, 'yellow')
    else:
        return colored(text, 'red')

def report(clf_type, score):
    print colored(clf_type, 'white', attrs=['bold'])
    print "Accuracy: ", conditional_color(score["accuracy"], score_type='accuracy'), " \t", \
        "F1 Score: ", conditional_color(score["f1_score"]), " \t", \
        "Recall: ", conditional_color(score["recall"]), "   \t", \
        "Precision: ", conditional_color(score["precision"])
    print

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    'poi',
    'salary',
    'bonus',
    'exercised_stock_options',
    'from_poi_to_this_person',
    'from_this_person_to_poi'
] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']

# Exploration
# for key, value in data_dict.iteritems():
#     print key
# print "\nThere are {} people in this dataset. Each has these columns:\n".format(len(data_dict))
# print data_dict['LAY KENNETH L'], "\n"

### Task 3: Create new feature(s)
# TODO ...?

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# plt.plot(features[0], reg.predict(features[0]), color="blue") # Based on regression prediction
# x_axis = 0
# y_axis = 1
# # for feature, target in zip(features, labels):
# #     plt.scatter(feature, target)
# plt.xlabel(features_list[x_axis + 1])
# plt.ylabel(features_list[y_axis + 1])
# plt.show()

### Task 4: Try a varity of classifiers
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
# scaler = MinMaxScaler()
# scaler.fit(features)
# print "Scaler max:", scaler.data_max_
# print "Scaler min:", scaler.data_min_
# scaled_features = scaler.transform(features)

random_state = random.randint(0, 2**32-1)
# random_state = 3753554380 # 89, 50, 50, 50
# random_state = 1328639293 # 94, 66, 100, 50
print "Random State Int:", random_state, "\n"
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=random_state)
# sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25)
# for train_idx, test_idx in sss.split(features, labels):
#     features_train = []
#     features_test  = []
#     labels_train   = []
#     labels_test    = []
#
#     for ii in train_idx:
#         features_train.append( features[ii] )
#         labels_train.append( labels[ii] )
#     for jj in test_idx:
#         features_test.append( features[jj] )
#         labels_test.append( labels[jj] )
#
#     algorithm = svm.LinearSVC()
#     clf = train(algorithm, features_train, labels_train)
#     score_metrics = score(clf, features_test, labels_test)
#     report((str(algorithm).split('(')[0]), score_metrics)

# pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
# n_components = 2
# pca = PCA(svd_solver='randomized', n_components=n_components, whiten=True).fit(features_train)
# X_train_pca = pca.transform(features_train)
# X_test_pca = pca.transform(features_test)
#
# for i in range(n_components):
#     print "pca.explained_variance_ratio_[{}]".format(i)
#     print pca.explained_variance_ratio_[i]
# print pca.components_[0]

algorithm = tree.ExtraTreeClassifier(random_state=random_state)
# algorithm = svm.LinearSVC(random_state=random_state)
cv = train(algorithm, features_train, labels_train)
clf = cv.best_estimator_
score_metrics = score(clf, features_test, labels_test)
report(str(clf), score_metrics)
# report((str(algorithm).split('(')[0]), score_metrics)

# for i, f in enumerate(clf.feature_importances_):
#     if f > 0.2: print i, ':', f, ':', vectorizer.get_feature_names()[i]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)