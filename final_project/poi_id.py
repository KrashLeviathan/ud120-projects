#!/usr/bin/python

import sys
import pickle
import pandas
sys.path.append("../tools/")
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from termcolor import colored

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
from sklearn import svm, naive_bayes, linear_model, tree, ensemble, neighbors, semi_supervised, neural_network, discriminant_analysis
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")


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

# Exploration
# for key, value in data_dict.iteritems():
#     print key
# print "\nThere are {} people in this dataset. Each has these columns:\n".format(len(data_dict))
# print data_dict['LAY KENNETH L'], "\n"

### Task 2: Remove outliers
# TODO ...?

### Task 3: Create new feature(s)
# TODO ...?

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print features
print "Number of rows:", len(labels)

# plt.plot(features[0], reg.predict(features[0]), color="blue") # Based on regression prediction
x_axis = 2
y_axis = 3
plt.scatter(features[x_axis], features[y_axis])
plt.xlabel(features_list[x_axis + 1])
plt.ylabel(features_list[y_axis + 1])
plt.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
algorithms = [
   # svm.SVC(kernel='linear', C = 1.0),   # QUITE SLOW
    # linear_model.SGDClassifier(),
    naive_bayes.GaussianNB(),
    naive_bayes.MultinomialNB(),
    naive_bayes.BernoulliNB(),
    # tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    # ensemble.ExtraTreesClassifier(),
    svm.LinearSVC(),
   neural_network.MLPClassifier(),   # VERY SLOW
    # neighbors.NearestCentroid(),
    # ensemble.RandomForestClassifier(),
    # linear_model.RidgeClassifier(),
]

param_grid = {
    "SGDClassifier": {
        "penalty": ['l2', 'l1', 'elasticnet'],
        "power_t": [0.4, 0.5, 0.6],
        "warm_start": [True, False],
        "average": [True, False]
    },
    "GaussianNB": {},
    "MultinomialNB": {},
    "BernoulliNB": {
        "alpha": [0.8, 0.9, 1.0],
        "fit_prior": [True, False]
    },
    "DecisionTreeClassifier": {
        "criterion": ['gini', 'entropy'],
        "max_depth": [5, 10, 25, 50, 100, 250, 500, 1000, None],
        "min_impurity_decrease": [0.0, 0.1, 0.2, 0.5]
    },
    "ExtraTreeClassifier": {
        "criterion": ['gini', 'entropy'],
        "max_depth": [5, 10, 25, 50, 100, 250, 500, 1000, None],
        "min_impurity_decrease": [0.0, 0.1, 0.2, 0.5]
    },
    "ExtraTreesClassifier": {
        # "criterion": ['gini', 'entropy'],
        # "max_depth": [5, 10, 25, 50, 100, 250, 500, 1000, None],
        # "min_impurity_decrease": [0.0, 0.1, 0.2, 0.5]
    },
    "LinearSVC": {
        # "loss": ['hinge', 'squared_hinge'],
        "C": [0.8, 0.9, 1.0],
        # "class_weight": [],
        "multi_class": ['ovr', 'crammer_singer']
    },
    "NearestCentroid": {},
    "RandomForestClassifier": {},
    "RidgeClassifier": {},
    "SVC": {},
    "LogisticRegressionCV": {},
    "MLPClassifier": {}
}

def train(algorithm, training_feature_data, training_target_data):
    # model = Pipeline([
    #     # ('vect', CountVectorizer()),
    #     # ('tfidf', TfidfTransformer()),
    #     ('clf', algorithm)
    # ])
    algo_name = str(algorithm).split('(')[0]
    model = GridSearchCV(algorithm, param_grid=param_grid[algo_name], cv=3, scoring='precision')
    model.fit(training_feature_data, training_target_data)
    return model

def score(model, features_test, labels_test):
    predictions = model.predict(features_test)
    results = {}
    results["accuracy"]  = metrics.accuracy_score(predictions, labels_test)
    results["f1_score"]  = metrics.f1_score(predictions, labels_test)
    results["recall"]    = metrics.recall_score(predictions, labels_test)
    results["precision"] = metrics.precision_score(predictions, labels_test)
    return results

score_color_thresholds = {
    'accuracy': [0.8, 0.6],
    'other': [0.3, 0.2]
}

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

# Check each algorithm and report on the scores of each
def test_algorithms(features_train, features_test, labels_train, labels_test):
    for algorithm in algorithms:
        model = train(algorithm, features_train, labels_train)
        score_metrics = score(model, features_test, labels_test)
        report((str(algorithm).split('(')[0]), score_metrics)
        # report(str(algorithm), score_metrics)

# clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point.
# TODO: Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# selector = SelectKBest(f_classif, k=3)
# selector.fit(features_train, labels_train)
# # features_train = selector.transform(features_train).toarray()
# # features_test  = selector.transform(features_test).toarray()
# features_train = selector.transform(features_train)
# features_test  = selector.transform(features_test)

test_algorithms(features_train, features_test, labels_train, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# TODO
# dump_classifier_and_data(clf, my_dataset, features_list)