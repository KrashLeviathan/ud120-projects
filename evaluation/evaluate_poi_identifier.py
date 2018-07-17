#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features,
    labels,
    test_size=0.3,
    random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
print "Score:\t", clf.score(X_test, y_test)

prediction_results = clf.predict(X_test)
num_pois = 0
for i in range(len(prediction_results)):
    result = prediction_results[i]
    if result == 1:
        num_pois += 1
        if y_test[i] == 1:
            print "True Positive"
        else:
            print "False Positive"

print "Number of POIs predicted:\t", num_pois
print "Number of people in test set:\t", len(X_test)

print "Precision:", metrics.precision_score(y_test, prediction_results)
print "Recall:", metrics.recall_score(y_test, prediction_results)
