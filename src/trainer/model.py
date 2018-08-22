#!/usr/bin/python

import datetime
import os
import pickle
import sys
import time
import warnings

from feature_format import feature_format
from numpy import random
from sklearn import svm, naive_bayes, linear_model, tree, neighbors
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from tensorflow.python.lib.io import file_io

#############################################################################
#####################        Start Configuration        #####################
#############################################################################

import trainer.common_configs as CONFIG

##########################################################
###   These two configs are set in the task.py file:   ###
##########################################################
OUTPUT_DIR = './output'
##########################################################

RANDOM_STATE = random.randint(0, 2 ** 32 - 1)
FEATURE_SELECTION_K = 3
GRID_SEARCH_SCORING = ['f1', 'recall', 'precision']
GRID_SEARCH_FIT = 'f1'

# Used for printing test scores in meaningful colors
# e.g. for 'accuracy'  1.0 to 0.8 is green, 0.8 to 0.6 is yellow,
# and anything below 0.6 is red
SCORE_COLOR_THRESHOLDS = {
    'accuracy': [0.8, 0.6],
    'other': [0.3, 0.2]
}

# The algorithms to be evaluated and selected from
ALGORITHMS = [
    linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, tol=None, random_state=RANDOM_STATE),
    naive_bayes.GaussianNB(),
    naive_bayes.BernoulliNB(),
    neighbors.KNeighborsClassifier(),
    neighbors.NearestCentroid(),
    linear_model.RidgeClassifier(random_state=RANDOM_STATE),
    tree.DecisionTreeClassifier(max_depth=1000, random_state=RANDOM_STATE),
    tree.ExtraTreeClassifier(random_state=RANDOM_STATE),
    svm.LinearSVC(random_state=RANDOM_STATE)
]

# Defines the parameters that GridSearchCV will use for each algorithm tested
TREE_TYPE_PARAMS = {
    "criterion": ['gini', 'entropy'],
    "max_features": ['auto', None, 10, 5, 2],
    "max_depth": [5, 10, 25, 50, None],
    "min_samples_split": [2, 3],
    "min_samples_leaf": [1, 2],
    "min_impurity_decrease": [0.0, 0.1, 0.2, 0.5],
    "class_weight": ['balanced', None]
}
SIMPLER_TREE_TYPE_PARAMS = TREE_TYPE_PARAMS.copy()
SIMPLER_TREE_TYPE_PARAMS.update({
    "n_estimators": [2, 5, 10],
    "max_depth": [5, 10, 25]
})
PARAM_GRID = {
    "SGDClassifier": {},
    "GaussianNB": {},
    "BernoulliNB": {
        "alpha": [1.0, 0.9, 0.5, 0.1, 0.0],
        "binarize": [None, 0.0, 0.1, 0.5, 0.9, 1.0],
        "fit_prior": [True, False]
    },
    "KNeighborsClassifier": {
        "n_neighbors": [1, 2, 5, 10, 20],
        "weights": ['uniform', 'distance'],
        "leaf_size": [2, 4, 8, 16, 32, 64],
        "p": [1, 2]
    },
    "NearestCentroid": {
        "shrink_threshold": [None, 0.1, 0.5, 0.9]
    },
    "RidgeClassifier": {
        "alpha": [1.0, 0.9, 0.5, 0.1, 0.0],
        "normalize": [False, True],
        "tol": [0.01, 0.001, 0.0001, 0.00001],
        "class_weight": [None, 'balanced'],
        "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
    },
    "DecisionTreeClassifier": TREE_TYPE_PARAMS,
    "ExtraTreeClassifier": TREE_TYPE_PARAMS,
    "LinearSVC": {
        # "loss": ['hinge', 'squared_hinge'],
        "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "tol": [0.01, 0.001, 0.0001, 0.00001]
    }
}

POI_LABEL = ['poi']
FINANCIAL_FEATURES = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
EMAIL_FEATURES = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']


#############################################################################
#######        End Configuration / Start Function Definition        #########
#############################################################################

### Print to the stderr stream
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


### Prints the model, model parameters, and metrics
def report(clf, score):
    print('  Parameters: (' + '('.join(str(x) for x in str(clf).split('(')[1:]),
          "\n")
    print("  Accuracy: ", score["accuracy"], " \t",
          "F1 Score: ", score["f1_score"], " \t",
          "Recall: ", score["recall"], "   \t",
          "Precision: ", score["precision"], "\n")


### Uses StratifiedShuffleSplit to test the model against many different
### train/test dataset combinations. This is necessary due to the small
### size of the dataset and the imbalanced class distribution.
### The code was adapted from the tester.py test_classifier method.
def test_classifier(clf, features, labels, folds=1000):
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for train_idx, test_idx in cv.split(features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                eprint("Warning: Found a predicted label not == 0 or 1.\n",
                       "All predictions should take value 0 or 1.\n",
                       "Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        algo_metrics = {
            "accuracy": 1.0 * (true_positives + true_negatives) / total_predictions,
            "precision": 1.0 * true_positives / (true_positives + false_positives),
            "recall": 1.0 * true_positives / (true_positives + false_negatives),
            "f1_score": 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        }
        return algo_metrics
    except:
        eprint("Got a divide by zero when trying out: {}".format(str(clf).split('(')[0]))
        eprint("Precision or recall may be undefined due to a lack of true positive predicitons.\n")
        return False


### Uses GridSearchCV to automatically tune the algorithm.
def train(algorithm, feature_data, target_data, print_best_params=False):
    algo_name = str(algorithm).split('(')[0]
    pipe = Pipeline([
        ('algorithm', algorithm)
    ])
    pipe_params = {}
    # noinspection PyCompatibility
    for key, value in PARAM_GRID[algo_name].items():
        pipe_params['algorithm__' + key] = value
    # model = GridSearchCV(pipe, param_grid=PARAM_GRID[algo_name], cv=5, scoring=GRID_SEARCH_SCORING, refit=GRID_SEARCH_FIT, n_jobs=1, error_score=0)
    model = GridSearchCV(pipe, param_grid=pipe_params, cv=5, scoring=GRID_SEARCH_SCORING, refit=GRID_SEARCH_FIT,
                         n_jobs=-1, error_score=0)
    model.fit(feature_data, target_data)
    if print_best_params:
        print("Best params for {}: {}\n".format(algo_name, model.best_params_))
    return model


### Determines which set of scores is better
def metrics_max(current_best, contender):
    if contender["precision"] < 0.3 or contender["recall"] < 0.3:
        return current_best, False
    f1_diff = contender["f1_score"] - current_best["f1_score"]
    if f1_diff > 0:
        return contender, True
    else:
        return current_best, False


### Using SelectKBest for feature selection
def selectkbest_features_list_revision(features, labels, features_list):
    print("################## SELECT {} BEST FEATURES ##################\n".format(FEATURE_SELECTION_K))
    selector = SelectKBest(f_classif, k=FEATURE_SELECTION_K)
    selector.fit(features, labels)

    mapped_scores = list(zip(features_list[1:], selector.scores_))
    mapped_scores = sorted(mapped_scores, key=lambda x: -x[1])
    for index, f in enumerate(mapped_scores):
        feature_name = ("[*] " + f[0]).ljust(30) if (index < FEATURE_SELECTION_K) else f[0].ljust(30)
        print("{}:  score = {:.3f}".format(feature_name, f[1]))
    print()

    # Add 'poi' to the start of the list again
    sorted_features_tuple = list(zip(*mapped_scores))[0]
    new_features_list = list(sorted_features_tuple[:FEATURE_SELECTION_K])
    return POI_LABEL + new_features_list, selector


### Tunes, trains, and evaluates each model based on the precision, recall,
### and f1 scores. Returns the best model that meets the specifications.
def find_best_classifier(features, labels):
    best_algo_index = -1
    best_model = None
    best_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    for index, algorithm in enumerate(ALGORITHMS):
        my_start_time = time.time()
        print(str(algorithm).split('(')[0])

        # Use GridSearchCV to tune the model
        clf = train(algorithm, features, labels).best_estimator_

        # Evaluate the model, pulling relevant metrics
        algo_metrics = test_classifier(clf, features, labels)

        my_elapsed_time = time.time() - my_start_time
        print("  Training/evaluation time:", time.strftime("%H:%M:%S", time.gmtime(my_elapsed_time)))

        if algo_metrics:
            # Print metrics
            report(clf, algo_metrics)
            # Update the "current best" model for output at the end
            best_metrics, is_new_best = metrics_max(best_metrics, algo_metrics)
            if is_new_best:
                best_algo_index = index
                best_model = clf
        else:
            eprint("Test classifier failed!")

    return best_algo_index, best_model, best_metrics


def save_files(clf, dataset, feature_list):
    things_to_save = {os.path.join(OUTPUT_DIR, CONFIG.EXPORT_CLF_FILENAME): clf,
                      os.path.join(OUTPUT_DIR, CONFIG.EXPORT_DATASET_FILENAME): dataset,
                      os.path.join(OUTPUT_DIR, CONFIG.EXPORT_FEATURE_LIST_FILENAME): feature_list}

    for filename in things_to_save:
        try:
            thing = things_to_save[filename]
            # TODO: Save files
        except Exception as e:
            eprint(e)


### Main Method
def train_and_evaluate(train_data_path):
    main_start_time = time.time()
    print(datetime.datetime.now().strftime('Starting time: %Y-%m-%d %H:%M:%S'))

    # Print random state that will be used in all calculations
    print("\nRandom State: {}\n".format(RANDOM_STATE))

    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    # Automated feature selection will happen later, so for now we
    # cast a wide net.
    features_list = POI_LABEL + FINANCIAL_FEATURES + EMAIL_FEATURES
    print("Original feature list before feature selection:  ({} features)\n{}\n".format(len(features_list),
                                                                                        features_list))

    ### Load the dictionary containing the dataset
    data_dict = {}
    try:
        with file_io.FileIO(train_data_path, 'rb') as data_file:
            data_dict = pickle.load(data_file)
    except Exception as e:
        # Exiting with a message returns exit code 1
        exit(e)

    ### Task 2: Remove outliers
    # This was a row in the dataset totaling all other rows, so we can discard
    print("Total number of rows:", len(data_dict))
    print("Removing TOTAL row outlier...")
    del data_dict['TOTAL']

    print("Total number of rows:", len(data_dict))
    num_poi = sum(data_dict[key]['poi'] for key in data_dict)
    num_non = len(data_dict) - num_poi
    print("Number of POI: {} ({:.1f}%)    Number of non-POI: {} ({:.1f}%)".format(
        num_poi,
        100 * float(num_poi) / len(data_dict),
        num_non,
        100 * float(num_non) / len(data_dict)))
    print()

    ### Task 3: Create new feature(s)
    # Given that the financial features and email features represent two different
    # types of data, I thought they may make a good combination for a new feature.
    # The choices for that combination were made by taking the "most significant"
    # financial and email features as reported by the SelectKBest algorithm.
    # noinspection PyCompatibility
    for key, person in data_dict.items():
        eso = 0 if person['exercised_stock_options'] == 'NaN' else person['exercised_stock_options']
        srwp = 0 if person['shared_receipt_with_poi'] == 'NaN' else person['shared_receipt_with_poi']
        person['email_financial_combo'] = eso * srwp
    features_list.append('email_financial_combo')
    print("Added new feature:  email_financial_combo = exercised_stock_options * shared_receipt_with_poi\n")

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = feature_format.featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = feature_format.targetFeatureSplit(data)
    features_list, selector = selectkbest_features_list_revision(features, labels, features_list)
    # Run these two lines again to select ONLY those best features
    data = feature_format.featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = feature_format.targetFeatureSplit(data)
    print("Removing any row containing 0 for all {} selected features".format(FEATURE_SELECTION_K))
    print("Total number of rows after feature selection:", len(data))
    num_poi = sum(is_poi for is_poi in labels)
    num_non = len(labels) - num_poi
    print("Number of POI: {} ({:.1f}%)    Number of non-POI: {} ({:.1f}%)".format(
        num_poi,
        100 * float(num_poi) / len(labels),
        num_non,
        100 * float(num_non) / len(labels)))
    print()

    ### Task 4: Try a varity of classifiers
    # See ALGORITHMS list at top

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script.

    # Find out which algorithm performs best, and select it
    print("################## TESTING VARIOUS CLASSIFERS ##################\n")
    print("GridSearchCV Scoring Metric:", GRID_SEARCH_SCORING)
    print("GridSearchCV Fit Metric:", GRID_SEARCH_FIT, "\n")

    best_algo_index, best_model, best_metrics = find_best_classifier(features, labels)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of model.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    # To make sure we get the best "final" model, train on ALL the data
    print("\n################## FINAL MODEL SELECTION ##################\n")
    if best_algo_index < 0:
        elapsed_time = time.time() - main_start_time
        print(datetime.datetime.now().strftime('Ending time: %Y-%m-%d %H:%M:%S'))
        print("Total elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        sys.exit("None of the models qualified! None achieved precision >= 0.3 and recall >= 0.3\n")
    else:
        print(str(ALGORITHMS[best_algo_index]).split('(')[0])
        report(best_model, best_metrics)
        print("\nFinal feature list after selection:  (", len(features_list), "features )\n", features_list, "\n")

        print("Retraining final model for export...\n")
        clf = best_model.fit(features, labels)

        print("Saving...")
        save_files(clf, my_dataset, features_list)
        print("Done!")

        elapsed_time = time.time() - main_start_time
        print(datetime.datetime.now().strftime('Ending time: %Y-%m-%d %H:%M:%S'))
        print("Total elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


###########################################################################
#########        End of Function Definition / Start Script        #########
###########################################################################

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    train_and_evaluate('./resources/dataset.pkl')
