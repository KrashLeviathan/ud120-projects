#!/usr/bin/python

from __future__ import print_function

import datetime
import os
import pickle
import re
import sys
import time
import warnings

from feature_format import feature_format
from numpy import random
from sklearn import ensemble, svm, naive_bayes, linear_model, tree, neighbors
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.lib.io import file_io
from termcolor import colored

#############################################################################
#####################        Start Configuration        #####################
#############################################################################

import common_configs as CONFIG

##########################################################
###   These two configs are set in the task.py file:   ###
##########################################################
VERBOSE = False
COLORED_OUTPUT = True
OUTPUT_DIR = './output'
##########################################################

PCA_EXPLAINED_VARIANCE_THRESHOLD = 0.05
PCA_FEATURE_CONTRIBUTION_THRESHOLD = 0.2
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
    svm.LinearSVC(random_state=RANDOM_STATE),
    # ensemble.RandomForestClassifier(random_state=RANDOM_STATE),     # Time:  00:02:22
    # svm.SVC(kernel='linear', C = 1.0, random_state=RANDOM_STATE),   # Time:  ........? (long time)
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
    },
    "RandomForestClassifier": SIMPLER_TREE_TYPE_PARAMS,
    "SVC": {},
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

def decolor(*args):
    cleaned_args = []
    for arg in args:
        cleaned_args.append(re.sub("\033\\[\\d+m", "", str(arg)))
    return cleaned_args


### Always print
def aprint(*args, **kwargs):
    cleaned_args = decolor(*args)
    try:
        filename = os.path.join(OUTPUT_DIR, CONFIG.EXPORT_LOG_FILENAME)
        # with open(filename, 'a+') as output:
        #     print(*cleaned_args, file=output, **kwargs)
        with file_io.FileIO(filename, 'a+') as output:
            output.write(*cleaned_args, **kwargs)
            output.write("\n")
    except:
        pass

    if COLORED_OUTPUT:
        print(*args, **kwargs)
    else:
        print(*cleaned_args)


### Print if VERBOSE == True
def vprint(*args, **kwargs):
    cleaned_args = decolor(*args)
    try:
        filename = os.path.join(OUTPUT_DIR, CONFIG.EXPORT_LOG_FILENAME)
        # with open(filename, 'a+') as output:
        #     print(*cleaned_args, file=output, **kwargs)
        with file_io.FileIO(filename, 'a+') as output:
            output.write(*cleaned_args, **kwargs)
            output.write("\n")
    except:
        pass

    if VERBOSE:
        if COLORED_OUTPUT:
            print(*args, **kwargs)
        else:
            print(*cleaned_args)


### Print to the stderr stream
def eprint(*args, **kwargs):
    cleaned_args = decolor(*args)
    try:
        filename = os.path.join(OUTPUT_DIR, CONFIG.EXPORT_LOG_FILENAME)
        # with open(filename, 'a+') as output:
        #     print(*cleaned_args, file=output, **kwargs)
        with file_io.FileIO(filename, 'a+') as output:
            output.write(*cleaned_args, **kwargs)
            output.write("\n")
    except:
        pass

    if COLORED_OUTPUT:
        print(*args, file=sys.stderr, **kwargs)
    else:
        print(*cleaned_args, file=sys.stderr)


### Used to print a score (float) in green, yellow, or red based on
### the SCORE_COLOR_THRESHOLDS and given score_type.
def conditional_color(score, score_type='other'):
    th = SCORE_COLOR_THRESHOLDS[score_type]
    text = str(round(score * 100, 3))
    if score >= th[0]:
        return colored(text, 'green')
    elif score >= th[1]:
        return colored(text, 'yellow')
    else:
        return colored(text, 'red')


### Prints the model, model parameters, and metrics
def report(clf, score):
    vprint(colored('  Parameters: (' + '('.join(str(x) for x in str(clf).split('(')[1:]), 'white', attrs=['bold']),
           "\n")
    aprint("  Accuracy: ", conditional_color(score["accuracy"], score_type='accuracy'), " \t",
           "F1 Score: ", conditional_color(score["f1_score"]), " \t",
           "Recall: ", conditional_color(score["recall"]), "   \t",
           "Precision: ", conditional_color(score["precision"]))
    vprint()


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
        # ('scaler', MinMaxScaler()),
        ('algorithm', algorithm)
    ])
    pipe_params = {}
    # noinspection PyCompatibility
    for key, value in PARAM_GRID[algo_name].iteritems():
        pipe_params['algorithm__' + key] = value
    # model = GridSearchCV(pipe, param_grid=PARAM_GRID[algo_name], cv=5, scoring=GRID_SEARCH_SCORING, refit=GRID_SEARCH_FIT, n_jobs=1, error_score=0)
    model = GridSearchCV(pipe, param_grid=pipe_params, cv=5, scoring=GRID_SEARCH_SCORING, refit=GRID_SEARCH_FIT,
                         n_jobs=-1, error_score=0)
    model.fit(feature_data, target_data)
    if print_best_params:
        vprint("Best params for {}: {}\n".format(colored(algo_name, 'green', attrs=['bold']), model.best_params_))
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


### Experimenting with PCA as a means of feature selection
def pca_features_list_revision(features, original_feature_list):
    vprint(colored("################## PCA AS FEATURE SELECTION ##################\n", "blue"))
    pca = PCA()
    pca.fit(features)

    # First entry in original_feature_list is 'poi', so remove it
    ofl_without_poi = original_feature_list[1:]

    new_feature_list_map = {}
    for index, component in enumerate(pca.components_):
        if pca.explained_variance_ratio_[index] < PCA_EXPLAINED_VARIANCE_THRESHOLD:
            vprint("All other components have explained_variance_ratio_ < {}\n"
                   .format(PCA_EXPLAINED_VARIANCE_THRESHOLD))
            break
        vprint("Component {}:  explained_variance_ratio_ = {:.3f}"
               .format(index, pca.explained_variance_ratio_[index]))
        mapped_features = zip(ofl_without_poi, component)
        mapped_features.sort(key=lambda x: -x[1])
        for f in mapped_features:
            if f[1] >= PCA_FEATURE_CONTRIBUTION_THRESHOLD:
                vprint("    {}    {:.2f}".format((f[0]).ljust(26), f[1]))
                new_feature_list_map[f[0]] = 1
        vprint()

    # Add 'poi' to the start of the list again
    return POI_LABEL + new_feature_list_map.keys(), pca


### Using SelectKBest for feature selection
def selectkbest_features_list_revision(features, labels, features_list):
    vprint(
        colored("################## SELECT {} BEST FEATURES ##################\n".format(FEATURE_SELECTION_K), "blue"))
    selector = SelectKBest(f_classif, k=FEATURE_SELECTION_K)
    selector.fit(features, labels)

    mapped_scores = zip(features_list[1:], selector.scores_)
    mapped_scores.sort(key=lambda x: -x[1])
    for index, f in enumerate(mapped_scores):
        feature_name = colored(("[*] " + f[0]).ljust(30), "green") if (index < FEATURE_SELECTION_K) else f[0].ljust(30)
        vprint("{}:  score = {:.3f}"
               .format(feature_name, f[1]))
    vprint()

    # Add 'poi' to the start of the list again
    new_features_list = list(zip(*mapped_scores)[0][:FEATURE_SELECTION_K])
    return POI_LABEL + new_features_list, selector


### Tunes, trains, and evaluates each model based on the precision, recall,
### and f1 scores. Returns the best model that meets the specifications.
def find_best_classifier(features, labels, features_list):
    best_algo_index = -1
    best_model = None
    best_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
    for index, algorithm in enumerate(ALGORITHMS):
        my_start_time = time.time()
        aprint(colored(str(algorithm).split('(')[0], 'white', attrs=['bold']))

        # Use GridSearchCV to tune the model
        clf = train(algorithm, features, labels).best_estimator_

        # Evaluate the model, pulling relevant metrics
        algo_metrics = test_classifier(clf, features, labels)

        my_elapsed_time = time.time() - my_start_time
        aprint("  Training/evaluation time:", time.strftime("%H:%M:%S", time.gmtime(my_elapsed_time)))

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
    things_to_save = {}
    things_to_save[os.path.join(OUTPUT_DIR, CONFIG.EXPORT_CLF_FILENAME)] = clf
    things_to_save[os.path.join(OUTPUT_DIR, CONFIG.EXPORT_DATASET_FILENAME)] = dataset
    things_to_save[os.path.join(OUTPUT_DIR, CONFIG.EXPORT_FEATURE_LIST_FILENAME)] = feature_list

    for filename in things_to_save:
        try:
            thing = things_to_save[filename]
            with file_io.FileIO(filename, 'w') as outfile:
                pickle.dump(thing, outfile)
        except Exception as e:
            eprint(e)


### Main Method
def train_and_evaluate(train_data_path):
    main_start_time = time.time()
    # Write to the log file with the start time. We don't use the aprint method
    # here because we want the file to be overwritten if it already existed before
    try:
        with file_io.FileIO(os.path.join(OUTPUT_DIR, CONFIG.EXPORT_LOG_FILENAME), 'w+') as output:
            output.write(datetime.datetime.now().strftime('Starting time: %Y-%m-%d %H:%M:%S'))
            output.write("\n")
    except Exception as e:
        eprint(e)
    print(datetime.datetime.now().strftime('Starting time: %Y-%m-%d %H:%M:%S'))

    # Print random state that will be used in all calculations
    vprint("\nRandom State: {}\n".format(RANDOM_STATE))

    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    # Automated feature selection will happen later, so for now we
    # cast a wide net.
    features_list = POI_LABEL + FINANCIAL_FEATURES + EMAIL_FEATURES
    vprint("Original feature list before feature selection:  ({} features)\n{}\n".format(len(features_list),
                                                                                         features_list))

    ### Load the dictionary containing the dataset
    data_dict = {}
    try:
        with file_io.FileIO(train_data_path, 'r') as data_file:
            data_dict = pickle.load(data_file)
    except Exception as e:
        # Exiting with a message returns exit code 1
        exit(colored(e, 'red'))

    ### Task 2: Remove outliers
    # This was a row in the dataset totaling all other rows, so we can discard
    vprint("Total number of rows:", len(data_dict))
    vprint("Removing TOTAL row outlier...")
    del data_dict['TOTAL']

    vprint("Total number of rows:", len(data_dict))
    num_poi = sum(data_dict[key]['poi'] for key in data_dict)
    num_non = len(data_dict) - num_poi
    vprint("Number of POI: {} ({:.1f}%)    Number of non-POI: {} ({:.1f}%)".format(
        num_poi,
        100 * float(num_poi) / len(data_dict),
        num_non,
        100 * float(num_non) / len(data_dict)))
    vprint()

    ### Task 3: Create new feature(s)
    # Given that the financial features and email features represent two different
    # types of data, I thought they may make a good combination for a new feature.
    # The choices for that combination were made by taking the "most significant"
    # financial and email features as reported by the SelectKBest algorithm.
    # noinspection PyCompatibility
    for key, person in data_dict.iteritems():
        eso = 0 if person['exercised_stock_options'] == 'NaN' else person['exercised_stock_options']
        srwp = 0 if person['shared_receipt_with_poi'] == 'NaN' else person['shared_receipt_with_poi']
        person['email_financial_combo'] = eso * srwp
    features_list.append('email_financial_combo')
    vprint("Added new feature:  email_financial_combo = exercised_stock_options * shared_receipt_with_poi\n")

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = feature_format.featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = feature_format.targetFeatureSplit(data)
    # I tried using PCA as a means of feature selection, but it didn't work as
    # well as SelectKBest, so it's been commented out.
    # scaled_features = MinMaxScaler().fit_transform(features)
    # features_list, pca = pca_features_list_revision(scaled_features, features_list)
    features_list, selector = selectkbest_features_list_revision(features, labels, features_list)
    # Run these two lines again to select ONLY those best features
    data = feature_format.featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = feature_format.targetFeatureSplit(data)
    vprint("Removing any row containing 0 for all {} selected features".format(FEATURE_SELECTION_K))
    vprint("Total number of rows after feature selection:", len(data))
    num_poi = sum(is_poi for is_poi in labels)
    num_non = len(labels) - num_poi
    vprint("Number of POI: {} ({:.1f}%)    Number of non-POI: {} ({:.1f}%)".format(
        num_poi,
        100 * float(num_poi) / len(labels),
        num_non,
        100 * float(num_non) / len(labels)))
    vprint()

    ### Task 4: Try a varity of classifiers
    # See ALGORITHMS list at top

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script.

    # Find out which algorithm performs best, and select it
    vprint(colored("################## TESTING VARIOUS CLASSIFERS ##################\n", "blue"))
    vprint("GridSearchCV Scoring Metric:", GRID_SEARCH_SCORING)
    vprint("GridSearchCV Fit Metric:", GRID_SEARCH_FIT, "\n")

    best_algo_index, best_model, best_metrics = find_best_classifier(features, labels, features_list)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of model.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    # To make sure we get the best "final" model, train on ALL the data
    aprint(colored("\n################## FINAL MODEL SELECTION ##################\n", "blue"))
    if best_algo_index < 0:
        elapsed_time = time.time() - main_start_time
        print(datetime.datetime.now().strftime('Ending time: %Y-%m-%d %H:%M:%S'))
        aprint("Total elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        sys.exit(colored("None of the models qualified! None achieved precision >= 0.3 and recall >= 0.3\n", "red"))
    else:
        aprint(colored(str(ALGORITHMS[best_algo_index]).split('(')[0], 'white', attrs=['bold']))
        report(best_model, best_metrics)
        aprint("\nFinal feature list after selection:  (", len(features_list), "features )\n", features_list, "\n")

        aprint("Retraining final model for export...\n")
        clf = best_model.fit(features, labels)

        aprint("Saving...")
        save_files(clf, my_dataset, features_list)
        aprint("Done!")

        elapsed_time = time.time() - main_start_time
        print(datetime.datetime.now().strftime('Ending time: %Y-%m-%d %H:%M:%S'))
        aprint("Total elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


###########################################################################
#########        End of Function Definition / Start Script        #########
###########################################################################

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    train_and_evaluate('./resources/dataset.pkl')
