import datetime
import os

VERBOSE = False
EXPORT_TO_GCP = True
BUCKET_ID = "adept-elf-206419-mlengine"
GS_PATH_PREFIX = os.path.join('gs://', BUCKET_ID,
                              datetime.datetime.now().strftime('enron_poi_classifier_%Y%m%d_%H%M%S'))

DATASET_DICTIONARY_FILE = "resources/final_project_dataset.pkl"

EXPORT_LOG_FILENAME = "output/log.txt"
EXPORT_CLF_FILENAME = "output/model.pkl"
EXPORT_DATASET_FILENAME = "output/dataset.pkl"
EXPORT_FEATURE_LIST_FILENAME = "output/feature_list.txt"
