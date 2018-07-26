#!/usr/bin/python

"""
Exports the trained model, dataset, feature list, and log files to Google Cloud Platform.
"""

from __future__ import print_function
import datetime
import os
import subprocess
import sys

import common_configs as CONFIG

BUCKET_ID = "adept-elf-206419-mlengine"
GS_PATH_PREFIX = os.path.join('gs://', BUCKET_ID,
                              datetime.datetime.now().strftime('enron_poi_classifier_%Y%m%d_%H%M%S'))


def main():
    print("Uploading the trained model, dataset, feature list, and log to {}\n".format(GS_PATH_PREFIX))

    gs_model_path = os.path.join(GS_PATH_PREFIX, 'model.pkl')
    subprocess.check_call(['gsutil', 'cp', CONFIG.EXPORT_CLF_FILENAME, gs_model_path], stderr=sys.stdout)

    gs_dataset_path = os.path.join(GS_PATH_PREFIX, 'dataset.pkl')
    subprocess.check_call(['gsutil', 'cp', CONFIG.EXPORT_DATASET_FILENAME, gs_dataset_path], stderr=sys.stdout)

    gs_feature_list_path = os.path.join(GS_PATH_PREFIX, 'feature_list.txt')
    subprocess.check_call(['gsutil', 'cp', CONFIG.EXPORT_FEATURE_LIST_FILENAME, gs_feature_list_path],
                          stderr=sys.stdout)

    gs_log_path = os.path.join(GS_PATH_PREFIX, 'log.txt')
    subprocess.check_call(['gsutil', 'cp', CONFIG.EXPORT_LOG_FILENAME, gs_log_path], stderr=sys.stdout)


if __name__ == '__main__':
    main()
