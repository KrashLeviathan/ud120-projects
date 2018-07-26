#!/usr/bin/python

import argparse
import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains and evaluates a machine learning model to predict Persons of Interest in the Enron scandal.'
    )
    parser.add_argument(
        '--train-data-path',
        help='A path to the data in cloud storage or in the local file system',
        nargs=1,
        default=['./resources/dataset.pkl']
    )
    parser.add_argument(
        '--output-dir',
        help='The directory to output the model, logs, etc',
        nargs=1,
        default=['./output']
    )
    args = parser.parse_args()

    # Run the training job
    model.OUTPUT_DIR = args.__dict__['output_dir'][0]
    model.train_and_evaluate(args.__dict__['train_data_path'][0])
