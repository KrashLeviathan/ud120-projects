#!/bin/bash

# Train the Final Project model on Google Cloud Machine Learning Engine

# Activate the python environment
source "/home/nkarasch/anaconda3/bin/activate" udacity

PROJECT_ID="adept-elf-206419"
BUCKET_ID="${PROJECT_ID}-mlengine"
REGION="us-central1"
TRAINING_PACKAGE_PATH="/home/nkarasch/GitStuff/ud120-projects/final_project/"
MAIN_TRAINER_MODULE="final_project.train_and_export_to_gcp"
MODEL_NAME="enron_poi_classifier"

gcloud ml-engine local train \
  --package-path ${TRAINING_PACKAGE_PATH} \
  --module-name ${MAIN_TRAINER_MODULE}
