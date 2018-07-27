#!/usr/bin/env bash

# Train the Final Project model on Google Cloud Machine Learning Engine

# Activate the python environment
source "/home/nkarasch/anaconda3/bin/activate" udacity

PROJECT_ID="adept-elf-206419"
BUCKET_ID="${PROJECT_ID}-mlengine"
REGION="us-central1"
TRAINING_PACKAGE_PATH="/home/nkarasch/GitStuff/ud120-projects/src/"
MAIN_TRAINER_MODULE="trainer.task"
MODEL_NAME="enron_poi_classifier"
JOBNAME="${MODEL_NAME}_$(date -u +%y%m%d_%H%M%S)"

TRAIN_DATA_PATH="gs://${BUCKET_ID}/resources/dataset.pkl"
OUTPUT_DIR="gs://${BUCKET_ID}/${JOBNAME}"

# Train locally (uncomment below)

#echo "cd src"
#cd src
#TRAINING_PACKAGE_PATH="/home/nkarasch/GitStuff/ud120-projects/src/trainer/"
#TRAIN_DATA_PATH="../resources/dataset.pkl"
#OUTPUT_DIR="../output"
#gcloud ml-engine local train \
#  --package-path ${TRAINING_PACKAGE_PATH} \
#  --module-name ${MAIN_TRAINER_MODULE} \
#  -- \
#  -t $TRAIN_DATA_PATH \
#  -o $OUTPUT_DIR \
#  --no-color
#cd -
#exit

# Train in the cloud
gcloud ml-engine jobs submit training ${JOBNAME} \
        --package-path=$PWD/src/trainer \
        --module-name=${MAIN_TRAINER_MODULE} \
        --region=${REGION} \
        --staging-bucket="gs://${BUCKET_ID}" \
        --scale-tier=BASIC \
        --runtime-version=1.8 \
        -- \
        -t $TRAIN_DATA_PATH \
        -o $OUTPUT_DIR \
        --no-color
