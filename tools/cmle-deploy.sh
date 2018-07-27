#!/usr/bin/env bash

usage() {
  echo "USAGE: cmle-deploy.sh -j <job_id>" 1>&2
}

while getopts ":j:" opt; do
  case ${opt} in
    j ) # job id
      JOB_ID=$OPTARG
      ;;
    \? )
      usage
      exit 1
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument" 1>&2
      usage
      exit 1
      ;;
  esac
done

if [ ${OPTIND} -eq 1 ]; then
  usage
  exit 1
fi


PROJECT_ID="adept-elf-206419"
BUCKET_ID="${PROJECT_ID}-mlengine"
REGION="us-central1"
MODEL_NAME="enron_poi_classifier"
JOBNAME="${MODEL_NAME}_${JOB_ID}"

OUTPUT_DIR="gs://${BUCKET_ID}/${JOBNAME}"
MODEL_DIR="${OUTPUT_DIR}/model"
VERSION_NAME="v${JOB_ID}"
FRAMEWORK="SCIKIT_LEARN"

# Move model to its own folder for deployment
echo "NOTE: If model was previously deployed, the next command will error, but the deployment will still work."
gsutil mv "${OUTPUT_DIR}/model.pkl" "${MODEL_DIR}/model.pkl"

# Deploy a new version
gcloud ml-engine versions create ${VERSION_NAME} \
  --model ${MODEL_NAME} --origin ${MODEL_DIR} \
  --runtime-version=1.8 --framework ${FRAMEWORK} \
  --python-version=2.7
