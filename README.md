## ud120-projects

This code was originally part of the final project for the Udacity course
[Intro to Machine Learning (ud120)](https://www.udacity.com/course/intro-to-machine-learning--ud120).
After completing the course, I wanted to operationalize the model to learn a
bit more about Google's [Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/).
This repository contains scripts to train and deploy the model to Cloud ML Engine.

At time of writing, the ability to deploy a [scikit-learn](http://scikit-learn.org/stable/)
model to Cloud ML Engine is still in beta. It doesn't have the GUI support that tensorflow does,
but you can still train and deploy models via the `gcloud` command line application.

Relevant documentation that I used can be found at the following locations:

- [Original Udacity project files for ud120](https://github.com/udacity/ud120-projects)
- [Training with scikit-learn and XGBoost](https://cloud.google.com/ml-engine/docs/scikit/getting-started-training)
- [How to train Machine Learning models in the cloud using Cloud ML Engine](https://towardsdatascience.com/how-to-train-machine-learning-models-in-the-cloud-using-cloud-ml-engine-3f0d935294b3)
- [Cloud ML Engine -> Documentation -> scikit-learn & XGBoost](https://cloud.google.com/ml-engine/docs/scikit/)
  - [Packaging a Training Application](https://cloud.google.com/ml-engine/docs/scikit/packaging-trainer)
  - [Running a Training Job](https://cloud.google.com/ml-engine/docs/scikit/training-jobs)
  - [Deploying Models](https://cloud.google.com/ml-engine/docs/scikit/deploying-models)
  - [Working with Cloud Storage](https://cloud.google.com/ml-engine/docs/scikit/working-with-cloud-storage)
- [Stack Overflow - 'No such file or directory' error after submitting a training job](https://stackoverflow.com/questions/39775417/no-such-file-or-directory-error-after-submitting-a-training-job)
- [Stack Overflow - Google Storage (gs) wrapper file input/output for CLoud ML?](https://stackoverflow.com/questions/40396552/google-storage-gs-wrapper-file-input-out-for-cloud-ml)
- [GitHub - cloudml-samples](https://github.com/GoogleCloudPlatform/cloudml-samples)


### Getting started

#### 1. Clone the project

```bash
git clone https://github.com/KrashLeviathan/ud120-projects.git
```

#### 1. [Install Anaconda](https://www.anaconda.com/download/)

#### 2. [Create the environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

```bash
conda create --yes -n ud120 python=2.7 \
    pip numpy termcolor scikit-learn scipy tensorflow
source activate ud120
pip install --upgrade pip
```

#### 3. View usage information

```bash
cd src
python -m trainer.task --help
```

#### 4. Run the model locally with default settings

```bash
cd src
python -m trainer.task
```

#### 5. Train and deploy the model in Google Cloud ML Engine

Make sure the [Google Cloud SDK](https://cloud.google.com/sdk/install) is installed
and configured for your Google Cloud Platform instance. You also need to create a
regional storage bucket, enable the appropriate APIs, set permissions, etc. Rather than diving
into how to do all that, I'll let you use the resources listed above to read about all
the GCP requirements for using ML Engine.


```bash
./tools/cmle-train.sh

# Take note of the job id, which is the numeric identifiers at the end
# of the job name. E.g. For job name enron_poi_classifier_180727_145537,
# the job id will be 180727_145537.

# Pass the job id as the deployment script parameter. 
./tools/cmle-deploy.sh -j <job_id>
```
