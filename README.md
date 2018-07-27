## ud120-projects

### Getting started

#### 1. Clone the project

```bash
git clone https://github.com/KrashLeviathan/ud120-projects.git
```

#### 1. [Install Anaconda](https://www.anaconda.com/download/)

#### 2. [Create the environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

```bash
conda create --yes -n ud120 python=2.7 \
    pip numpy termcolor sklearn scipy
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

```bash
./tools/cmle-train.sh

# Take note of the job id, which is the numeric identifiers at the end
# of the job name. E.g. For job name enron_poi_classifier_180727_145537,
# the job id will be 180727_145537.

# Pass the job id as the deployment script parameter. 
./tools/cmle-deploy.sh -j <job_id>
```
