## ud120-projects

### Getting started

#### 1. Clone the project

```bash
git clone https://github.com/KrashLeviathan/ud120-projects.git
```

#### 1. [Install Anaconda](https://www.anaconda.com/download/)

#### 2. [Create thegi environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

```bash
conda create --yes -n ud120 python=2.7 \
    pip numpy termcolor sklearn scipy
source activate ud120
pip install --upgrade pip
```

#### 3. View usage information

```bash
python -m src.trainer.task --help
```

#### 4. Run the model locally with default settings

```bash
python -m src.trainer.task
```
