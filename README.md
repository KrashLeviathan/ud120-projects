## ud120-projects

### Getting started

#### 1. [Install Anaconda](https://www.anaconda.com/download/)

#### 2. [Create an environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

```bash
conda create -n ud120 python=2.7 pip
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
