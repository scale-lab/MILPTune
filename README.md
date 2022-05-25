# MILPTune

## Getting Started

### Load Processed Dataset

Download and import processed data (`.json` collection files) from the [following link](https://drive.google.com/drive/folders/12BYk-EqtlJ_Ext5lVgsoPGp_qNz_89X8?usp=sharing).


### Predict Configuration Parameters
In order to predict a parameters configuration, follow the below steps.

#### **STEP 1:** Environment Setup
There are two options to set up the environment: 

(1) using `conda env create -n milptune -f conda.yaml` to install dependencies in a new conda environment. Activate the environment using `conda activate milptune`. After that, run `python setup.py install` to install MILPTune package.

or

(2) using the `docker build -t milptune .` to build an image with all dependencies encapsulated. Run the docker image using `docker run -it -v <local-data-dir>:/  --gpus all milptune`. This also mount the downloaded data directory to `/` inside the container to access cached data (bypass connecting to the DB).


#### **STEP 2:** Run MILPTune
To predict a parameters configuration, simply run:

```Python
from milptune.configurator.knn import get_configuration_parameters
from milptune.scip.solver import solve_milp

configs, distances = get_configuration_parameters(
    instance_file=<instance-file.mps.gz>,
    dataset_name=<dataset-name>,
    n_neighbors=1, n_configs=1)

solution, cost, time = solve_milp(
    params=configs[0]['params'],
    instance=<instance-file.mps.gz>)
```

where <instance-file> is the MILP instance file in `.mps` format, <dataset-name> is the dataset to predict fromt (i.e. `1_item_placement`, `2_load_balancing`, or `3_anonymous`). The parameters `n_neighbors` and `n_configs` instruct MILPTune to predict a single parameters configuration from the nearest neighbor.

The function `solve_milp` calls the SCIP solver with the predeicted parameters configuration and reports back the solution, cost (i.e. primal bound) and time.

---
## Reproducing The Whole Training Process

### 0. Data Management
We use [MongoDB](https://mongodb.com/) as a data store solution.
It helps in the MILP sampling process as well as in the deployment of learned models for inference.
While model training does not require MongoDB, it is a central component for efficient storage and parallel execution of offline space exploration using [SMAC](https://github.com/automl/SMAC3).

So, as a first step, download and run MongoDB service.
You may also run it using Docker official image [here](https://hub.docker.com/_/mongo).

After successfully running a DB server, perform the following steps:
1. Create a `.env` file from `.env.example` and enter database server connection credentials.
2. Create a database called `milptunedb`.
3. Create collections for each dataset with the dataset name. You should have 3 collections named: `1_item_placement`, `2_load_balancing`, and `3_anonymous`. In addition, create a collection named `milptune_metadata`.

> You may use [MongoDB Compass](https://www.mongodb.com/products/compass) to make data management easier from a GUI.

### 1. Environment Setup
There are two options to set up the environment: 

(1) using `conda env create -n milptune -f conda.yaml` to install dependencies in a new conda environment. Activate the environment using `conda activate milptune`. After that, run `python setup.py install` to install MILPTune package.

(2) using the `docker build -t milptune .` to build an image with all dependencies encapsulated. Run the docker image using `docker run -it -v <local-data-dir>:/  --gpus all milptune`. This also mount the downloaded data directory to `/` inside the container to access cached data (bypass connecting to the DB).


### 2. Data Preprocessing
TBA

### 3. Training
TBA

### 4. Indexing Embeddings
TBA

### 5. Evaluation
TBA