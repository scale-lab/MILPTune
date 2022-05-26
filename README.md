# MILPTune

## Getting Started

### 0. Data Management
We use [MongoDB](https://mongodb.com/) as a data store solution.
It helps in the MILP sampling process as well as in the deployment of learned models for inference.
While model training does not require MongoDB, it is a central component for efficient storage and parallel execution of offline space exploration using [SMAC](https://github.com/automl/SMAC3).

So, as a first step, download and run MongoDB.
You may also run it using the official Docker image [here](https://hub.docker.com/_/mongo).

After successfully running a DB server, perform the following steps:
1. Create a database called `milptunedb`.
2. Create collections for each dataset with the dataset name. You should have 3 collections named: `1_item_placement`, `2_load_balancing`, and `3_anonymous`. In addition, create a collection named `milptune_metadata`.

> You may use [MongoDB Compass](https://www.mongodb.com/products/compass) to make data management easier from a GUI.


### 1. Download Code/Data

1. In your home directory, download this repo to your home directory.
2. Download dataset and trained model from the [following link](https://drive.google.com/file/d/1-qzBym0TBsfk4WuemB9ffuTyvFNY5s7u/view?usp=sharing)
3. Extract downloaded data using `tar -xzv milptune-data.tar.gz`
4. Copy trained models to the repository using `cp -r milptune-data/models MILPTune`. This will add trained models and learned embeddings to the `models` folder.
5. Import parameters configuration data (inside `milptune-data/dataset`) to the database -- each in its corresponding collection. This step will import all the configurations explored during the offline exploration in this work. The database is queried with a MILP instance embedding to gets its stored parameters configurations and their associated costs.
6. Inside the `MILPTune` directory, create a `.env` file from `.env.example` and enter database server connection credentials.
7. Setup the environment according to the section below.

**Environment Setup**
There are two options to set up the environment: 

(1) using `conda env create -n milptune -f conda.yaml` to install dependencies in a new conda environment. Activate the environment using `conda activate milptune`. After that, run `python setup.py install` from inside the MILPTune directory to install the package.

or

(2) using the `docker build -t milptune .` to build an image with all dependencies encapsulated. Run the docker image using `docker run -it -v <local-data-dir>:/data milptune`. This also mount the downloaded data directory to `/data` inside the container to access cached data.


### 2. Run MILPTune

To predict a parameters configuration for a given problem instance, simply run:

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

where `<instance-file>` is the MILP instance file in `.mps` format, `<dataset-name>` is the dataset to predict from (i.e. `1_item_placement`, `2_load_balancing`, or `3_anonymous`). The parameters `n_neighbors` and `n_configs` instruct MILPTune to predict a single parameters configuration from the nearest neighbor.
If you are running from inside docker, the `<instance-file>` would be `/data/<instance-file-path>`.
Also, make sure that the MongoDB server is reachable from inside the container.

The function `solve_milp` calls the SCIP solver with the predicted parameters configuration and reports back the solution, cost (i.e. primal bound) and time.

There is also a helper script inside the `evaluation` folder:

```Shell
python evaluation/configure.py <instance-file> <dataset-name>
```

---
## Reproducing The Whole Training Process


### 1. Data Preprocessing
Data preprocessing is the step of converting all dataset instances into their bipartite graph representation and saving them in the database for the training pipeline.

This can be done using:

```Python
from milptune.train.preprocess import index_bipartite

index_bipartite("<path-to-train-instances>", "<dataset-name>")
```
This will preprocess all instances in the `train` directory of a dataset to its corresponding collection in the database.

### 2. Training
We provide the training script for each dataset in the `models` directory.
The scripts load data from the database and caches them locally in the first epoch, but then uses the cached tensors for the subsequent epochs.

### 3. Indexing Embeddings
After training, the collection of `milptune_metadata` needs to be populated with the necessary metadata for the trained model.
Below is the schema of the document:

```JSON
{  
    "<dataset-name>": {
        "model": "~/MILPTune/models/<dataset-name>/model.pt",
        "dims": {
            "embedding_dim": <embedding-dimension>,
            "n_gnn_layers": <number-of-gnn-layers>,
            "gnn_hidden_dim": <hidden-dimension-size>
        }
    }
}

```


After that, embeddings of the training instances are indnexed back into the database for the nearest neighbor search process.

This can be achieved by:
```Python
from milptune.train.index import index_embeddings

index_embeddings("<dataset-name")
```

### 4. Evaluation
Evaluation and comparison against the default configuration and against using SMAC can be found under `evaluation/evaluate.py`.

You can run it on a single instance using:

```Shell
python evaluate.py <instance-path> <dataset-name> <output-dir>
```

We provide `evaluate.sh` to run the evaluation on all instances in a given directory using:

```
./evaluate.sh /path/to/dataset/valid/ <dataset-name> <output-dir>
```

