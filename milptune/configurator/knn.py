import metric_learn
import numpy as np
from joblib import load
from sklearn.neighbors import KNeighborsTransformer

from milptune.db.connections import get_client
from milptune.db.helpers import from_mongo_binary, to_mongo_binary
from milptune.features.A import get_A


def _check_(test, array):
    return list(
        map(
            lambda i: i[0],
            filter(lambda i: i[1], [(i, np.allclose(x, test)) for i, x in enumerate(array)]),
        )
    )  # noqa


def get_configuration_parameters(instance_file: str, dataset_name: str, n_neighbors=5, n_configs=5):
    # 0. Connect to database
    client = get_client()
    db = client.milptunedb
    dataset = db["milptune_metadata"]

    # 1. Load metric learning model
    r = dataset.find_one({f"{dataset_name}.mlkr.model": {"$exists": True}})
    if not r:
        raise Exception("Cannot find saved model file")
    mlkr: metric_learn.MLKR = load(r[dataset_name]["mlkr"]["model"])

    # 2. Transform instance to new metric space
    r = dataset.find_one(
        {
            f"{dataset_name}.vars_index": {"$exists": True},
            f"{dataset_name}.conss_index": {"$exists": True},
        }
    )
    if not r:
        raise Exception(f"Cannot load metadata for {dataset_name}")

    mask = from_mongo_binary(r[dataset_name]["mask"])
    A = get_A(instance_file, r[dataset_name]["vars_index"], r[dataset_name]["conss_index"])
    X = np.asarray(A.todense().flatten())
    X = X.reshape(1, -1)
    X = np.delete(X, mask, axis=1)
    X_mlkr = mlkr.transform(X)

    # 3. Load full data to get knn
    dataset = db[dataset_name]
    r = dataset.find(
        {"A_mlkr": {"$exists": True}, "configs": {"$exists": True}}, sort=[("path", 1)]
    )
    X_mlkr_trained_list = []
    for instance in r:
        A = from_mongo_binary(instance["A_mlkr"])
        X_mlkr_trained_list.append(A)
    X_mlkr_trained = np.vstack(X_mlkr_trained_list)

    # 4. Run knn
    transformer = KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance", n_jobs=-1)
    transformer.fit(X_mlkr_trained)
    distances, neighbors = transformer.kneighbors(X_mlkr, return_distance=True)
    neighbors = X_mlkr_trained[neighbors[0], :]
    neighbors = list(map(lambda n: to_mongo_binary(n.flatten()), neighbors))

    # 5. Get all configs of the k nearest neighbors
    dataset = db[dataset_name]
    distances = distances.flatten()
    configs, config_distances = [], []
    for index, neighbor in enumerate(neighbors):
        # We do this query one at a time (and not using Mongo $in operator to preserve order)
        r = dataset.find_one({"A_mlkr": neighbor}, projection=["configs"])
        instance_configs = sorted(r["configs"], key=lambda c: c["cost"])
        configs.extend(instance_configs[:n_configs])
        config_distances.extend([distances[index]] * len(instance_configs[:n_configs]))

    # 6. Suggest `n_configs` configurations with lowest cost from all neighbors
    suggested = list(zip(configs, config_distances))
    suggested = sorted(suggested, key=lambda c: c[0]["cost"])
    suggested_configs, distances = zip(*suggested)

    return suggested_configs, distances
