import os

import metric_learn
import numpy as np
from joblib import load
from sklearn.manifold import TSNE

from milptune.db.connections import get_client
from milptune.db.helpers import from_mongo_binary, to_mongo_binary


def index_A_transofrmed(dataset_name):
    np.random.seed(10)
    client = get_client()
    db = client.milptunedb
    dataset = db["milptune_metadata"]

    r = dataset.find_one({f"{dataset_name}.model": {"$exists": True}})
    if not r:
        raise Exception("Cannot find trained model")
    mask = np.load(os.path.expanduser(r[dataset_name]["model"]["mask"]))
    mlkr: metric_learn.MLKR = load(os.path.expanduser(r[dataset_name]["model"]["mlkr"]))
    tsne = None
    if r[dataset_name]["model"].get("tsne", None):
        tsne: TSNE = load(os.path.expanduser(r[dataset_name]["model"]["tsne"]))

    dataset = db[dataset_name]
    r = dataset.find({"A": {"$exists": True}})
    for instance in r:
        A = np.asarray(from_mongo_binary(instance["A"]).todense().flatten())
        X = A.reshape(1, -1)
        X = np.delete(X, mask, axis=1)

        if tsne:
            X = tsne.fit_transform(X)

        X_mlkr = mlkr.transform(X)

        r1 = dataset.update_one(instance, {"$set": {"A_mlkr": to_mongo_binary(X_mlkr.flatten())}})
        print(instance["path"], r1.modified_count)
