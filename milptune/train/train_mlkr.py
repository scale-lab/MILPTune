import os

import metric_learn
import numpy as np
from joblib import dump
from pymongo import ReturnDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

from milptune.db.connections import get_client
from milptune.db.helpers import from_mongo_binary
from milptune.viz.tsne import plot_tsne


def load_from_database(dataset_name, cost_threshold, model_dir=""):
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]

    X = []
    y = []
    r = dataset.find({"default_config": {"$exists": True}}, sort=[("path", 1)])
    for instance in r:
        A = np.asarray(from_mongo_binary(instance["A"]).todense().flatten())
        X.append(A)
        if float(instance["default_config"][0]["cost"]) < cost_threshold:
            y.append(float(instance["default_config"][0]["cost"]))
        else:
            y.append(float(cost_threshold))
    X = np.vstack(X)
    y = np.vstack(y)

    mask1 = np.where(np.sum(X, axis=0) == 0)  # delete features that are all zeros in all samples
    mask2 = np.where(
        np.sum(X, axis=0) == X.shape[0]
    )  # delete features that are all 1 in all samples
    mask = np.hstack([mask1[0], mask2[0]]).astype(np.int32)
    X = np.delete(X, mask, axis=1)

    scaler = MinMaxScaler(feature_range=(0, 100))
    y = scaler.fit_transform(y)

    output_dir = os.path.expanduser(model_dir)
    np.save(os.path.join(output_dir, f"X.{dataset_name}.npy"), X)
    np.save(os.path.join(output_dir, f"y.{dataset_name}.npy"), y)
    np.save(os.path.join(output_dir, f"mask.{dataset_name}.npy"), mask)

    # Save mask to database metadata
    dataset = db["milptune_metadata"]
    dataset.find_one_and_update(
        {dataset_name: {"$exists": True}},
        {"$set": {f"{dataset_name}.model.mask": f"{model_dir}/mask.{dataset_name}.npy"}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    return X, y


def train_mlkr(dataset_name, cost_threshold=10000, load_local=False, reduce_dim=None, model_dir=""):
    if load_local:
        print("Loading locally cached data ..")
        output_dir = os.path.expanduser(model_dir)
        X = np.load(os.path.join(output_dir, f"X.{dataset_name}.npy"))
        y = np.load(os.path.join(output_dir, f"y.{dataset_name}.npy"))
        print("Loaded locally cached data ..")
    else:
        print("Loading data from database ..")
        X, y = load_from_database(dataset_name, cost_threshold, model_dir)
        print("Loaded data from database ..")

    if reduce_dim:
        svd = TruncatedSVD(n_components=reduce_dim, algorithm='arpack')
        X = svd.fit_transform(X)
        
        output_dir = os.path.expanduser(model_dir)
        dump(svd, os.path.join(output_dir, f"model.svd.{dataset_name}.gz"))

    plot_tsne(X, y, os.path.join(output_dir, f"{dataset_name}.png"))
    mlkr = metric_learn.MLKR()
    print("Starting fitting ..")
    X_mlkr = mlkr.fit_transform(X, y.flatten())

    print("Fit complete ..")
    output_dir = os.path.expanduser(model_dir)
    np.save(os.path.join(output_dir, f"X_mlkr.{dataset_name}.npy"), X_mlkr)
    dump(mlkr, os.path.join(output_dir, f"model.mlkr.{dataset_name}.gz"))
    plot_tsne(X_mlkr, y, os.path.join(output_dir, f"{dataset_name}.mlkr.png"))

    client = get_client()
    db = client.milptunedb
    dataset = db["milptune_metadata"]

    saved_model = {f"{dataset_name}.model.mlkr": f"{model_dir}/model.mlkr.{dataset_name}.gz"}
    if reduce_dim:
        saved_model[f"{dataset_name}.model.svd"] = f"{model_dir}/model.svd.{dataset_name}.gz"

    _ = dataset.find_one_and_update(
        {f"{dataset_name}": {"$exists": True}},
        {"$set": saved_model},
    )

if __name__ == '__main__':
    train_mlkr("3_anonymous", cost_threshold=10000, load_local=False, reduce_dim=90, model_dir='~/MILPTune/models/3_anonymous')