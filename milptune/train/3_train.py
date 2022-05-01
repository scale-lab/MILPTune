import metric_learn
import numpy as np
from joblib import dump
from pymongo import ReturnDocument

from milptune.db.connections import get_client
from milptune.db.helpers import from_mongo_binary, to_mongo_binary
from milptune.viz.tsne import plot_tsne


def load_from_database(dataset_name):
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]

    X = []
    y = []
    r = dataset.find({'incumbent': {'$exists': True}}, sort=[('path', 1)])
    for instance in r:
        A = np.asarray(from_mongo_binary(instance['A']).todense().flatten())

        if float(instance['incumbent'][-1]['cost']) < 100:
            X.append(A)
            y.append(float(instance['incumbent'][-1]['cost']))
    X = np.vstack(X)
    y = np.vstack(y)

    mask1 = np.where(np.sum(X, axis=0) == 0)            # delete features that are all zeros in all samples
    mask2 = np.where(np.sum(X, axis=0) == X.shape[0])   # delete features that are all 1 in all samples
    mask = np.hstack([mask1[0], mask2[0]])

    X = np.delete(X, mask, axis=1)
    np.save(f'X.{dataset_name}.npy', X)
    np.save(f'y.{dataset_name}.npy', y)
    np.save(f'mask.{dataset_name}.npy', mask)

    # Save mask to database metadata
    dataset = db['milptune_metadata']
    dataset.find_one_and_update(
        {dataset_name: {'$exists': True}},
        {'$set': {f'{dataset_name}.mask': to_mongo_binary(mask)}},
        upsert=True, return_document=ReturnDocument.AFTER)

    return X, y


def train_mlkr(dataset_name, load_local=False):
    if load_local:
        X = np.load(f'X.{dataset_name}.npy')
        y = np.load(f'y.{dataset_name}.npy')
    else:
        X, y = load_from_database(dataset_name)

    plot_tsne(X, y, f'{dataset_name}.png')
    mlkr = metric_learn.MLKR()
    print('Starting fitting ..')
    X_mlkr = mlkr.fit_transform(X, y.flatten())

    print('Fit complete ..')
    np.save(f'X_mlkr.{dataset_name}.npy', X_mlkr)
    dump(mlkr, f'mlkr.{dataset_name}.gz')

    plot_tsne(X_mlkr, y, f'{dataset_name}.mlkr.png')
