from distutils.command.config import config
import joblib
import metric_learn
from joblib import load
from milptune.db.helpers import from_mongo_binary, to_mongo_binary
from milptune.features.A import get_A
from pymongo import MongoClient
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsTransformer
from bson.binary import Binary


def _check_(test, array):
    print(test.shape, array.shape)
    return list(map(lambda i: i[0], filter(lambda i: i[1], [(i, np.allclose(x, test)) for i, x in enumerate(array)])))


def get_configuration_parameters(
        instance_file: str,
        dataset_name: str,
        n_configs=5
        ):
    # 0. Connect to database
    uri = "mongodb://%s:%s@%s:%s" % ('milptune', 'MILP*tune2023', '20.25.127.142', 31331)
    client = MongoClient(uri)
    db = client.milptunedb
    dataset = db['milptune_metadata']

    # 1. Load metric learning model
    r = dataset.find_one({f'{dataset_name}.mlkr.model': {'$exists': True}})
    if not r:
        raise Exception('Cannot find saved model file')
    mlkr: metric_learn.MLKR = load(r[dataset_name]['mlkr']['model'])

    # 2. Transform instance to new metric space
    r = dataset.find_one({f'{dataset_name}.vars_index': {'$exists': True}, f'{dataset_name}.conss_index': {'$exists': True}})
    if not r:
        raise Exception(f'Cannot load metadata for {dataset_name}')
    
    mask = from_mongo_binary(r[dataset_name]['mask'])
    A = get_A(instance_file, r[dataset_name]['vars_index'], r[dataset_name]['conss_index'])
    X = np.asarray(A.todense().flatten())
    X = X.reshape(1, -1)
    X = np.delete(X, mask, axis=1)
    X_mlkr = mlkr.transform(X)

    # 3. Load full data to get knn
    # X_mlkr_trained = np.load(r[dataset_name]['mlkr']['transformed_data']) -- load from database for accuracy
    dataset = db[dataset_name]
    r = dataset.find({'A_mlkr': {'$exists': True}, 'configs': {'$exists': True}}, sort=[('path', 1)])
    X_mlkr_trained = []
    for instance in r:
        A = from_mongo_binary(instance['A_mlkr'])
        X_mlkr_trained.append(A)
    X_mlkr_trained = np.vstack(X_mlkr_trained)
    print(X_mlkr_trained.shape)
    
    print(_check_(X_mlkr.flatten(), X_mlkr_trained))

    # 4. Run knn
    transformer = KNeighborsTransformer(n_neighbors=n_configs, mode='distance', n_jobs=-1)
    transformer.fit(X_mlkr_trained)
    _, neighbors = transformer.kneighbors(X_mlkr)
    neighbors = X_mlkr_trained[neighbors[0],:]    
    neighbors = list(map(lambda n: to_mongo_binary(n.flatten()), neighbors))

    # 5. Get all configs of the k nearest neighbors
    dataset = db[dataset_name]
    r = dataset.find({'A_mlkr': {'$in': neighbors}}, projection=['configs'])
    
    configs = list(map(lambda doc: doc['configs'], r))
    configs = [item for sublist in configs for item in sublist]

    # 6. Suggest `n_configs` configurations with lowest cost from all neighbors
    suggested_configs = sorted(configs, key=lambda c: c['cost'])

    return suggested_configs[:n_configs]
