import json
import os
import numpy as np
import torch

from sklearn.neighbors import KNeighborsTransformer

from milptune.db.connections import get_client
from milptune.features.bipartite import get_milp_bipartite
from milptune.train.helpers.gnn import ConfigPerformanceRegressor
from milptune.train.helpers.data import MilpBipartiteData


def _check_(test, array):
    return list(
        map(
            lambda i: i[0],
            filter(lambda i: i[1], [(i, np.allclose(x, test)) for i, x in enumerate(array)]),
        )
    )  # noqa


def _load_cached_embeddings_(dataset_name: str, device=torch.device('cpu')):
    local_cache_file = os.path.expanduser(f'~/MILPTune/models/{dataset_name}/{dataset_name}.all.pt')
    if os.path.exists(local_cache_file):
        print('loading local cached embeddings')
        embedded_instances = torch.load(local_cache_file, map_location=device)
        embedded_instances = torch.unique(embedded_instances, dim=0)
        return embedded_instances
    print('cached embeddings not found')


def _load_embeddings_from_db_(dataset_name: str, device=torch.device('cpu')):
    local_cache_file = os.path.expanduser(f'~/MILPTune/models/{dataset_name}/{dataset_name}.all.pt')
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]
    r = dataset.find(
        {"embedding": {"$exists": True}, "configs": {"$exists": True}}, sort=[("path", 1)]
    )
    embedded_instances = []
    
    for instance in r:
        embedded_instances.append(instance["embedding"])
    embedded_instances = torch.tensor(embedded_instances)
    
    torch.save(embedded_instances, local_cache_file)
    return embedded_instances

def _load_cached_model_(dataset_name: str, device=torch.device('cpu')):
    local_cache_file = os.path.expanduser(f'~/MILPTune/models/{dataset_name}/model.pt')
    model_params_file = os.path.expanduser(f'~/MILPTune/models/{dataset_name}/model.json')
    if os.path.exists(local_cache_file) and os.path.exists(model_params_file):
        print('loading local cached model')
        with open(model_params_file, 'r') as f:
            params = json.load(f)
        model = ConfigPerformanceRegressor(params['embedding_dim'], params['n_gnn_layers'], params['gnn_hidden_dim']).to(device)
        model.load_state_dict(torch.load(local_cache_file, map_location=device))
        model.eval()
        return model
    print('cached model not found')

def _load_model_from_db_(dataset_name: str, device=torch.device('cpu')):
    client = get_client()
    db = client.milptunedb
    dataset = db["milptune_metadata"]

    # 1. Load metric learning model
    r = dataset.find_one({f"{dataset_name}.model": {"$exists": True}})
    if not r:
        raise Exception("Cannot find trained model")
    model = ConfigPerformanceRegressor(
        r[dataset_name]['dims']['embedding_dim'],
        r[dataset_name]['dims']['n_gnn_layers'],
        r[dataset_name]['dims']['gnn_hidden_dim']).to(torch.device('cpu'))
    model.load_state_dict(torch.load(r[dataset_name]["model"], map_location=torch.device('cpu')))
    model.eval()

    return model


def get_configuration_parameters(instance_file: str, dataset_name: str, load_cache=True, n_neighbors=5, n_configs=5):
    if load_cache:
        model = _load_cached_model_(dataset_name)
        embedded_instances = _load_cached_embeddings_(dataset_name)
    if model is None or embedded_instances is None:
        model = _load_model_from_db_(dataset_name)
        embedded_instances = _load_embeddings_from_db_(dataset_name)
    
    if model is None or embedded_instances is None:
        print('Cannor find any trained model or embedded instances for the dataset')
        return
    
    # 1. Transform instance to new metric space
    vars_features, conss_features, edge_indices, edge_values = get_milp_bipartite(instance_file)
    instance_data = MilpBipartiteData(
        var_feats=vars_features,
        cstr_feats=conss_features,
        edge_indices=edge_indices,
        edge_values=edge_values,
        force_cpu=True
    )
    embedding = model(instance_data).detach()

    # 2. Run knn
    transformer = KNeighborsTransformer(n_neighbors=n_neighbors, mode="distance", n_jobs=-1)
    transformer.fit(embedded_instances)
    distances, neighbors = transformer.kneighbors(embedding, return_distance=True)
    neighbors = embedded_instances[neighbors[0], :]
    neighbors = list(map(lambda n: n.tolist(), neighbors))

    # 3. Get all configs of the k nearest neighbors
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]
    distances = distances.flatten()
    configs, config_distances = [], []
    for index, neighbor in enumerate(neighbors):
        # We do this query one at a time (and not using Mongo $in operator to preserve order)
        r = dataset.find_one({"embedding": neighbor}, projection=["configs"])
        instance_configs = sorted(r["configs"], key=lambda c: c["cost"])
        configs.extend(instance_configs[:n_configs])
        config_distances.extend([distances[index]] * len(instance_configs[:n_configs]))

    # 4. Suggest `n_configs` configurations with lowest cost from all neighbors
    suggested = list(zip(configs, config_distances))
    suggested = sorted(suggested, key=lambda c: (c[1], c[0]["cost"]))
    suggested_configs, distances = zip(*suggested)

    return suggested_configs, distances
