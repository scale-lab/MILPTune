
from collections import Counter
import torch
from torch_geometric.data import Batch
from milptune.train.helpers.data import MilpBipartiteData
from milptune.db.connections import get_client
from milptune.db.helpers import from_mongo_binary
from milptune.train.helpers.gnn import MilpGNN, ConfigPerformanceRegressor
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from os.path import exists

from pyscipopt import Model as SCIPModel
from ecole.observation import MilpBipartite
from ecole.scip import Model
from torch_geometric.loader import DataLoader
import pathlib
from milptune.features.bipartite import get_milp_bipartite
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def train_set_size(dataset_name='1_item_placement', cost_threshold=1000):
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]

    r = dataset.count_documents({"bipartite": {"$exists": True}, "incumbent.0.cost": {"$lt": cost_threshold}})

    return r


def load_batch(dataset_name='1_item_placement', n_instances=100, start_index=0, cost_threshold=1000, load_local=True):
    # check locally
    instances_file = f'{dataset_name}.instances.{n_instances}.{start_index}.{cost_threshold}.pt'
    costs_file = f'{dataset_name}.costs.{n_instances}.{start_index}.{cost_threshold}.pt'
    if exists(instances_file) and exists(costs_file) and load_local:
        return torch.load(instances_file), torch.load(costs_file)

    print('loading from data store ..')
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]

    instances = []
    costs = []
    r = dataset.find(
        {"bipartite": {"$exists": True}, "incumbent.0.cost": {"$lt": cost_threshold}},
        sort=[("path", 1)], limit=n_instances, skip=start_index)
    for instance in r:
        vars_features = from_mongo_binary(instance["bipartite"]["vars_features"])
        conss_features = from_mongo_binary(instance["bipartite"]["conss_features"])
        edge_indices = from_mongo_binary(instance["bipartite"]["edge_features"]["indices"])
        edge_values = from_mongo_binary(instance["bipartite"]["edge_features"]["values"])
    
        # print(vars_features, conss_features, edge_indices, edge_values)
        instance_data = MilpBipartiteData(
            var_feats=vars_features,
            cstr_feats=conss_features,
            edge_indices=edge_indices,
            edge_values=edge_values,
        )
        instances.append(instance_data)
        costs.append(instance['incumbent'][0]['cost'])
    
    # bins = range(0, 100, 5)
    # costs = np.digitize(costs, bins)
    # print(costs)
    instances = Batch.from_data_list(instances)
    costs = torch.tensor(costs)
    torch.save(instances, instances_file)
    torch.save(costs, costs_file)
    return instances, costs



def plot_tsne(X, y, file_name, colormap=plt.cm.Blues):
    plt.figure(figsize=(9, 6))

    # clean the figure
    plt.clf()

    # tsne = TSNE(learning_rate='auto')
    # tsne = TSNE(learning_rate="auto", init="pca")
    # tsne = TSNE(learning_rate="auto", init="pca", perplexity=20)
    tsne = TSNE(perplexity=5)
    X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=colormap)

    plt.xticks(())
    plt.yticks(())
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(file_name)

if __name__ == '__main__':
    torch.manual_seed(31331)
    batch_size = 10000

    instances, costs = load_batch(n_instances=batch_size, start_index=0, cost_threshold=100)
    model = ConfigPerformanceRegressor(256, 4, 128).to(torch.device('cpu'))
    model.eval()
    embeddings_before = model(instances.to(torch.device('cpu')))
    plot_tsne(embeddings_before.cpu().detach().numpy(), costs.cpu().detach().numpy(), 'embeddings_before.pdf')
    
    instances, costs = load_batch(n_instances=batch_size, start_index=0, cost_threshold=100)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    mode = model.to(torch.device('cpu'))
    embeddings_after = model(instances.to(torch.device('cpu')))
    plot_tsne(embeddings_after.cpu().detach().numpy(), costs.cpu().detach().numpy(), 'embeddings_after.pdf')
    
    