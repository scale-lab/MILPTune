
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
from milptune.viz.tsne import plot_tsne

from pyscipopt import Model as SCIPModel
from ecole.observation import MilpBipartite
from ecole.scip import Model
from torch_geometric.loader import DataLoader
import pathlib
from milptune.features.bipartite import get_milp_bipartite
import numpy as np


def train_set_size(dataset_name='1_item_placement', cost_threshold=1000):
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]

    r = dataset.count_documents({"bipartite": {"$exists": True}, "incumbent.0.cost": {"$lt": cost_threshold}})

    return r


def load_batch(dataset_name='1_item_placement', n_instances=100, start_index=0, cost_threshold=100):
    # check locally
    instances_file = f'{dataset_name}.instances.{n_instances}.{start_index}.{cost_threshold}.pt'
    costs_file = f'{dataset_name}.costs.{n_instances}.{start_index}.{cost_threshold}.pt'
    if exists(instances_file) and exists(costs_file):
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
    bins = range(0, 100, 5)
    costs = np.digitize(costs, bins)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instances = Batch.from_data_list(instances).to(device)
    costs = torch.tensor(costs, device=device)
    torch.save(instances, instances_file)
    torch.save(costs, costs_file)
    return instances, costs


if __name__ == '__main__':
    torch.manual_seed(31331)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    num_epochs = 100

    model = ConfigPerformanceRegressor(256, 4, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    distance = distances.LpDistance(normalize_embeddings=False)
    # distance = distances.CosineSimilarity()
    # reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.1, distance=distance)
    mining_func = miners.TripletMarginMiner(
        margin=0.4, distance=distance, type_of_triplets="hard"
    )
    # accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    
    # size = train_set_size()
    size = 2599


    f = open('train.log', 'w')
    for epoch in range(1, num_epochs + 1):
        model.train()
        for i in range(int(size / batch_size)):
            start_index = i * batch_size
            instances, costs = load_batch(n_instances=batch_size, start_index=start_index)
            c = Counter(costs.tolist())
            # print(c)
            embeddings = model(instances.to(device))
            indices_tuple = mining_func(embeddings, costs)
            
            # print(embeddings, embeddings.shape)
            # exit()
            # mat = distance(embeddings, embeddings)
            # print(mat, mat.shape)
            # ap_dists = mat[indices_tuple[0], indices_tuple[1]]
            # an_dists = mat[indices_tuple[0], indices_tuple[2]]
            # print(ap_dists, ap_dists.shape)
            # print(an_dists, an_dists.shape)
            # exit()
            loss = loss_func(embeddings, costs, indices_tuple)
            loss.backward()
            optimizer.step()
            log_line = "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, i, loss, mining_func.num_triplets
                )
            print(log_line)
            f.write(log_line + "\n")
            f.flush()
            if epoch == 50:
                loss_func = losses.TripletMarginLoss(margin=0.1, distance=distance)
                print('changing mining function .. ')
                mining_func = miners.TripletMarginMiner(
                    margin=0.2, distance=distance, type_of_triplets="semihard"
                )       
    torch.save(model.state_dict(), 'model.pt')
    f.close()
