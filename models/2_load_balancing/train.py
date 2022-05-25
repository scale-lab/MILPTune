
from collections import Counter
import torch
from torch_geometric.data import Batch
from milptune.train.helpers.data import MilpBipartiteData
from milptune.db.connections import get_client
from milptune.db.helpers import from_mongo_binary
from milptune.train.helpers.gnn import InstanceEmbeddor
from pytorch_metric_learning import distances, losses, miners
from os.path import exists

import numpy as np


def train_set_size(dataset_name='2_load_balancing', cost_threshold=1000):
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]

    r = dataset.count_documents({"bipartite": {"$exists": True}, "incumbent.0.cost": {"$lt": cost_threshold}}) 

    return r


def load_batch(dataset_name='2_load_balancing', n_instances=100, start_index=0, cost_threshold=1000):
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

        instance_data = MilpBipartiteData(
            var_feats=vars_features,
            cstr_feats=conss_features,
            edge_indices=edge_indices,
            edge_values=edge_values,
        )
        instances.append(instance_data)
        costs.append(instance['incumbent'][0]['cost'])
    bins = range(0, 1000, 10)
    costs = np.digitize(costs, bins)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instances = Batch.from_data_list(instances).to(device)
    costs = torch.tensor(costs, device=device)
    torch.save(instances, instances_file)
    torch.save(costs, costs_file)
    return instances, costs


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.manual_seed(31331)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_epochs = 20

    model = InstanceEmbeddor(128, 2, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    distance = distances.LpDistance(normalize_embeddings=False)
    loss_func = losses.TripletMarginLoss(margin=0.1, distance=distance)
    mining_func = miners.TripletMarginMiner(
        margin=0.4, distance=distance, type_of_triplets="hard"
    )
    
    # size = train_set_size()
    size = 1727

    f = open('train.log', 'w')
    for epoch in range(1, num_epochs + 1):
        model.train()
        for i in range(int(size / batch_size)+1):
            start_index = i * batch_size
            instances, costs = load_batch(n_instances=batch_size, start_index=start_index)
            c = Counter(costs.tolist())
            embeddings = model(instances.to(device))
            indices_tuple = mining_func(embeddings, costs)
            loss = loss_func(embeddings, costs, indices_tuple)
            loss.backward()
            optimizer.step()
            log_line = "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}\t{}".format(
                    epoch, i, loss, mining_func.num_triplets, c
                )
            print(log_line)
            f.write(log_line + "\n")
            f.flush()
            if epoch == 10:
                print('changing mining function .. ')
                mining_func = miners.TripletMarginMiner(
                    margin=0.2, distance=distance, type_of_triplets="semihard"
                )
            torch.save(model.state_dict(), f'model-{epoch}.pt')       
    torch.save(model.state_dict(), 'model.pt')
    f.close()
