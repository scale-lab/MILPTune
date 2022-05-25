
import torch
from torch_geometric.data import Batch
from milptune.train.helpers.data import MilpBipartiteData
from milptune.db.connections import get_client
from milptune.db.helpers import from_mongo_binary
from milptune.train.helpers.gnn import InstanceEmbeddor
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def train_set_size(dataset_name='2_load_balancing', cost_threshold=1000):
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]

    r = dataset.count_documents({"bipartite": {"$exists": True}, "incumbent.0.cost": {"$lt": cost_threshold}})

    return r


def load_batch(dataset_name='2_load_balancing', n_instances=100, start_index=0, cost_threshold=1000, load_local=True):
    # check locally
    instances_file = f'{dataset_name}.instances.{n_instances}.{start_index}.{cost_threshold}.pt'
    costs_file = f'{dataset_name}.costs.{n_instances}.{start_index}.{cost_threshold}.pt'
    print(instances_file)
    if exists(instances_file) and exists(costs_file) and load_local:
        return torch.load(instances_file, 'cpu'), torch.load(costs_file, 'cpu')

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
            edge_values=edge_values
        )
        instances.append(instance_data)
        costs.append(instance['incumbent'][0]['cost'])

    instances = Batch.from_data_list(instances)
    costs = torch.tensor(costs)
    torch.save(instances, instances_file)
    torch.save(costs, costs_file)
    return instances, costs



def plot_tsne(X, y, file_name, colormap=plt.cm.GnBu):
    plt.figure(figsize=(9, 6))

    # clean the figure
    plt.clf()

    tsne = TSNE(perplexity=10)
    X_embedded = tsne.fit_transform(X)
    cond = y > 400
    plt.scatter(X_embedded[cond, 0], X_embedded[cond, 1], c=y[cond], cmap=colormap)

    plt.xticks(())
    plt.yticks(())
    cbar = plt.colorbar(pad=0.01)
    cbar.ax.tick_params(labelsize=18)
    cbar.formatter.set_powerlimits((0, 0))
    plt.tight_layout()
    plt.savefig(file_name)

if __name__ == '__main__':
    torch.manual_seed(31331)
    model = InstanceEmbeddor(128, 2, 64).to(torch.device('cuda'))
    batch_size = 64
    
    all_embeddings = []
    all_costs = []

    for i in range(int(1727 / batch_size) + 1):
        instances, costs = load_batch(n_instances=batch_size, start_index=i * batch_size, cost_threshold=1000)
        all_costs.extend(costs.cpu().tolist())
        model.eval()
        embeddings_before = model(instances.to(torch.device('cuda')))
        all_embeddings.append(embeddings_before.cpu().detach())
        
        del instances
        del costs
        del embeddings_before
        torch.cuda.empty_cache()
    
    all_embeddings = torch.cat(all_embeddings)
    print(all_embeddings.shape)
    
    plot_tsne(all_embeddings.cpu().detach().numpy(), np.array(all_costs), 'embeddings_before.pdf')

    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    for i in range(int(1727 / batch_size) + 1):
        instances, costs = load_batch(n_instances=batch_size, start_index=i * batch_size, cost_threshold=1000)
        all_costs.extend(costs.cpu().tolist())
        model.eval()
        embeddings_after = model(instances.to(torch.device('cuda')))
        all_embeddings.append(embeddings_after.cpu().detach())
        
        del instances
        del costs
        del embeddings_after
        torch.cuda.empty_cache()
    
    all_embeddings = torch.cat(all_embeddings)
    plot_tsne(all_embeddings.cpu().detach().numpy(), np.array(all_costs), 'embeddings_after.pdf')
    
    