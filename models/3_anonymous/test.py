
import torch
from torch_geometric.data import Batch
from milptune.train.helpers.data import MilpBipartiteData
from milptune.db.connections import get_client
from milptune.db.helpers import from_mongo_binary
from milptune.train.helpers.gnn import InstanceEmbeddor
from os.path import exists

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_batch(dataset_name='3_anonymous', n_instances=100, start_index=0, cost_threshold=20000, load_local=True):
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
        {"bipartite": {"$exists": True}, "default_config.0.cost": {"$lt": cost_threshold}},
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
        costs.append(instance['default_config'][0]['cost'])

    instances = Batch.from_data_list(instances)
    costs = torch.tensor(costs)
    torch.save(instances, instances_file)
    torch.save(costs, costs_file)
    return instances, costs



def plot_tsne(X, y, file_name, colormap=plt.cm.GnBu):
    plt.figure(figsize=(9, 6))

    # clean the figure
    plt.clf()

    tsne = TSNE(perplexity=5)
    X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=colormap)

    plt.xticks(())
    plt.yticks(())
    cbar = plt.colorbar(pad=0.01)
    cbar.ax.tick_params(labelsize=18)
    cbar.formatter.set_powerlimits((0, 0))
    plt.tight_layout()
    plt.savefig(file_name)

if __name__ == '__main__':
    torch.manual_seed(31331)
    batch_size = 10000

    instances, costs = load_batch(n_instances=batch_size, start_index=0, cost_threshold=20000)
    model = InstanceEmbeddor(32, 4, 16).to(torch.device('cpu'))
    model.eval()
    embeddings_before = model(instances.to(torch.device('cpu')))
    plot_tsne(embeddings_before.cpu().detach().numpy(), costs.cpu().detach().numpy(), 'embeddings_before.pdf')
    
    instances, costs = load_batch(n_instances=batch_size, start_index=0, cost_threshold=20000)
    model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
    model.eval()
    mode = model.to(torch.device('cpu'))
    embeddings_after = model(instances.to(torch.device('cpu')))
    plot_tsne(embeddings_after.cpu().detach().numpy(), costs.cpu().detach().numpy(), 'embeddings_after.pdf')
    
    