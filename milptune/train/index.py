import torch
import numpy as np

from milptune.db.connections import get_client
from milptune.db.helpers import from_mongo_binary
from milptune.train.helpers.gnn import InstanceEmbeddor
from milptune.train.helpers.data import MilpBipartiteData


def index_embeddings(dataset_name):
    np.random.seed(10)
    client = get_client()
    db = client.milptunedb
    dataset = db["milptune_metadata"]

    r = dataset.find_one({f"{dataset_name}.model": {"$exists": True}})
    if not r:
        raise Exception("Cannot find trained model")
    model = InstanceEmbeddor(r[dataset_name]["embedding_dim"], r[dataset_name]["n_gnn_layers"], r[dataset_name]["gnn_hidden_dim"]).to(torch.device('cpu'))
    model.load_state_dict(torch.load(r[dataset_name]["model"], map_location=torch.device('cpu')))
    model.eval()

    dataset = db[dataset_name]
    r = dataset.find({"bipartite": {"$exists": True}, "configs": {"$exists": True}})
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
        embedding = model(instance_data).detach().flatten()

        r1 = dataset.update_one(instance, {"$set": {"embedding": embedding.tolist()}})
        print(instance["path"], r1.modified_count)


if __name__ == '__main__':
    index_embeddings('2_load_balancing')
