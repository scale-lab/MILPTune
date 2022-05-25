import pathlib

from pymongo import ReturnDocument

from milptune.db.connections import get_client
from milptune.db.helpers import to_mongo_binary
from milptune.features.bipartite import get_milp_bipartite


def index_bipartite(instances_dir, dataset_name):
    instances_path = pathlib.Path(instances_dir)
    instances = list(instances_path.glob("*.mps.gz"))

    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]
    for instance in instances:
        vars_features, conss_features, edge_features_indices, edge_features_values = get_milp_bipartite(instance)
        r = dataset.find_one_and_update(
            {"path": str(instance)},
            {"$set": {"bipartite": {
                "vars_features": to_mongo_binary(vars_features),
                "conss_features": to_mongo_binary(conss_features),
                "edge_features": {
                    "indices" : to_mongo_binary(edge_features_indices),
                    "values" : to_mongo_binary(edge_features_values)}
                }}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if r:
            print(r["path"], r["_id"])