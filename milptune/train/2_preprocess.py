import pathlib

from milptune.db.connections import get_client
from milptune.db.helpers import to_mongo_binary
from milptune.features.A import get_A


def index_A(instances_dir, dataset_name):
    instances_path = pathlib.Path(instances_dir)
    instances = list(instances_path.glob("/*.mps.gz"))

    dataset_name = instances_path.stem
    client = get_client()
    db = client.milptunedb
    dataset = db["milptune_metadata"]

    # primer to just focus on common matrix coeficients
    r = dataset.find_one(
        {
            f"{dataset_name}.vars_index": {"$exists": True},
            f"{dataset_name}.conss_index": {"$exists": True},
        }
    )
    if not r:
        raise Exception("Please, run `preprocess_1.py first to index the variable/constraint names")

    vars_index, conss_index = (
        r[f"{dataset_name}"]["vars_index"],
        r[f"{dataset_name}"]["conss_index"],
    )
    print("Retrieved vars_index and conss_index")

    dataset = db[dataset_name]
    for instance in instances:
        A = get_A(instance, vars_index, conss_index)
        r = dataset.find_one_and_update(
            {"path": str(instance)}, {"$set": {"A": to_mongo_binary(A)}}, upsert=True
        )
        if r:
            print(r["_id"], r["path"])
