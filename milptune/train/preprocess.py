import pathlib

from pymongo import ReturnDocument

from milptune.db.connections import get_client
from milptune.db.helpers import to_mongo_binary
from milptune.features.A import get_A, get_mapping


def index_vars_conss(instance, dataset_name):
    vars_index, conss_index = get_mapping(instance)

    client = get_client()
    db = client.milptunedb
    dataset = db["milptune_metadata"]

    r = dataset.find_one_and_update(
        {f"{dataset_name}": {"$exists": True}},
        {
            "$set": {
                f"{dataset_name}.vars_index": vars_index,
                f"{dataset_name}.conss_index": conss_index,
            }
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    print(r["_id"])


def index_A(instances_dir, dataset_name):
    instances_path = pathlib.Path(instances_dir)
    instances = list(instances_path.glob("*.mps.gz"))

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
        raise Exception(
            "Please, run `index_vars_conss() first to index the variable/constraint names"
        )

    vars_index, conss_index = (
        r[f"{dataset_name}"]["vars_index"],
        r[f"{dataset_name}"]["conss_index"],
    )
    print("Retrieved vars_index and conss_index")

    dataset = db[dataset_name]
    for instance in instances:
        A = get_A(instance, vars_index, conss_index)
        r = dataset.find_one_and_update(
            {"path": str(instance)},
            {"$set": {"A": to_mongo_binary(A)}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if r:
            print(r["_id"], r["path"])
