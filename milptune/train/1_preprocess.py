from milptune.db.connections import get_client
from milptune.features.A import get_mapping


def index_vars_conss(instance, dataset_name):
    vars_index, conss_index = get_mapping(instance)

    client = get_client()
    db = client.milptunedb
    dataset = db['milptune_metadata']

    r = dataset.find_one_and_update(
        {
            f'{dataset_name}': {'$exists': True}
        },
        {
            '$set': {
                f'{dataset_name}.vars_index': vars_index,
                f'{dataset_name}.conss_index': conss_index
            }
        },
        upsert=True)
    print(r['_id'])
