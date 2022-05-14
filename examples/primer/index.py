from milptune.db.connections import get_client



if __name__ == '__main__':
    client = get_client()
    db = client.milptunedb
    dataset = db['3_anonymous']

    with open('runs_incumbent.csv', 'r') as f:
        for line in f:
            cost, instance = line.strip().split(',')
            cost = float(cost.strip())
            instance = instance.strip()
            r = dataset.find_one_and_update(
                {"path": instance},
                {"$push": {"incumbent": {'cost': cost}}})

            print(instance, cost, r['_id'])