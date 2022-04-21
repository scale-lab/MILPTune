import pathlib

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.configspace import ConfigurationSpace
from smac.runhistory.runhistory import RunHistory

from milptune.db.connections import get_client

if __name__ == '__main__':
    data_path = pathlib.Path('.')
    workers_paths = list(data_path.glob('smac3*/run_*'))
    
    params = [
        # Branching
        CategoricalHyperparameter('branching/scorefunc', choices=['s', 'p', 'q'], default_value='p'),
        UniformFloatHyperparameter('branching/scorefac', 0.0, 1.0, default_value=0.167),
        CategoricalHyperparameter('branching/preferbinary', choices=[True, False], default_value=False),
        UniformFloatHyperparameter('branching/clamp', 0.0, 0.5, default_value=0.2),
        UniformFloatHyperparameter('branching/midpull', 0.0, 1.0, default_value=0.75),
        UniformFloatHyperparameter('branching/midpullreldomtrig', 0.0, 1.0, default_value=0.5),
        CategoricalHyperparameter('branching/lpgainnormalize', choices=['d', 'l', 's'], default_value='s'),
        # LP
        CategoricalHyperparameter('lp/pricing', choices=['l','a','f','p','s','q','d'], default_value='l'),
        UniformIntegerHyperparameter('lp/colagelimit', -1, 2147483647, default_value=10),
        UniformIntegerHyperparameter('lp/rowagelimit', -1, 2147483647, default_value=10),
        # Node Selection
        CategoricalHyperparameter('nodeselection/childsel', choices=["d",'u','p','i','l','r','h'], default_value='h'),
        # Separating
        UniformFloatHyperparameter('separating/minortho', 0.0, 1.0, default_value=0.9),
        UniformFloatHyperparameter('separating/minorthoroot', 0.0, 1.0, default_value=0.9),
        UniformIntegerHyperparameter('separating/maxcuts', 0, 2147483647, default_value=100),
        UniformIntegerHyperparameter('separating/maxcutsroot', 0, 2147483647, default_value=2000),
        UniformIntegerHyperparameter('separating/cutagelimit', -1, 2147483647, default_value=80),
        UniformIntegerHyperparameter('separating/poolfreq', -1, 65534, default_value=10)
    ]
    

    for worker_path in workers_paths:
        rh_path = worker_path.joinpath('runhistory.json')
        runhistory = RunHistory()
        cs = ConfigurationSpace()
        cs.add_hyperparameters(params)
        runhistory.load_json(rh_path, cs)

        client = get_client()
        db = client.milptunedb
        
        print(worker_path)
        for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in runhistory.data.items():
            instance_id = instance_id.replace('//', '/')
            dataset = db[instance_id.split('/')[-3]]
            r = dataset.find_one({'path': instance_id})
            config = {
                'seed': seed,
                'cost': cost,
                'time': time,
                'params': runhistory.ids_config[config_id]._values
            }
            update_result = dataset.update_one(r, {'$push': {'configs': config}})
            print(instance_id, update_result.modified_count)
        print('----------------')
