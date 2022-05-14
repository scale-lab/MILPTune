import json
import os
import pathlib
from smac.runhistory.runhistory import RunHistory
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter

if __name__ == '__main__':
    data_path = pathlib.Path('anonymous_0506')
    workers_paths = list(data_path.glob('run_*'))
    
    
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
    
    workers = {
    }
    for worker_path in workers_paths:
        rh_path = worker_path.joinpath('runhistory.json')
        runhistory = RunHistory()
        cs = ConfigurationSpace()
        cs.add_hyperparameters(params)
        runhistory.load_json(rh_path, cs)

        incumbent = {
            'config': '',
            'cost': float('inf'),
            'instances': []
        }
        for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in runhistory.data.items():
            if cost < incumbent['cost']:
                incumbent['cost'] = cost
                incumbent['config'] = runhistory.ids_config[config_id]
                incumbent['instance'] = instance_id
            incumbent['instances'].append(instance_id)
        workers[str(worker_path)] = incumbent

    incumbent_of_incumbents = {
        'cost': float('inf'),
        'config': ''
    }
    
    for worker_id, worker in workers.items():
        if worker['cost'] < incumbent_of_incumbents['cost']:
            incumbent_of_incumbents['cost'] = worker['cost']
            incumbent_of_incumbents['config'] = worker['config']._values
            incumbent_of_incumbents['instance'] = worker['instance']
    
    with open('incumbent.json', 'w') as f:
        json.dump(incumbent_of_incumbents, f)
