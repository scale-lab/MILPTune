import numpy as np

from pyscipopt import Model as SCIPModel
from ecole.observation import MilpBipartite
from ecole.scip import Model


def get_milp_bipartite(instance):
    model = SCIPModel()
    model.readProblem(instance)
    model = Model.from_pyscipopt(model)

    features_extractor = MilpBipartite()
    features = features_extractor.extract(model, False)
    return features.variable_features.astype(np.float32), features.constraint_features.astype(np.float32), features.edge_features.indices.astype(np.uint32), features.edge_features.values.astype(np.float32)
