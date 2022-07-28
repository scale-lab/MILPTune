import numpy as np

from pyscipopt import Model as SCIPModel
from collections import OrderedDict


def get_milp_shallow(instance):
    model = SCIPModel()
    model.readProblem(instance)

    # Preprocessing to ensure that we parse the problem in the same order
    # for all problem instances since shallow embedding is not invariant
    # to the problem definition
    vars = list(map(lambda v: str(v), model.getVars()))
    _mapping_coeffs = OrderedDict.fromkeys(vars, 0)
    for var, coeff in model.getObjective().terms.items():
        _mapping_coeffs[str(var.vartuple[0])] = coeff
    _vars_index = OrderedDict.fromkeys(vars, 0)
    for i, var in enumerate(model.getVars()):
        _vars_index[str(var)] = i
    
    # Problem Size
    problem_size = np.array([len(model.getVars()), len(model.getConss())], dtype=np.float32)

    # Propotion of var types
    var_types = np.array([0., 0.]) # (discrete, contiuous)
    for var in model.getVars():
        if var.vtype() in ['BINARY', 'INTEGER', 'IMPLINT']:
            var_types[0] +=1
        else:
            var_types[1] += 1
    var_types /= np.linalg.norm(var_types) # proportions

    # Constraint types
    for cons in model.getConss():
        pass    # all constraints are linear in this dataset
    
    # A matrix, and b
    A = np.zeros((len(model.getConss()), len(model.getVars())))
    b = np.zeros((len(model.getConss()), 1))
    for i, cons in enumerate(model.getConss()):
        lhs = model.getValsLinear(cons)
        rhs = model.getRhs(cons)
        mask = list(map(lambda k: _vars_index[k], lhs.keys()))
        A[i][mask] = np.array(list(lhs.values()))          
        
        # https://www.scipopt.org/doc/html/cons__linear_8c_source.php#l01192
        direction = {
            '==': model.isEQ(model.getRhs(cons), model.getLhs(cons)),
            '<=': not model.isInfinity(model.getRhs(cons)),
            '>=': not model.isInfinity(-model.getLhs(cons))
        }
        bb = {
            '==': model.getRhs(cons),
            '<=': model.getRhs(cons),
            '>=': model.getLhs(cons)
        }
        d = [k for k, v in direction.items() if v][0]
        b[i][0] = bb[d]
    A /= np.linalg.norm(A)
    b /= np.linalg.norm(b)

    # Normalized Cost Vector
    cost_vector = np.zeros(len(model.getVars()))
    for i, (_, coeff) in enumerate(_mapping_coeffs.items()):
        cost_vector[i] = coeff
    cost_vector /= np.linalg.norm(cost_vector) # normalize

    return np.concatenate((problem_size, var_types, A.flatten(), b.flatten(), cost_vector))
