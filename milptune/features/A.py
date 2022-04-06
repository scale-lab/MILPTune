import numpy as np
import pandas as pd
from pyscipopt import Model
from collections import deque
from scipy.sparse import coo_matrix


def get_mapping(instance_path: str):
    model = Model()
    model.readProblem(instance_path)
    vars = model.getVars()    
    conss = model.getConss()

    vars_indices = {str(vars[i]): i for i in range(len(vars))}
    conss_indices = {str(conss[i]): i for i in range(len(conss))}

    return vars_indices, conss_indices


def get_A(instance_path: str, vars_index, conss_index) -> np.array:
    model = Model()
    model.readProblem(instance_path)
    conss = model.getConss()

    conss = filter(lambda c: str(c) in conss_index, conss)

    row, col, data = [], [], []
    def breakdown(cons):
        coeffs = model.getValsLinear(cons)
        coeffs = dict(filter(lambda v: str(v[0]) in vars_index, coeffs.items()))
        row.extend([conss_index[str(cons)]] * len(coeffs))
        col.extend(map(lambda k: vars_index[str(k)], coeffs.keys()))
        data.extend(coeffs.values())

    deque(map(breakdown, conss))
    A = coo_matrix((data, (row, col)), shape=(len(conss_index), len(vars_index)))

    return A
