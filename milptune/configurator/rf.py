import joblib
import json
import random
import numpy as np
from milptune.db.connections import get_client
from milptune.features.shallow import get_milp_shallow
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

def load_dataset(dataset_name):
    client = get_client()
    db = client.milptunedb
    dataset = db[dataset_name]

    train = []
    M, P, C = [], [], []
    r = dataset.find({'configs': {'$exists': True}}, projection=['configs', 'path', 'shallow'])
    for index, instance in enumerate(r):
        best_config = sorted(instance['configs'], key=lambda c: c['cost'])[0]
        if best_config['cost'] < 100000:
            train.append((instance['shallow'], best_config['cost'], best_config))
    
    for index, instance in enumerate(train):
        M.append(instance[0])
        
        label = [100000] * len(train)
        label[index] = instance[1]
        P.append(label)

        C.append(instance[2])
    return M, P, C


def DF(M, P, dataset_name='3_anonymous'):
    n = len(M)
    k = len(M[0])
    m = len(P[0])
    
    dfs = {}
    
    for i, j in list(combinations(range(m), r=2)):
        labels = []
        weights = []
        for inst_index in range(n):
            if P[inst_index][i] < P[inst_index][j]:
                labels.append(i)
            elif P[inst_index][i] > P[inst_index][j]:
                labels.append(j)
            else:
                # tie: random break
                labels.append(random.choice([i, j]))
            
            # add weights to the samples
            weights.append(abs(P[inst_index][i] - P[inst_index][j]))
        
        # Now we have M and labels (in addition to sample weights)
        i_j_df = RandomForestClassifier()
        i_j_df.fit(M, labels, sample_weight=weights)    # pass weights to the fitting function

        joblib.dump(i_j_df, f'models-rf/{dataset_name}.{i}-{j}.rf')
 

def get_configuration_parameters(instance_file: str, dataset_name: str = '3_anonymous', m=53):
    # Extract feature vector
    embedding = get_milp_shallow(instance_file).reshape(1, -1)

    # load the model from disk
    predictions = []
    for i, j in list(combinations(range(m), r=2)):
        i_j_df = joblib.load(f'models-rf/{dataset_name}.{i}-{j}.rf')
        predictions.extend(i_j_df.predict(embedding))
    
    # Voting
    c = Counter(predictions)
    pred = c.most_common(1)[0][0]

    # Retrieving config
    C = joblib.load(f'models-rf/{dataset_name}.C.arr')

    return C[pred]
