import pickle

from bson.binary import Binary


def to_mongo_binary(obj):
    return Binary(pickle.dumps(obj, protocol=4))


def from_mongo_binary(obj):
    return pickle.loads(obj)
