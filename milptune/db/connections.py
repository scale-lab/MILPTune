from os import environ

from pymongo import MongoClient


def get_client():
    CONNECTION_STRING = 'mongodb://{}:{}@{}:{}'.format(
        environ.get('MONGO_USERNAME'),
        environ.get('MONGO_PASSWORD'),
        environ.get('MONGO_HOST'),
        environ.get('MONGO_PORT'))
    return MongoClient(CONNECTION_STRING)
