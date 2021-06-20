from elasticsearch import Elasticsearch
import requests

def init(host: str, port: int):
    es = Elasticsearch([{'host': host, 'port': port}])
    return es

def check_elasticsearch_running():
    res = requests.get('http://localhost:9200')
    return res.status_code


def search(es: Elasticsearch, idx: str, query: str):
    try:
        return es.search(index=idx, body={"query": {"match": {"text": query}}})
    except Exception as e:
        return []