from elasticsearch import Elasticsearch
import requests

def init(host: str, port: int):
    #es = Elasticsearch([{'host': host, 'port': port}])
    es = Elasticsearch()
    return es

def check_elasticsearch_running():
    res = requests.get('http://localhost:9200')
    return res.status_code


def search(es: Elasticsearch, idx: str, query: str, n_results: int = 10):
    try:
        print("Searching...")
        return es.search(index=idx, body={"query": {"match": {"text": query}}}, size=n_results)
    except Exception as e:
        print(e)
        return []
