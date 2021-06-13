from dotenv import load_dotenv, find_dotenv
import os
from fastapi import FastAPI
from app.services import elasticsearch

load_dotenv(find_dotenv())

ES_HOST = os.getenv("ES_HOST")
ES_PORT = os.getenv("ES_PORT")

# init fastapi
app = FastAPI()

# init elastic search object
es = elasticsearch.init(ES_HOST, ES_PORT)

@app.get("/es_status")
def read_root():
    try:
        es_is_running = elasticsearch.check_elasticsearch_running()
    except Exception as e:
        es_is_running = 400
    return {"es_status": es_is_running}


@app.get("/documents/{idx}/{query}")
def get_es_document(idx, query):
    res = elasticsearch.search(es, idx, query)
    return {"result": res}

