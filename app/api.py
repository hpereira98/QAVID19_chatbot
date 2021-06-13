from dotenv import load_dotenv, find_dotenv
import os
from fastapi import FastAPI
from app.services import elasticsearch
from app.services import bert

load_dotenv(find_dotenv())

ES_HOST = os.getenv("ES_HOST")
ES_PORT = os.getenv("ES_PORT")

# init fastapi
app = FastAPI()

# init elastic search object
es = elasticsearch.init(ES_HOST, ES_PORT)
# Load the fine-tuned model
tokenizer = bert.init_tokenizer("./models/bert/bbu_squad2")
model = bert.init_model("./models/bert/bbu_squad2")

@app.get("/es_status")
def es_status():
    try:
        es_is_running = elasticsearch.check_elasticsearch_running()
    except Exception as e:
        es_is_running = 400
    return {"es_status": es_is_running}


# @app.get("/documents/{idx}/{query}")
# def get_es_document(idx, query):
#     res = elasticsearch.search(es, idx, query)
#     return {"result": res}

@app.get("/chatbot/{question}")
def get_chatbot_answer(question):
    # context = elasticsearch.get_most_relevant_document(es, question)
    context = "12345"

    chunked, inputs = bert.tokenize(tokenizer, model, question, context)
    answer = bert.get_answer(tokenizer, model, chunked, inputs)

    return {"result": answer}