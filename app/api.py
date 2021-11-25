import json
from dotenv import load_dotenv, find_dotenv
import os
from fastapi import FastAPI, Request, HTTPException
from app.services import elasticsearch
from app.services import bert
import functools
from datetime import datetime

load_dotenv(find_dotenv())

ES_HOST = os.getenv("ES_HOST")
ES_PORT = os.getenv("ES_PORT")

# init fastapi
app = FastAPI()

# init elastic search object
es = elasticsearch.init(ES_HOST, ES_PORT)
# Load the fine-tuned models
# en
tokenizer_en = bert.init_tokenizer("app/models/tokenizers/bert_en")
model_en = bert.init_model("app/models/bert_en")
pipe_en = bert.init_pipeline(model_en, tokenizer_en)
# pt
tokenizer_pt = bert.init_tokenizer("app/models/tokenizers/bert_pt")
model_pt = bert.init_model("app/models/bert_pt")
pipe_pt = bert.init_pipeline(model_pt, tokenizer_pt)


model_switcher = {
    'pt': model_pt,
    'en': model_en
}

tokenizer_switcher = {
    'pt': tokenizer_pt,
    'en': tokenizer_en
}

pipe_switcher = {
    'pt': pipe_pt,
    'en': pipe_en
}

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

@app.post("/ask_question")
async def get_chatbot_answer(request: Request):
    start_time = datetime.now()
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Request body not found: {e}")

    try:
        lang = body["lang"]
        question = body["question"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Missing parameter: {e}")

    num_pages = 3

    if 'version' in body:
        if body['version']:
            idx = f"covid_{lang}_{body['version']}"
            if body['version'] == 'v3':
                num_pages = 1
    else:
        idx = f"covid_{lang}"

    print(f"Searching on index: {idx}")
    es_res = elasticsearch.search(es, idx, question, num_pages)
    print(f"ElasticSearch Fetch Time: {datetime.now() - start_time}")

    if not es_res:
        raise HTTPException(status_code=404, detail=f"No relevant documents found.")

    with open('file.txt', 'w') as file:
     file.write(json.dumps(es_res))

    found_documents = es_res['hits']['hits']

    answers = []

    for idx, doc in enumerate(found_documents):
        doc_score = doc['_score']
        doc_url = doc['_source']['url']
        doc_text = doc['_source']['text']

        bert_start_time = datetime.now()

        answer = bert.get_answer_with_pipeline(pipe_switcher.get(lang, pipe_en), doc_text, question)

        exec_time = (datetime.now() - bert_start_time).total_seconds()

        # chunked, inputs = bert.tokenize(tokenizer_switcher.get(lang, tokenizer_en), model_switcher.get(lang, model_en), question, doc_text)
        # answer = bert.get_answer(tokenizer_switcher.get(lang, tokenizer_en), model_switcher.get(lang, model_en), chunked, inputs)

        print(f"url: {doc_url}, answer: {answer}")
        print(f"BERT Execution Time (s): {exec_time}")
        print(f"ES Score: {doc_score}")
        answer['url'] = doc_url
        answer['exec_time'] = exec_time
        answer['es_score'] = doc_score
        answers.append(answer)

    if answers:
        best_answer = functools.reduce(lambda a, b: a if a['score'] > b['score'] else b, answers)

        print(f"Total Answering Time: {datetime.now() - start_time}")
        return {"result": best_answer}

    raise HTTPException(status_code=404, detail=f"Couldn't generate answer for that question.")
