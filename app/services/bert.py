import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer, BertForQuestionAnswering
from collections import OrderedDict

def init_tokenizer(model):
    tokenizer = BertTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def init_model(model):
    model = BertForQuestionAnswering.from_pretrained(model, return_dict=False)
    return model

def init_pipeline(model, tokenizer):
    pipe = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return pipe

def tokenize(tokenizer, model, question, text):
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    chunked = False

    if len(input_ids) > model.config.max_position_embeddings:
        inputs = chunkify(inputs, model)
        chunked = True

    return chunked, inputs

def chunkify(inputs, model):
    # create question mask based on token_type_ids
    # value is 0 for question tokens, 1 for context tokens
    qmask = inputs['token_type_ids'].lt(1)
    qt = torch.masked_select(inputs['input_ids'], qmask)
    chunk_size = model.config.max_position_embeddings - qt.size()[0] - 1
    # the "-1" accounts for having to add an ending [SEP] token to the end

    # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
    chunked_input = OrderedDict()
    for k,v in inputs.items():
        q = torch.masked_select(v, qmask)
        c = torch.masked_select(v, ~qmask)
        chunks = torch.split(c, chunk_size)

        for i, chunk in enumerate(chunks):
            if i not in chunked_input:
                chunked_input[i] = {}

            thing = torch.cat((q, chunk))
            if i != len(chunks)-1:
                if k == 'input_ids':
                    thing = torch.cat((thing, torch.tensor([102])))
                else:
                    thing = torch.cat((thing, torch.tensor([1])))

            chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
    return chunked_input


def convert_ids_to_string(tokenizer, input_ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))


def get_answer(tokenizer, model, chunked, inputs):
    if chunked:
        answer = ''
        for k, chunk in inputs.items():
            answer_start_scores, answer_end_scores = model(**chunk)

            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            ans = convert_ids_to_string(tokenizer, chunk['input_ids'][0][answer_start:answer_end])
            if ans != '[CLS]':
                answer += ans + " / "
        return answer
    else:
        answer_start_scores, answer_end_scores = model(**inputs)

        answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score

        return convert_ids_to_string(tokenizer, inputs['input_ids'][0][answer_start:answer_end])

def get_answer_with_pipeline(pipe, context, question):
    answer = pipe(
        {
            'question': question,
            'context': context
        }
    )
    return answer
