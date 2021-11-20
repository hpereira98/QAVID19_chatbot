import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import wikipedia as wiki
from collections import OrderedDict

tokenizer = AutoTokenizer.from_pretrained("app/models/tokenizers/bert_en", use_fast=False)
model = AutoModelForQuestionAnswering.from_pretrained("app/models/bert_en", return_dict=False)
pipe = pipeline('question-answering', model=model, tokenizer=tokenizer)


def convert_ids_to_string(tokenizer, input_ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))

def get_answer(context, question):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    print(f"This translates into {len(inputs['input_ids'][0])} tokens.")

    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    print(f"Question: {question}\nAnswer: {answer}")

def get_answer_with_chunks(context, question):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    # identify question tokens (token_type_ids = 0)
    qmask = inputs['token_type_ids'].lt(1)
    qt = torch.masked_select(inputs['input_ids'], qmask)
    print(f"The question consists of {qt.size()[0]} tokens.")

    chunk_size = model.config.max_position_embeddings - qt.size()[0] - 1 # the "-1" accounts for
    # having to add a [SEP] token to the end of each chunk
    print(f"Each chunk will contain {chunk_size - 2} tokens of the Wikipedia article.")

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

    for i in range(len(chunked_input.keys())):
        print(f"Number of tokens in chunk {i}: {len(chunked_input[i]['input_ids'].tolist()[0])}")

    answer = ''

    for _, chunk in chunked_input.items():
        answer_start_scores, answer_end_scores = model(**chunk)

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        ans = convert_ids_to_string(tokenizer, chunk['input_ids'][0][answer_start:answer_end])

        # if the ans == [CLS] then the model did not find a real answer in this chunk
        if ans != '[CLS]':
            answer += ans + " / "

    print(f"Question: {question}\nAnswer: {answer}")

def get_answer_with_pipeline(context, question):
    answer = pipe(
        {
            'question': question,
            'context': context
        }
    )
    print(f"Question: {question}\nAnswer: {answer}")


def test1():
    context = """Macedonia was an ancient kingdom on the periphery of Archaic and Classical Greece,
        and later the dominant state of Hellenistic Greece. The kingdom was founded and initially ruled
        by the Argead dynasty, followed by the Antipatrid and Antigonid dynasties. Home to the ancient
        Macedonians, it originated on the northeastern part of the Greek peninsula. Before the 4th
        century BC, it was a small kingdom outside of the area dominated by the city-states of Athens,
        Sparta and Thebes, and briefly subordinate to Achaemenid Persia."""

    question = "Who ruled Macedonia?"

    get_answer_with_pipeline(context, question)


def test2():
    question = 'What is the wingspan of an albatross?'

    results = wiki.search(question)
    page = wiki.page(results[0])
    context = page.content

    get_answer_with_pipeline(context, question)

def main():
    print("Running Test 1...")
    test1()

    print("Running Test 2...")
    test2()

if __name__ == "__main__":
    main()
