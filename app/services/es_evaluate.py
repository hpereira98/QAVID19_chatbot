import numpy as np
import pandas as pd
def average_precision(binary_results):

    ''' Calculates the average precision for a list of binary indicators '''

    m = 0
    precs = []

    for i, val in enumerate(binary_results):
        if val == 1:
            m += 1
            precs.append(sum(binary_results[:i+1])/(i+1))

    ap = (1/m)*np.sum(precs) if m else 0

    return ap


def evaluate_retriever(es_obj, index_name, qa_records, n_results):
    '''
    This function loops through a set of question/answer examples from SQuAD2.0 and
    evaluates Elasticsearch as a information retrieval tool in terms of recall, mAP, and query duration.

    Args:
        es_obj (elasticsearch.client.Elasticsearch) - Elasticsearch client object
        index_name (str) - name of index to query
        qa_records (list) - list of qa_records from preprocessing steps
        n_results (int) - the number of results ElasticSearch should return for a given query

    Returns:
        test_results_df (pd.DataFrame) - a dataframe recording search results info for every example in qa_records

    '''

    results = []

    for i, qa in enumerate(tqdm(qa_records)):

        ex_id = qa['example_id']
        question = qa['question_text']
        answer = qa['short_answer']

        # execute query
        res = search_es(es_obj=es_obj, index_name=index_name, question_text=question, n_results=n_results)

        # calculate performance metrics from query response info
        duration = res['took']
        binary_results = [int(answer.lower() in doc['_source']['document_text'].lower()) for doc in res['hits']['hits']]
        ans_in_res = int(any(binary_results))
        ap = average_precision(binary_results)

        rec = (ex_id, question, answer, duration, ans_in_res, ap)
        results.append(rec)

    # format results dataframe
    cols = ['example_id', 'question', 'answer', 'query_duration', 'answer_present', 'average_precision']
    results_df = pd.DataFrame(results, columns=cols)

    # format results dict
    metrics = {'Recall': results_df.answer_present.value_counts(normalize=True)[1],
               'Mean Average Precision': results_df.average_precision.mean(),
               'Average Query Duration':results_df.query_duration.mean()}

    return results_df, metrics
