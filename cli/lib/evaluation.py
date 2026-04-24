import argparse
import json
from lib.search_utils import root_file
from lib.hybrid_search import rrfsearch

def test_cases():
    with open(root_file/'data'/'golden_dataset.json','r') as f:
        return json.load(f)['test_cases']

def evaluate(limit):
    test_cases_data = test_cases()
    for test in test_cases_data:
        query = test['query']
        k = 60
        results = rrfsearch(query, k, limit,print_output=False)
        return_results = []
        for res in results:
            return_results.append(res['title'])

        relevant_docs = test['relevant_docs']
        Matched_results = [doc for doc in return_results if doc in relevant_docs]
        relevant_count = len(Matched_results)
        precision = relevant_count / len(return_results) if return_results else 0
        recall = relevant_count / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f'k = {limit}\n'
              f'-Query: {query}\n'
              f'Precision@{limit}: {precision:.4f}\n'
              f'Recall@{limit}: {recall:.4f}\n'
              f'F1 Score: {f1:.4f}\n'
              f'Retrieved: {return_results}\n'
              f'Relevant Documents: {relevant_docs}\n'  
               
        )

