import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from lib.search_utils import load_movies
from lib.llm import correct_spell, rewrite_query, expand_query ,evaluate_results
from lib.rerank import Rerank, batch_rerank
import logging
import json

class HybridSearch:
    def __init__(self,documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
       
        if not os.path.exists(self.idx.indexpath):
            self.idx.build()
            self.idx.save()
            
    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit*500) #{1:9,2:8,3:7,4:6,5:5}  doc_id : score
        semanticresults = self.semantic_search.search_chunks(query, limit*500) 
        bm25_results = normalize_dict(bm25_results)
        semantic_results = {}

        for _ in semanticresults:
            semantic_results[_['id']] = float(_['score'])

        semantic_results = normalize_dict(semantic_results)
        documents = self.idx.docmap
        combined = {}
        for doc_id in documents.keys():
            combined[doc_id] = {
            'bm25_score': bm25_results.get(doc_id, 0),
            'semantic_score': semantic_results.get(doc_id, 0),
            'hybrid_score': hybrid_score(bm25_results.get(doc_id, 0), semantic_results.get(doc_id, 0), alpha),
            'document': documents[doc_id]['description'],
            'title': documents[doc_id]['title'],
            'doc_id': doc_id
    }
        combined = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)[:limit]
        return combined

        
      # rrf search uses ranks instead of scores and K is a parameter that determines how much influence the rank has on the final score. higher K 
      # values will give more weight to higher-ranked results, while lower K values will make the scores more similar regardless of rank.  


    def rrf_search(self, query, k, limit):
        bm25_results = self._bm25_search(query, limit*500) #{1:9,2:8,3:7,4:6,5:5}  doc_id : score
        semanticresults = self.semantic_search.search_chunks(query, limit*500) 
        bm25_results = rank_dict(bm25_results,k)
        semantic_results = {}

        for _ in semanticresults:
            semantic_results[_['id']] = float(_['score'])

        semantic_results = rank_dict(semantic_results,k)
        documents = self.idx.docmap
        common_doc_ids = set(bm25_results.keys()) & set(semantic_results.keys())
        combined = {}
        for doc_id in common_doc_ids:
            combined[doc_id] = {
            'bm25_score': reverse_rank(bm25_results.get(doc_id, 0), k),
            'semantic_score': reverse_rank(semantic_results.get(doc_id, 0), k),
            'rrf_score': bm25_results.get(doc_id, 0) + semantic_results.get(doc_id, 0),
            'document': documents[doc_id]['description'],
            'title': documents[doc_id]['title'],
            'doc_id': doc_id,
            'bm25_rank': bm25_results.get(doc_id, 0),
            'semantic_rank': semantic_results.get(doc_id, 0),
            
    }
        combined = sorted(combined.values(), key=lambda x: x['rrf_score'], reverse=True)[:limit]
        return combined
        

       
    
# Normalization is really important and not doing it can cuz a lot of bad results. 
# For example, if the BM25 search returns a score of 9 and the semantic search returns a score of 0.8, 
# we need to normalize these scores to a common scale before combining them. Otherwise, the BM25 search results 
# will dominate the final ranking, even if the semantic results are more relevant to the query.

def normalize(scores : list):
    if not scores:
        return None
    
    elif max(scores) == min(scores):
        a= [1 for _ in scores]  # All scores are the same, assign a neutral value
        for _score in a:
            print(f"* {_score:.2f}")
    else: 
        normalized_scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]
      
    return normalized_scores

# after normalizing we will combine the score using alpha .
#alpha is a weight that determines the importance of the BM25 score relative to the semantic score.
# for something that you feel is the exact match we can use a higher alpha value to give more weight to the BM25 score.
# This is why it's so important to tune your search system's constants based on the
#  types of data and queries you're working with in your application! It's not a one-size-fits-all solution, 

def hybrid_score(bm25_score, semantic_score, alpha):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_dict(scores_dict):
    if not scores_dict:
        return None
    
    score = list(scores_dict.values())
    if max(score) == min(score):
        return {k: 1 for k in scores_dict}  # All scores are the same, assign a neutral value
    else:
        min_score = min(score)
        max_score = max(score)
        return {k: (v - min_score) / (max_score - min_score) for k, v in scores_dict.items()}

def combined_results(query, alpha,limit):
    data = load_movies()
    HS = HybridSearch(data)
    result = HS.weighted_search(query, alpha, limit)
    for idx, res in enumerate(result):
        print(f'{res['title']}\nHybrid Score: {res['hybrid_score']:.4f}\nBM25: {res['bm25_score']:.4f} , Semantic: {res['semantic_score']:.4f}\nDocument: {res['document'][:100]}...\n')

def rrf_score(rank, k = 60):
    return 1 / (k + rank)

def rank_dict(scores_dict,k = 60):
    result_dict = {}
    i = 1
    for key,value in scores_dict.items():
        result_dict[key] = rrf_score(i,k)
        i+=1
    
    return result_dict

def reverse_rank(score,k=60):
    return (1/score) - k


def rrfsearch(query, k, limit,enhance = None,rerank_method = None, print_output=True):
    #print(f" enhance = {enhance!r}")
    #logging.debug(f"Searching for query: {query}")
    match enhance:
        case "expand":
            new_query = expand_query(query)
            print(f"Enhanced query ({enhance}): '{query}' -> '{new_query}'\n")
            query = new_query
            logging.debug(f"Enhanced query ({enhance}): '{query}' -> '{new_query}'")

        case "spell":
            new_query = correct_spell(query)
            print(f"Enhanced query ({enhance}): '{query}' -> '{new_query}'\n")
            query = new_query
            logging.debug(f"Enhanced query ({enhance}): '{query}' -> '{new_query}'")
        case "rewrite":
            new_query = rewrite_query(query)
            print(f"Enhanced query ({enhance}): '{query}' -> '{new_query}'\n")
            query = new_query
            logging.debug(f"Enhanced query ({enhance}): '{query}' -> '{new_query}'")
    data = load_movies()
    HS = HybridSearch(data)
    limit = limit*5 if rerank_method in ["individual", "batch","cross_encoder"] else limit
    result = HS.rrf_search(query, k, limit)
    #logging.debug(f"Initial RRF results for query '{query}': {len(result)} documents found")
    if rerank_method == "individual":
        for res in result:
            score = Rerank(query,{'title':res['title'],'document':res['document']})
            res['Re-rank Score'] = score
        result = sorted(result, key=lambda x: x['Re-rank Score'], reverse=True)[:limit//5]
        if print_output:
            for idx, res in enumerate(result):
                print(f'Re-ranking top {limit} results using individual method...\n'
    f'Reciprocal Rank Fusion Results for {query} (k={k}):\n'
    f'{idx+1}. {res["title"]}\n'
    f'Re-rank Score: {res["Re-rank Score"]:.3f} / 10\n'
    f'RRF Score: {res["rrf_score"]:.3f} / 10\n'
    f'BM25 Rank: {res["bm25_rank"]:.3f}, '
    f'Semantic Rank: {res["semantic_rank"]:.3f}\n'
    f'Document: {res["document"][:100]}...\n')
    
    elif rerank_method == "batch":
        scores = batch_rerank(query,result)
        id_to_rank = {doc_id : rank for rank, doc_id in enumerate(scores, start=1)}
        result = sorted(result, key=lambda x: id_to_rank.get(x['doc_id'], float('inf')))[:limit//5]
        if print_output:
            for idx, res in enumerate(result):
             print(f'Re-ranking top {limit} results using batch method...\n'
    f'Reciprocal Rank Fusion Results for {query} (k={k}):\n'
    f'{idx+1}. {res["title"]}\n'
    f'Re-rank Rank: {id_to_rank.get(res["doc_id"], float("inf"))}\n'
    f'RRF Score: {res["rrf_score"]:.3f}\n'
    f'BM25 Rank: {res["bm25_rank"]:.3f}, '
    f'Semantic Rank: {res["semantic_rank"]:.3f}\n'
    f'Document: {res["document"][:100]}...\n')


    elif rerank_method == "cross_encoder":
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        pairs =[]
        for res in result:
            pair = []
            pair.append(query)
            pair.append(f'{res['title']} - {res['document']}')
            pairs.append(pair)
        # `predict` returns a list of numbers, one for each pair
        scores = cross_encoder.predict(pairs) 
        for idx, res in enumerate(result):
            res['Cross-Encoder Score'] = scores[idx]
        result = sorted(result, key=lambda x: x['Cross-Encoder Score'], reverse=True)[:limit//5]
        if print_output:
            for idx, res in enumerate(result):
                print(f'Re-ranking top {limit} results using cross-encoder method...\n'
    f'Reciprocal Rank Fusion Results for {query} using cross encoder method....(k={k}):\n'
    f'{idx+1}. {res["title"]}\n'
    f'Cross-Encoder Score: {res["Cross-Encoder Score"]:.3f}\n'
    f'RRF Score: {res["rrf_score"]:.3f}n'
    f'BM25 Rank: {res["bm25_rank"]:.3f}, '
    f'Semantic Rank: {res["semantic_rank"]:.3f}\n'
    f'Document: {res["document"][:100]}...\n')
    
    else:
        if print_output:
            for idx, res in enumerate(result):
                print(f'{res['title']}\nHybrid Score: {res['rrf_score']:.4f}\nBM25: {res['bm25_score']:.4f} , Semantic: {res['semantic_score']:.4f}\nBM25: {res['bm25_rank']:.4f} , Semantic: {res['semantic_rank']:.4f}\nDocument: {res['document'][:100]}...\n')
    #logging.debug(f"Final results for query '{query}': {result}")
    
    return result



def evaluate_by_llm(query, k, limit,enhance = None,rerank_method = None, print_output=False):
    results = rrfsearch(query,k,limit,enhance,rerank_method,print_output)
    results_for_evaluation = []
    for idx, res in enumerate(results):
        results_for_evaluation.append(f"{idx+1}. {res['title']} - {res['document'][:200]}...")
    results_for_evaluation = chr(10).join(results_for_evaluation)
    evaluation = evaluate_results(query,results_for_evaluation)
    #print(evaluation)
    eval = json.loads(evaluation)
    for i in range(len(eval)):
        print(f'{i+1}. {results[i]["title"]} : {eval[i]}/3')

    return eval

