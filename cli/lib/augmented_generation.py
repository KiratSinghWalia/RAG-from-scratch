from lib.hybrid_search import rrfsearch
from lib.llm import RAG, summarize ,generate_citation ,question_answering



def augmented_generation(query,limit) -> str:
    results = rrfsearch(query, k=60, limit=limit, enhance=None, rerank_method=None, print_output=False)
    docs=[]
    for idx, res in enumerate(results):
        docs.append(f"{idx+1}. {res['title']} - {res['document'][:200]}...")
    RAG_result = RAG(query, docs, limit)
    print('search results:\n')
    for res in results:
        print(f'- {res["title"]}')
    
    print('\nRAG Response:\n')
    print(RAG_result)

def summarize_RAG(query,limit) -> str:
    results = rrfsearch(query, k=60, limit=limit, enhance=None, rerank_method=None, print_output=False)
    docs=[]
    for idx, res in enumerate(results):
        docs.append(f"{idx+1}. {res['title']} - {res['document'][:200]}...")
    sum_result = summarize(query, docs)
    print('search results:\n')
    for res in results:
        print(f'- {res["title"]}')
    
    print('\nSummarized Response:\n')
    print(sum_result)

def citation_RAG(query,limit) -> str:
    results = rrfsearch(query, k=60, limit=limit, enhance=None, rerank_method=None, print_output=False)
    docs=[]
    for idx, res in enumerate(results):
        docs.append(f"{idx+1}. {res['title']} - {res['document'][:200]}...")
    citation_result = generate_citation(query, docs)
    print('search results:\n')
    for res in results:
        print(f'- {res["title"]}')
    
    print('\nCitation Response:\n')
    print(citation_result)

def question_RAG(query,limit) -> str:
    results = rrfsearch(query, k=60, limit=limit, enhance=None, rerank_method=None, print_output=False)
    docs=[]
    for idx, res in enumerate(results):
        docs.append(f"{idx+1}. {res['title']} - {res['document']}...")
    question_result = question_answering(query, docs)
    print('search results:\n')
    for res in results:
        print(f'- {res["title"]}')
    
    print('\nQuestion Answering Response:\n')
    print(question_result)