from collections import defaultdict
import enum
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from lib.search_utils import load_movies
import re
import json
class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = Path('cache/movie_embeddings.npy')

    def generate_embedding(self, text):
        if not text or not text.strip()  :
            raise ValueError ("Invalid token")

        embeddings = self.model.encode([text])[0]  
        return embeddings

    def build_embeddings(self, documents:list[dict]):
        self.documents = documents
        movie_string=[]
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc # something like - {1.{doc},2.{doc}}
            movie_string.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_string, show_progress_bar =True)
        np.save(self.embeddings_path,self.embeddings)
        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError('"No embeddings loaded. Call `load_or_create_embeddings` first."')
        query_embedding = self.generate_embedding(query)
        similarities = []
        for doc_embeddings, doc in zip(self.embeddings,self.documents):
            _similarity = cosine_similarity(doc_embeddings,query_embedding)
            similarities.append((_similarity,doc))
        
        similarities = sorted(similarities,key = lambda x : x[0] , reverse = True)
        res =[]
        for i in similarities[:limit]:
            score,doc = i
            res.append({
                'score':score,
                'title':doc['title'],
                'description':doc['description']
            })

        return res


    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)


def verify_model():
    ss= SemanticSearch()
    print(f'Model loaded: {ss.model}')
    print(f'Max sequence length: {ss.model.max_seq_length}')

    pass

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    SS = SemanticSearch()
    documents = load_movies()
    embeddings = SS.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")   


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1) #magnitude
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2) 


def semantic_search(query :str , limit = 5 ):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    res = ss.search(query,limit)
    for i,result in enumerate(res):
        print(f'{i}. {result['title']}: {result['score']:.4f}\n{result['description'][:100]}')

def semantic_chunk(text :str,overlap,limit):
    text = text.strip() # remove leading and trailing whitespace
    if not text:
        return [] 
    words = re.split(r"(?<=[.!?])\s+",text)
    if len(words) == 1 and words[0].endswith(('.','!','?')):
        pass
    words = [s.strip() for s in words if len(s) > 0]
        
    chunks = [] 
    jump = limit - overlap
    for i in range(0,len(words),jump):
        chunkwords = words[i:i+limit]
        if len(chunkwords) <= overlap:
            break
        chunks.append(" ".join(chunkwords))
    return chunks

def chunk_text(text,overlap,limit):
    chunks = semantic_chunk(text,overlap,limit)
    print(f'Semantically chunking {len(text)} characters')
    for i , chunk in enumerate(chunks):
        print(f'{i+1}. {chunk}')


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = Path('cache/chunk_embeddings.npy')
        self.chunk_metadata_path = Path('cache/chunk_metadata.json')

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        chunks = []
        chunk_metadata = []

        for idx , doc in enumerate(documents):
            if not doc['description']:
                continue
            _chunk = semantic_chunk(doc['description'],1,4)
            chunks += _chunk
            for cidx in range(len(_chunk)):
                single_chunk_metadata = {
                        'movie_idx':idx,
                        'chunk_idx':cidx,
                        'total_chunks':len(_chunk)
                }
                chunk_metadata.append(single_chunk_metadata)
                
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar =True)   
        self.chunk_metadata = chunk_metadata
        np.save(self.chunk_embeddings_path,self.chunk_embeddings)
        with open(self.chunk_metadata_path,'w') as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc

        if self.chunk_metadata_path.exists() and self.chunk_embeddings_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path,'r') as f:
                self.chunk_metadata = json.load(f)['chunks']

            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)


    def search_chunks(self, query: str, limit: int):
        query_embeddings = self.generate_embedding(query)
        chunk_score = []
        movie_scores = defaultdict(lambda:0)
        for i in range(len(self.chunk_embeddings)):
            _chunk_embeddings = self.chunk_embeddings[i]
            metadata = self.chunk_metadata[i]
            chunk_idx, movie_idx = metadata['chunk_idx'],metadata['movie_idx'] 
            cosine_score =cosine_similarity(_chunk_embeddings,query_embeddings)
            chunk_score.append({
                'chunk_idx': chunk_idx,
                'movie_idx': movie_idx,
                'score' : cosine_score
            })
            movie_scores[movie_idx] = max(movie_scores[movie_idx],cosine_score)
        movie_scores = sorted(movie_scores.items() , key= lambda x : x[1] , reverse= True)
        res = []
        for midx, score in movie_scores[:limit]:
            doc = self.documents[midx]
            res.append({
  "id": doc['id'],
  "title": doc['title'],
  "document": doc['description'][:limit],
  "score": round(score, 4),
  "metadata": {}
})
        return res

            
    
        
def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)

    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query,limit):
    data = load_movies()
    css = ChunkedSemanticSearch()
    css.load_or_create_chunk_embeddings(data)
    chunk_result = css.search_chunks(query,limit)
    for idx ,chunk in enumerate(chunk_result):
        print(f"\n{idx+1}. {chunk['title']} (score: {chunk['score']:.4f})")
        print(f"   {chunk['document']}...")

    
