
from math import e
import string
from turtle import title
from typing import Optional
from lib.search_utils import load_movies,load_stopwords
from collections import defaultdict,Counter
from pathlib import Path
import pickle
import os


BM25_K1 = 1.5 
BM25_B = 0.75

class InvertedIndex():

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap = {}
        self.stopwardlist = load_stopwords()
        self.data = load_movies()
        self.rootfile = Path(__file__).resolve().parent.parent.parent
        self.dockmappath= self.rootfile/'cache'/'docmap.pkl'
        self.indexpath= self.rootfile/'cache'/'index.pkl'
        self.frequenciespath= self.rootfile/'cache'/'term_frequencies.pkl'
        self.term_frequencies = {}
        self.doc_lenghts = {}
        self.doc_lenghts_path = self.rootfile/'cache'/'doc_lengths.pkl'



    def __add_document(self,doc_id,text):
        tokenized_text = text_transformer(text,self.stopwardlist)
        self.term_frequencies[doc_id] = Counter()
        self.doc_lenghts[doc_id] = len(tokenized_text)
        for token in tokenized_text:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] +=1

        
    def get_tf(self,doc_id,term):
        tokenized_text = text_transformer(term,self.stopwardlist)
        if len(tokenized_text) > 1 :
            raise ValueError("More than one token.")
        if len(tokenized_text) == 1:
            if doc_id not in self.term_frequencies:
                return 0
            else:
                return self.term_frequencies[doc_id][tokenized_text[0]]
        else:
            raise ValueError("No token")


    def get_documents(self,term):
        term = term.lower()
        document_list = self.index.get(term,[])
        return sorted(document_list)

    def get_idf(self,term):
        import math
        token = text_transformer(term,self.stopwardlist)
        if len(token) != 1:
            raise ValueError('more than one word')
        tokenterm =token[0]
        
        total_doc_count = len(self.docmap)
        term_set = self.index[tokenterm]
        term_match_doc_count = len(term_set)    
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf

    def get_tfidf(self,doc_id,term):
        token = text_transformer(term,self.stopwardlist)
        tf = self.term_frequencies[doc_id][term]
        idf = self.get_idf(term)
        tfidf = tf * idf 
        return tfidf

    def get_bm25_idf(self, term: str) -> float:
        import math
        token = text_transformer(term,self.stopwardlist)
        if len(token) != 1:
            raise ValueError("More than one token or no token")
        df = self.index[token[0]]
        df =len(df)
        N = len(self.docmap)
        BM25 = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return float(BM25)


    def get_bm25_tf(self, doc_id, term, k1=BM25_K1,b=BM25_B):
        tf = self.get_tf(doc_id,term)
        doc_length = self.doc_lenghts[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * float((doc_length / avg_doc_length))
        BM25 = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return BM25

    def bm25(self, doc_id, term) :
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id,term)
        BM25 = tf * idf
        # idf = how rare a term is
        # tf = how often the term is in the doc
        return BM25

    def bm25_search(self, query, limit):
        tokens = text_transformer(query,self.stopwardlist)
        scores = {} 
        for doc_id in self.docmap:
            score = 0 
            for token in tokens:
                score += self.bm25(doc_id,token)
            scores[doc_id] = score
        sorted_scores = dict(sorted(scores.items(), key = lambda x : x[1],reverse=True))
        results = dict(list(sorted_scores.items())[:limit])
        return results

    def build(self):
        for movie in self.data:
            full_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie['id'],full_text)
            self.docmap[movie['id']] = movie
        print(f"movies in self.data: {len(self.data)}")
        print(f"docmap size: {len(self.docmap)}")

    def save(self):

        #os.makedirs("cache", exist_ok=True)
        with open(self.indexpath, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.dockmappath,"wb") as h:
            pickle.dump(self.docmap,h) 

        with open(self.frequenciespath, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lenghts_path, "wb") as f:
            pickle.dump(self.doc_lenghts, f)

    def load(self) :
        try :
            with open(self.indexpath, "rb") as f:
                self.index = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("The file 'index.pkl' does not exist")
        

        try :
            with open(self.dockmappath, "rb") as h:
                self.docmap = pickle.load(h)
        except FileNotFoundError:
            raise FileNotFoundError("The file 'docmap.pkl' does not exist")


        try :
            with open(self.frequenciespath, "rb") as h:
                self.term_frequencies = pickle.load(h)
        except FileNotFoundError:
            raise FileNotFoundError("The file 'term_frequencies.pkl' does not exist")

        try :
            with open(self.doc_lenghts_path, "rb") as h:
                self.doc_lenghts = pickle.load(h)
        except FileNotFoundError:
            raise FileNotFoundError("The file 'doc_lenghts.pkl' does not exist")

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lenghts) != 0 :
            avg_doc_len =float( sum(self.doc_lenghts.values())/ len(self.doc_lenghts))       
        else : 
            avg_doc_len = 0.0
        return avg_doc_len

    def transform_query(self,query):
        return text_transformer(query,self.stopwardlist)


def text_transformer(text,stopwardlist) -> list:

    import string 
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    text = text.lower() #1
    text = text.translate(str.maketrans("","",string.punctuation)) #2
    res=[]
    tokens = [tok for tok in text.split() if tok]
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwardlist]
    return tokens


def search_command(query,n_results):
    idx = InvertedIndex()
    idx.load()
    stopwardlist = load_stopwords()
    token_query = idx.transform_query(query)
    seen,res=set(),[]
    for token in token_query:
        matching_ids = idx.get_documents(token)
        for matching_id in matching_ids:
            if matching_id in seen:
                continue
            seen.add(matching_id)
            matching_doc = idx.docmap[matching_id]
            res.append(matching_doc)

            if len(res) >= n_results:
                return res
    return res


def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    BM25 = idx.get_bm25_idf(term)

    return BM25


def bm25_tf_command(doc_id,term,k1,B_value):
    idx = InvertedIndex()
    idx.load()
    
    BM25 = idx.get_bm25_tf(doc_id,term,k1,B_value)

    return BM25

def BMsearch(query,limit = 5 ):
    idx = InvertedIndex()
    idx.load()
    final_output = {}
    BM25 = idx.bm25_search(query,limit)
    i=1
    for doc_id, score in BM25.items():
        a=[]
        a.append(doc_id) 
        a.append(idx.docmap[doc_id]['title'])
        a.append(score)
        final_output[i] = a
        i+=1
    return final_output