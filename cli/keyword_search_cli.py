import argparse
from lib.keyword_search import search_command ,InvertedIndex ,bm25_idf_command ,bm25_tf_command,BMsearch
import math
BM25_K1 = 1.5 
BM25_B = 0.75

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")


    build_praser = subparsers.add_parser("build",help="Build the inverted index")

    tf_praser = subparsers.add_parser("tf",help="returns the number of time token repeats in the document")
    tf_praser.add_argument('doc_id',type = int , help='doc_id to search from')
    tf_praser.add_argument('term',type = str , help='term to search from')


    idf_praser = subparsers.add_parser("idf",help="returns the Inverse Document Frequency")
    idf_praser.add_argument('term',type = str , help='term to search from')


    tfidf_praser = subparsers.add_parser("tfidf",help="returns the number of time token repeats in the document")
    tfidf_praser.add_argument('doc_id',type = int , help='doc_id to search from')
    tfidf_praser.add_argument('term',type = str , help='term to search from')

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
  "bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    
    
    
    args = parser.parse_args()

    match args.command:
        case "bm25search":
            
            bm = BMsearch(args.query)
            for key , value in bm.items():
                print(f"{key}. ({value[0]}) {value[1]} - Score: {value[2]:.2f}" )
            
            

        case "search":

            movies = search_command(args.query,5)
            for  movie in movies:
                print (f'{movie['id']}. {movie['title']}')

        case "build":
            II = InvertedIndex()
            II.build()
            II.save()
            docs = II.get_documents('merida')
            print(f"First document for token 'merida' = {docs[0]}")
            

        case "tf":
    
            idx = InvertedIndex()
            idx.load()
            print("doc_id in tf map?", args.doc_id in idx.term_frequencies)
            #print("counter for doc:", idx.term_frequencies.get(args.doc_id))
            print(idx.get_tf(args.doc_id, args.term))

        case "idf":
            idx= InvertedIndex()
            idx.load()
            idf = idx.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            
        case "tfidf":
            idx= InvertedIndex()
            idx.load()
            tfidf = idx.get_tfidf(args.doc_id,args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
            
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
             
            bm25tf = bm25_tf_command(args.doc_id,args.term,args.k1,args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()