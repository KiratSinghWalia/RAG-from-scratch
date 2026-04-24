#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model,embed_text,verify_embeddings ,embed_query_text,semantic_search,chunk_text,embed_chunks,search_chunked


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("verify", help="verify the model loaded")

    embed_parser = subparsers.add_parser("embed_text", help="Using the model to embed the text")
    embed_parser.add_argument('text',type = str , help='text to embed')

    VE_parser = subparsers.add_parser("verify_embeddings", help="Create or load embeddings and verify embeddings")

    embedquery = subparsers.add_parser("embedquery", help="embed a given query")
    embedquery.add_argument('text',type = str , help='text to embed')

    search = subparsers.add_parser("search", help="semantic search on the query")
    search.add_argument('text',type = str , help='text to search')
    search.add_argument('--limit',type = int,default = 5  , help='limit of search')
    
    semantic_chunk = subparsers.add_parser("semantic_chunk", help="making chunk of text")
    semantic_chunk.add_argument('text',type = str , help='text to chunk')
    semantic_chunk.add_argument('--max-chunk-size',type = int ,default=4, help='text to chunk')
    semantic_chunk.add_argument('--overlap',type = int ,default=0, help='overlap')

    embed_chunks1= subparsers.add_parser("embed_chunks", help="making embeddings of chunks")

    search_chunks= subparsers.add_parser("search_chunked", help="searching of chunks")
    search_chunks.add_argument('text',type = str , help='text to search')
    search_chunks.add_argument('--limit',type = int ,default=5, help='text to search')
    


    args = parser.parse_args()

    match args.command:
        case 'search_chunked':
            search_chunked(args.text,args.limit)


        case 'semantic_chunk':
            chunk_text(args.text,args.overlap,args.max_chunk_size)

        case 'search':
            semantic_search(args.text,args.limit)

        case 'embed_chunks':
            embed_chunks()

        case 'embedquery':
            embed_query_text(args.text)

        case 'verify_embeddings':
            verify_embeddings()

        case 'embed_text':
            embed_text(args.text)

        case 'verify':
            verify_model()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()