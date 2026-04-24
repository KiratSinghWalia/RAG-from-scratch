import argparse
from lib.augmented_generation import augmented_generation,summarize_RAG ,citation_RAG,question_RAG

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--limit", type=int, default=5, help="Limit for search results")
    
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Limit for search results")

    citation_parser = subparsers.add_parser(
        "citations", help="Generate citations for search results"
    )
    citation_parser.add_argument("query", type=str, help="Search query for citation generation")
    citation_parser.add_argument("--limit", type=int, default=5, help="Limit for search results")

    question_parser = subparsers.add_parser(
        "question", help="Answer questions based on search results"
    )
    question_parser.add_argument("query", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, default=5, help="Limit for search results")

    args = parser.parse_args()

    match args.command:
        case "question":
            query = args.query
            limit = args.limit
            question_RAG(query, limit)
        case "citations":
            query = args.query
            limit = args.limit
            citation_RAG(query, limit)  

        case "rag":
            query = args.query
            limit = args.limit
            augmented_generation(query, limit)
        case "summarize":
            query = args.query
            limit = args.limit
            summarize_RAG(query, limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()