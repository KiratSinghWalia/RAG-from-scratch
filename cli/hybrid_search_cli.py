import argparse
from lib.hybrid_search import normalize , combined_results , rrfsearch,evaluate_by_llm
from lib.llm import generate_content


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, help="Weight for BM25 score")
    weighted_search_parser.add_argument("--limit", type=int, default=10, help="Number of results to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="RRF parameter")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_search_parser.add_argument(
    "--enhance",
    type=str,
    choices=["spell", "rewrite", "expand"],
    help="Query enhancement method"
)
  
    rrf_search_parser.add_argument("--rerank-method",type=str,choices=["individual","batch","cross_encoder"],help="Reranking method")
    rrf_search_parser.add_argument(
    "--evaluate",
    action="store_true",
    help="Rate the search results after running the search"
)
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize(args.scores)
        case "weighted-search":
            combined_results(args.query, args.alpha, args.limit)
        case "rrf-search":
            if args.evaluate:
                evaluate_by_llm(args.query, args.k, args.limit,args.enhance,args.rerank_method,print_output=False)
            else:
                rrfsearch(args.query, args.k, args.limit,args.enhance,args.rerank_method)
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()