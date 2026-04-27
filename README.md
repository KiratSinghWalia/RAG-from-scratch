# RAG Search Engine

A comprehensive Retrieval-Augmented Generation (RAG) search engine for movie discovery on Hoopla, a movie streaming service. This project combines keyword-based search (BM25), semantic search (embeddings), and advanced NLP techniques to provide intelligent movie recommendations.

## Features

- **Keyword Search**: BM25-based full-text search with inverted indexing
- **Semantic Search**: Embedding-based search using sentence transformers
- **Chunked Semantic Search**: Search within document chunks for more granular results
- **Hybrid Search**: Combines BM25 and semantic search using Reciprocal Rank Fusion (RRF)
- **Query Enhancement**: Spell correction, query rewriting, and query expansion using LLMs
- **Re-ranking**: Individual, batch, and cross-encoder based re-ranking methods
- **Multimodal Search**: Image-based movie search using CLIP embeddings
- **RAG**: Retrieval-augmented generation with various output formats (RAG, summarization, citations, Q&A)
- **Evaluation**: Built-in evaluation metrics (Precision, Recall, F1-Score)

## Project Structure

```
cli/
├── augmented_generation_cli.py      # RAG interface
├── describe_image_cli.py            # Image description & search
├── evaluation_cli.py                # Evaluation interface
├── hybrid_search_cli.py             # Hybrid search interface
├── keyword_search_cli.py            # Keyword search interface
├── multimodal_search_cli.py         # Multimodal search interface
├── semantic_search_cli.py           # Semantic search interface
└── lib/
    ├── augmented_generation.py      # RAG implementation
    ├── evaluation.py                # Evaluation metrics
    ├── hybrid_search.py             # Hybrid search logic
    ├── keyword_search.py            # BM25 & inverted index
    ├── llm.py                       # LLM interactions (Google Gemini)
    ├── multimodel_search.py         # Multimodal search logic
    ├── rerank.py                    # Re-ranking methods
    ├── search_utils.py              # Utility functions
    ├── semantic_search.py           # Embedding-based search
    └── prompts/                     # LLM prompt templates

data/
├── movies.json                      # Movie dataset
├── golden_dataset.json              # Evaluation test cases
└── stopwords.txt                    # Stopwords list

cache/
├── chunk_embeddings.npy             # Cached chunk embeddings
├── chunk_metadata.json              # Chunk metadata
├── docmap.pkl                       # Document mapping
├── index.pkl                        # Inverted index
├── term_frequencies.pkl             # Term frequencies
└── doc_lengths.pkl                  # Document lengths
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KiratSinghWalia/RAG-from-scratch
   cd rag-search-engine
   ```

2. **Install dependencies (uv needed)**
   ```bash
   uv sync
   ```
   

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

## Usage

### Keyword Search
```bash
python cli/keyword_search_cli.py search "action movie"
python cli/keyword_search_cli.py build
python cli/keyword_search_cli.py bm25search "adventure"
```

### Semantic Search
```bash
python cli/semantic_search_cli.py verify
python cli/semantic_search_cli.py embed_text "Inception"
python cli/semantic_search_cli.py search "mind-bending science fiction" --limit 5
python cli/semantic_search_cli.py search_chunked "animated family movie" --limit 5
```

### Hybrid Search
```bash
python cli/hybrid_search_cli.py rrf-search "scary bear movie" --k 60 --limit 5
python cli/hybrid_search_cli.py rrf-search "action" --enhance spell --rerank-method individual
python cli/hybrid_search_cli.py rrf-search "comedy" --evaluate
```

### RAG (Retrieval-Augmented Generation)
```bash
python cli/augmented_generation_cli.py rag "What horror movies are available?"
python cli/augmented_generation_cli.py summarize "animated movies for kids"
python cli/augmented_generation_cli.py citations "best drama movies"
python cli/augmented_generation_cli.py question "recommend movies like Inception"
```

### Multimodal Search
```bash
python cli/multimodal_search_cli.py image_search /path/to/image.jpg
python cli/multimodal_search_cli.py verify_image_embedding /path/to/image.jpg
```

### Image Description
```bash
python cli/describe_image_cli.py --image /path/to/image.jpg --query "action scene"
```

### Evaluation
```bash
python cli/evaluation_cli.py --limit 5
```

## Key Components

### Search Utilities
Helper functions for loading movies, stopwords, and prompt templates.

### Semantic Search
- Uses `all-MiniLM-L6-v2` model for text embeddings
- Supports chunked embeddings for fine-grained search
- Cosine similarity-based relevance scoring

### Keyword Search
- BM25 ranking with tunable parameters (k1, b)
- Inverted index with term frequencies
- Stopword removal and porter stemming

### Hybrid Search
- Combines BM25 and semantic scores
- Reciprocal Rank Fusion (RRF) algorithm
- Individual, batch, and cross-encoder re-ranking

### LLM Integration
- Uses Google Gemini API (gemma-3-27b-it model)
- Query enhancement (spell correction, rewriting, expansion)
- RAG, summarization, citation generation, and Q&A

## Configuration

### BM25 Parameters
```python
BM25_K1 = 1.5    # Term saturation parameter
BM25_B = 0.75    # Length normalization parameter
```

### RRF Parameters
```python
k = 60           # RRF constant (higher = more weight to top results)
```

### Models
- **Semantic Search**: `all-MiniLM-L6-v2` (384 dimensions)
- **Multimodal**: `clip-ViT-B-32`
- **Cross-Encoder**: `cross-encoder/ms-marco-TinyBERT-L2-v2`
- **LLM**: Google Gemini `gemma-3-27b-it`

## Evaluation Metrics

The system evaluates search results using:
- **Precision@k**: Percentage of top-k results that are relevant
- **Recall@k**: Percentage of relevant documents found in top-k
- **F1-Score**: Harmonic mean of precision and recall

## Performance Tips

1. **Cache embeddings**: The system automatically caches generated embeddings
2. **Tune alpha**: For weighted search, experiment with alpha values (0.0-1.0)
3. **Use re-ranking**: Re-ranking improves result quality significantly
4. **Query enhancement**: Enable spell correction or expansion for better results

## Environment Variables

```env
GEMINI_API_KEY=your_api_key_here
```

## Requirements

- Python 3.13+
- Google GenAI API access
- See [`pyproject.toml`](pyproject.toml) for full dependency list

## License

MIT License - See [`LICENSE`](LICENSE) file for details.

## Author

Kirat Singh (2026)

















BM25_K1 = 1.5    # Term saturation parameter
BM25_B = 0.75    # Length normalization parameter

k = 60           # RRF constant (higher = more weight to top results)

Models
Semantic Search: all-MiniLM-L6-v2 (384 dimensions)
Multimodal: clip-ViT-B-32
Cross-Encoder: cross-encoder/ms-marco-TinyBERT-L2-v2
LLM: Google Gemini gemma-3-27b-it
Evaluation Metrics
The system evaluates search results using:

Precision@k: Percentage of top-k results that are relevant
Recall@k: Percentage of relevant documents found in top-k
F1-Score: Harmonic mean of precision and recall
Performance Tips
Cache embeddings: The system automatically caches generated embeddings
Tune alpha: For weighted search, experiment with alpha values (0.0-1.0)
Use re-ranking: Re-ranking improves result quality significantly
Query enhancement: Enable spell correction or expansion for better results
Environment Variables
Requirements
Python 3.13+
Google GenAI API access
See pyproject.toml for full dependency list
License
MIT License - See LICENSE file for details.

Author
Kirat Singh (2026)