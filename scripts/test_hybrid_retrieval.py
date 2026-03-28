#!/usr/bin/env python3
"""
PROMPT 5: Test Hybrid Retrieval

Test hybrid search combining dense + sparse results using RRF.

Usage:
    python test_hybrid_retrieval.py
    python test_hybrid_retrieval.py --query "Ceza hukuku nedir"
"""

import argparse
import sys
import logging
import json
import pickle
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever


def main():
    """Main entry point for testing hybrid retrieval."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Test hybrid retrieval'
    )
    parser.add_argument(
        '--dense-dir',
        type=Path,
        default=PROJECT_ROOT / 'models' / 'retrieval_index',
        help='Directory containing FAISS index'
    )
    parser.add_argument(
        '--sparse-dir',
        type=Path,
        default=PROJECT_ROOT / 'models' / 'sparse_index',
        help='Directory containing BM25 index'
    )
    parser.add_argument(
        '--chunks',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'chunked' / 'turkish_law_chunked.jsonl',
        help='Path to chunks file (for sparse retriever)'
    )
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Single query to test (optional)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='distiluse-base-multilingual-cased-v2',
        help='Embedding model name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Compute device'
    )
    parser.add_argument(
        '--k-dense',
        type=int,
        default=20,
        help='Number of dense results'
    )
    parser.add_argument(
        '--k-sparse',
        type=int,
        default=20,
        help='Number of sparse results'
    )
    parser.add_argument(
        '--k-final',
        type=int,
        default=10,
        help='Number of final hybrid results'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("PROMPT 5: TEST HYBRID RETRIEVAL")
    logger.info("="*80 + "\n")
    
    # Setup dense retriever
    logger.info("Initializing dense retriever...")
    try:
        dense_retriever = DenseRetriever(
            model_name=args.model,
            index_dir=str(args.dense_dir),
            device=args.device
        )
        
        if not dense_retriever.load_index():
            logger.error("Failed to load dense index")
            return 1
        
        logger.info(f"✅ Dense index loaded: {dense_retriever.index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to initialize dense retriever: {e}", exc_info=True)
        return 1
    
    # Setup sparse retriever
    logger.info("Initializing sparse retriever...")
    try:
        sparse_retriever = SparseRetriever(
            tokenizer_name=args.model,
            device=args.device
        )
        
        # Load BM25 index
        bm25_file = args.sparse_dir / "bm25.pkl"
        if not bm25_file.exists():
            logger.error(f"BM25 index not found: {bm25_file}")
            return 1
        
        with open(bm25_file, 'rb') as f:
            sparse_retriever.bm25 = pickle.load(f)
        
        logger.info(f"✅ BM25 index loaded")
        
        # Load chunks for sparse retriever
        logger.info(f"Loading chunks for sparse retriever...")
        chunks = []
        with open(args.chunks, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
        
        sparse_retriever.chunks = chunks
        
        # Build tokenized chunks (for search functionality)
        sparse_retriever.tokenized_chunks = []
        for chunk in chunks:
            text = chunk.get('chunk_text', '')
            tokens = text.lower().split()
            tokens = [t.strip('.,!?;:"\'-') for t in tokens]
            tokens = [t for t in tokens if len(t) > 1]
            sparse_retriever.tokenized_chunks.append(tokens)
        
        logger.info(f"✅ Sparse retriever initialized with {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Failed to initialize sparse retriever: {e}", exc_info=True)
        return 1
    
    # Setup hybrid retriever
    logger.info("Initializing hybrid retriever...")
    try:
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            dense_weight=0.6,
            sparse_weight=0.4,
            rrf_k=60
        )
    except Exception as e:
        logger.error(f"Failed to initialize hybrid retriever: {e}", exc_info=True)
        return 1
    
    # Test queries
    if args.query:
        test_queries = [args.query]
    else:
        test_queries = [
            "Ceza hukuku nedir",
            "Kasten adam öldürme cezası",
            "Türkiye anayasası"
        ]
    
    logger.info("\n" + "="*80)
    logger.info(f"Testing hybrid search with {len(test_queries)} queries")
    logger.info(f"(k_dense={args.k_dense}, k_sparse={args.k_sparse}, k_final={args.k_final})")
    logger.info("="*80 + "\n")
    
    all_results = []
    
    for query_idx, query in enumerate(test_queries, 1):
        logger.info(f"\n[Query {query_idx}/{len(test_queries)}] {query}")
        logger.info("-" * 80)
        
        try:
            results = hybrid_retriever.search(
                query,
                k_dense=args.k_dense,
                k_sparse=args.k_sparse,
                k_final=args.k_final,
                fusion_method="rrf"
            )
            
            logger.info(f"Got {len(results)} hybrid results:\n")
            
            for i, result in enumerate(results[:5], 1):  # Show top 5
                logger.info(f"  Rank {i}:")
                logger.info(f"    Hybrid Score: {result.hybrid_score:.6f}")
                logger.info(f"    Dense Score:  {result.dense_score:.4f} (rank {result.dense_rank})")
                logger.info(f"    Sparse Score: {result.sparse_score:.4f} (rank {result.sparse_rank})")
                logger.info(f"    Text: {result.chunk_text[:80]}...")
                logger.info()
            
            all_results.append({
                'query': query,
                'results': [
                    {
                        'chunk_id': r.chunk_id,
                        'hybrid_score': r.hybrid_score,
                        'dense_score': r.dense_score,
                        'sparse_score': r.sparse_score,
                        'text_snippet': r.chunk_text[:100]
                    }
                    for r in results[:5]
                ]
            })
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return 1
    
    # Save results
    logger.info("\n" + "="*80)
    logger.info("Exporting sample results...")
    
    results_file = args.sparse_dir / "hybrid_search_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Results saved: {results_file}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Hybrid Retrieval Test Complete")
    logger.info("="*80)
    logger.info("\nReady for Prompt 6: Cross-Encoder Reranking")
    logger.info("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
