#!/usr/bin/env python3
"""
PROMPT 4: Test Dense Retrieval

Load FAISS index and test search functionality.

Usage:
    python test_dense_retrieval.py
    python test_dense_retrieval.py --query "Kasten adam öldürme"
"""

import argparse
import sys
import logging
from pathlib import Path
import json

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from retrieval.dense import DenseRetriever


def main():
    """Main entry point for testing."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Test dense retrieval'
    )
    parser.add_argument(
        '--index-dir',
        type=Path,
        default=PROJECT_ROOT / 'models' / 'retrieval_index',
        help='Directory containing FAISS index'
    )
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Single query to test (optional)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return'
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
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("PROMPT 4: TEST DENSE RETRIEVAL")
    logger.info("="*70)
    
    # Load retriever with existing index
    try:
        logger.info(f"\nLoading FAISS index from {args.index_dir}...")
        
        retriever = DenseRetriever(
            model_name=args.model,
            index_dir=str(args.index_dir),
            device=args.device
        )
        
        if not retriever.load_index():
            logger.error("Failed to load index")
            sys.exit(1)
        
        logger.info(f"✅ Index loaded: {retriever.index.ntotal:,} vectors")
        logger.info(f"   Dimension: {retriever.config['embedding_dimension']}")
        logger.info(f"   Model: {retriever.config['model_name']}")
    
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        sys.exit(1)
    
    # Define test queries
    if args.query:
        # Single query provided
        test_queries = [args.query]
    else:
        # Default test queries
        test_queries = [
            "Kasten adam öldürme suçu nedir?",
            "Taksir halinde ölüm cezası",
            "Ağırlaştırıcı sebepler Türk Ceza Kanunu",
            "Hapis cezası minimum ve maksimum süre",
            "Madde 81 ve 82 arasındaki fark"
        ]
    
    # Test search
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing search with {len(test_queries)} queries (top-{args.top_k})...")
    logger.info(f"{'='*70}\n")
    
    for query in test_queries:
        logger.info(f"Query: {query}")
        
        try:
            results = retriever.search(query, k=args.top_k)
            
            logger.info(f"Results: {len(results)} chunks")
            
            for result in results:
                logger.info(f"\n  Rank {result.rank}:")
                logger.info(f"    Score: {result.score:.4f}")
                logger.info(f"    Law: {result.metadata.get('law_name')} "
                           f"Madde {result.metadata.get('article_no')}")
                logger.info(f"    Text: {result.chunk_text[:100]}...")
            
            logger.info(f"\n{'─'*70}\n")
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            sys.exit(1)
    
    # Export sample results
    logger.info(f"{'='*70}")
    logger.info("Exporting sample results...")
    logger.info(f"{'='*70}\n")
    
    sample_query = test_queries[0]
    results = retriever.search(sample_query, k=5)
    
    export_data = {
        'query': sample_query,
        'results': [result.to_dict() for result in results]
    }
    
    export_file = args.index_dir / 'sample_search_results.json'
    with open(export_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Sample results exported: {export_file}")
    logger.info(f"\nReady for Prompt 5: Reranking with Cross-Encoder")
    logger.info(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
