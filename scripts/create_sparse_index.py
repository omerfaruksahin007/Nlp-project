#!/usr/bin/env python3
"""
PROMPT 5: Create Sparse (BM25) Index

Build BM25 index from chunked documents for fast keyword-based search.

Usage:
    python create_sparse_index.py
    python create_sparse_index.py --chunks data/chunked/turkish_law_chunked.jsonl
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

from retrieval.sparse import setup_sparse_retriever


def main():
    """Main entry point for sparse index creation."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Create BM25 sparse retrieval index'
    )
    parser.add_argument(
        '--chunks',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'chunked' / 'turkish_law_chunked.jsonl',
        help='Path to chunked JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'models' / 'sparse_index',
        help='Output directory for BM25 index'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='distiluse-base-multilingual-cased-v2',
        help='Tokenizer model name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Compute device (cpu or cuda)'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        default=None,
        help='Log file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(str(args.log_file)),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
    
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("PROMPT 5: Create Sparse (BM25) Index")
    logger.info("="*80)
    
    # Validate input
    if not args.chunks.exists():
        logger.error(f"Chunks file not found: {args.chunks}")
        return 1
    
    logger.info(f"Chunks file: {args.chunks}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Device: {args.device}\n")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup sparse retriever
    logger.info("Setting up sparse retriever...")
    retriever, chunk_count = setup_sparse_retriever(
        chunks_file=str(args.chunks),
        tokenizer_name=args.tokenizer,
        device=args.device
    )
    
    if retriever is None:
        logger.error("Failed to setup sparse retriever")
        return 1
    
    # Save BM25 index
    logger.info(f"\nSaving BM25 index...")
    try:
        index_file = args.output_dir / "bm25.pkl"
        with open(index_file, 'wb') as f:
            pickle.dump(retriever.bm25, f)
        
        logger.info(f"✅ BM25 index saved: {index_file}")
        
        # Save tokenized chunks for reference
        tokens_file = args.output_dir / "tokenized_chunks.json"
        tokens_data = {
            'chunk_count': len(retriever.tokenized_chunks),
            'sample_tokenized_chunks': [
                {
                    'chunk_idx': i,
                    'tokens': retriever.tokenized_chunks[i][:50],  # First 50 tokens
                    'token_count': len(retriever.tokenized_chunks[i])
                }
                for i in range(min(5, len(retriever.tokenized_chunks)))
            ]
        }
        
        with open(tokens_file, 'w', encoding='utf-8') as f:
            json.dump(tokens_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Tokenized chunks info saved: {tokens_file}")
        
        # Save metadata
        metadata_file = args.output_dir / "bm25_metadata.json"
        metadata = {
            'chunk_count': chunk_count,
            'tokenizer': args.tokenizer,
            'k1': retriever.k1,
            'b': retriever.b,
            'avg_tokens_per_chunk': sum(len(t) for t in retriever.tokenized_chunks) / len(retriever.tokenized_chunks) if retriever.tokenized_chunks else 0
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Metadata saved: {metadata_file}")
        
    except Exception as e:
        logger.error(f"Failed to save index: {e}", exc_info=True)
        return 1
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("✅ Sparse Index Creation Complete!")
    logger.info("="*80)
    logger.info(f"Index location: {args.output_dir}")
    logger.info(f"Files:")
    logger.info(f"  - bm25.pkl: BM25 index")
    logger.info(f"  - bm25_metadata.json: Index metadata")
    logger.info(f"  - tokenized_chunks.json: Sample tokenized chunks")
    logger.info(f"\nNext step: scripts/test_hybrid_retrieval.py")
    logger.info("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
