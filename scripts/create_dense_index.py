#!/usr/bin/env python3
"""
PROMPT 4: Create Dense Retrieval Index

Build FAISS index from chunked documents for fast similarity search.

Usage:
    python create_dense_index.py
    python create_dense_index.py --chunks data/chunked/turkish_law_chunked.jsonl
    python create_dense_index.py --force-rebuild
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from retrieval.dense import setup_dense_retriever


def main():
    """Main entry point for index creation."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Create dense retrieval FAISS index'
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
        default=PROJECT_ROOT / 'models' / 'retrieval_index',
        help='Output directory for FAISS index'
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
        help='Compute device (cpu or cuda)'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Rebuild index even if exists'
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
            format=log_format
        )
    
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("PROMPT 4: CREATE DENSE RETRIEVAL INDEX")
    logger.info("="*70)
    
    # Validate inputs
    if not args.chunks.exists():
        logger.error(f"Chunks file not found: {args.chunks}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Chunks file: {args.chunks}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Device: {args.device}")
    
    if args.force_rebuild:
        logger.warning(f"  Force rebuild: YES (existing index will be overwritten)")
    
    # Create index
    try:
        logger.info(f"\n{'='*70}")
        logger.info("Creating dense retrieval index...")
        logger.info(f"{'='*70}")
        
        retriever = setup_dense_retriever(
            chunks_file=str(args.chunks),
            model_name=args.model,
            index_dir=str(args.output_dir),
            device=args.device,
            force_rebuild=args.force_rebuild
        )
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ Index creation complete!")
        logger.info(f"{'='*70}")
        logger.info(f"Index location: {args.output_dir}")
        logger.info(f"Files:")
        
        for f in sorted(args.output_dir.glob('*')):
            if f.is_file():
                size_mb = f.stat().st_size / (1024**2)
                logger.info(f"  {f.name}: {size_mb:.2f} MB")
        
        logger.info(f"\nNext step: scripts/test_dense_retrieval.py")
        logger.info(f"{'='*70}\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Failed to create index: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
