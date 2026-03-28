#!/usr/bin/env python3
"""
PROMPT 3: Document Chunking Script

Main script for chunking all processed legal documents.

Usage:
    python chunk_documents.py
    
    Optional:
    --input-dir data/processed
    --output-dir data/chunked
    --chunk-size 300
    --overlap 50
    --limit 100  (for testing)
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from ingestion.chunker import DocumentChunker, setup_logging


def main():
    """Main entry point for chunking."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Chunk legal documents for retrieval'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'processed',
        help='Input directory with processed JSONL files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'chunked',
        help='Output directory for chunked files'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=300,
        help='Tokens per chunk'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=50,
        help='Token overlap between chunks'
    )
    parser.add_argument(
        '--min-chunk',
        type=int,
        default=20,
        help='Minimum tokens per chunk'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit records processed (for testing)'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        default=None,
        help='Log file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(str(args.log_file) if args.log_file else None)
    
    logger.info("="*70)
    logger.info("PROMPT 3: DOCUMENT CHUNKING")
    logger.info("="*70)
    
    # Validate paths
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Configuration:")
    logger.info(f"  Input dir: {args.input_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Chunk size: {args.chunk_size} tokens")
    logger.info(f"  Overlap: {args.overlap} tokens")
    logger.info(f"  Min chunk: {args.min_chunk} tokens")
    
    if args.limit:
        logger.warning(f"  TESTING MODE: limited to {args.limit} records")
    
    # Initialize chunker
    try:
        chunker = DocumentChunker(
            chunk_size=args.chunk_size,
            overlap_size=args.overlap,
            min_chunk_size=args.min_chunk,
            tokenizer_name="distilbert-base-multilingual-cased"
        )
        logger.info(f"✅ DocumentChunker initialized")
    except Exception as e:
        logger.error(f"Failed to initialize DocumentChunker: {e}")
        sys.exit(1)
    
    # Process all JSONL files
    input_files = list(args.input_dir.glob('*.jsonl'))
    
    if not input_files:
        logger.warning(f"No JSONL files found in {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} JSONL files to process")
    
    # Tracking
    total_records = 0
    total_chunks = 0
    total_errors = 0
    
    # Process each file
    for input_file in input_files:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {input_file.name}")
        logger.info(f"{'='*70}")
        
        # Output file
        output_file = args.output_dir / f"{input_file.stem}_chunked.jsonl"
        
        try:
            records, chunks, errors = chunker.process_jsonl_file(
                str(input_file),
                str(output_file),
                limit=args.limit
            )
            
            total_records += records
            total_chunks += chunks
            total_errors += errors
            
            logger.info(f"✅ Complete:")
            logger.info(f"   Records: {records:,}")
            logger.info(f"   Chunks: {chunks:,}")
            logger.info(f"   Errors: {errors}")
            
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024*1024)
                logger.info(f"   Output file: {output_file.name} ({size_mb:.2f} MB)")
        
        except Exception as e:
            logger.error(f"❌ Failed to process {input_file.name}: {e}")
            total_errors += 1
            continue
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL STATISTICS")
    logger.info(f"{'='*70}")
    logger.info(f"Total records processed: {total_records:,}")
    logger.info(f"Total chunks created: {total_chunks:,}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Avg chunks per record: {total_chunks/max(total_records, 1):.2f}")
    
    # Create summary report
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'input_dir': str(args.input_dir),
            'output_dir': str(args.output_dir),
            'chunk_size': args.chunk_size,
            'overlap': args.overlap,
            'min_chunk_size': args.min_chunk,
        },
        'results': {
            'files_processed': len(input_files),
            'total_records': total_records,
            'total_chunks': total_chunks,
            'total_errors': total_errors,
            'chunks_per_record_avg': round(total_chunks / max(total_records, 1), 2),
        }
    }
    
    # Save report
    report_file = args.output_dir / 'chunking_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📊 Report saved: {report_file}")
    logger.info(f"{'='*70}\n")
    
    return 0 if total_errors == 0 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
