#!/usr/bin/env python3
"""
Main data ingestion script.

Demonstrates how to use the ingestion pipeline on sample datasets.

Usage:
    python scripts/ingest_data.py
    
    python scripts/ingest_data.py --input data/raw/my_data.json --output processed_data --type qa
"""

import logging
import argparse
from pathlib import Path
from typing import Optional, Dict

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_qa_dataset(
    input_file: Path,
    output_name: str,
    source_name: str = 'unknown',
    field_mapping: Optional[Dict[str, str]] = None,
):
    """Ingest QA dataset."""
    pipeline = IngestionPipeline(output_dir=Path('data/processed'))
    
    stats = pipeline.ingest_qa_dataset(
        input_path=input_file,
        output_name=output_name,
        source_name=source_name,
        field_mapping=field_mapping,
        remove_duplicates=True,
    )
    
    logger.info(f"Ingestion complete!")
    logger.info(f"  Total loaded: {stats['total_loaded']}")
    logger.info(f"  Duplicates removed: {stats['duplicates_removed']}")
    logger.info(f"  Final count: {stats['final_count']}")
    
    return stats


def ingest_legal_text_dataset(
    input_file: Path,
    output_name: str,
    source_name: str = 'unknown',
    field_mapping: Optional[Dict[str, str]] = None,
):
    """Ingest legal text dataset."""
    pipeline = IngestionPipeline(output_dir=Path('data/processed'))
    
    stats = pipeline.ingest_legal_text_dataset(
        input_path=input_file,
        output_name=output_name,
        source_name=source_name,
        field_mapping=field_mapping,
        remove_duplicates=True,
    )
    
    logger.info(f"Ingestion complete!")
    logger.info(f"  Total loaded: {stats['total_loaded']}")
    logger.info(f"  Duplicates removed: {stats['duplicates_removed']}")
    logger.info(f"  Final count: {stats['final_count']}")
    
    return stats


def run_sample_ingestion():
    """
    Run ingestion on sample datasets.
    
    This demonstrates the full pipeline with the sample data included in the project.
    """
    logger.info("=" * 60)
    logger.info("TURKISH LEGAL RAG - DATA INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # Ingest sample QA dataset
    logger.info("\n1. Ingesting sample QA dataset...")
    sample_qa_path = Path('data/raw/sample_qa_dataset.json')
    if sample_qa_path.exists():
        ingest_qa_dataset(
            input_file=sample_qa_path,
            output_name='sample_qa_dataset',
            source_name='Turkish Legal QA v1',
        )
    else:
        logger.warning(f"Sample QA file not found: {sample_qa_path}")
    
    # Ingest sample legal text dataset
    logger.info("\n2. Ingesting sample legal text dataset...")
    sample_legal_path = Path('data/raw/sample_legal_texts.json')
    if sample_legal_path.exists():
        ingest_legal_text_dataset(
            input_file=sample_legal_path,
            output_name='sample_legal_texts',
            source_name='Turkish Legal Texts v1',
        )
    else:
        logger.warning(f"Sample legal file not found: {sample_legal_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Data ingestion complete! Check data/processed/ for results.")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Ingest and normalize Turkish legal datasets.'
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        help='Input file path (CSV, JSON, or JSONL)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='processed_data',
        help='Output name (without extension, will be .jsonl)'
    )
    parser.add_argument(
        '--type',
        choices=['qa', 'legal'],
        default='qa',
        help='Dataset type: QA pairs or legal texts'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='unknown',
        help='Name of the data source'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Run on sample data (default if no --input provided)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.input:
            # Ingest specific file
            if args.type == 'qa':
                ingest_qa_dataset(
                    input_file=args.input,
                    output_name=args.output,
                    source_name=args.source,
                )
            else:
                ingest_legal_text_dataset(
                    input_file=args.input,
                    output_name=args.output,
                    source_name=args.source,
                )
        else:
            # Run sample ingestion
            run_sample_ingestion()
    
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        exit(1)


if __name__ == '__main__':
    main()
