#!/usr/bin/env python3
"""
Ingest real Turkish legal datasets.

This script demonstrates loading and processing:
1. HuggingFace: Renicames/turkish-lawchatbot
2. Kaggle: batuhankalem/turkishlaw-dataset-for-llm-finetuning

Usage:
    python ingest_turkish_datasets.py --hf              # Load HuggingFace dataset
    python ingest_turkish_datasets.py --kaggle <path>   # Load Kaggle dataset from folder
    python ingest_turkish_datasets.py --all <path>      # Load both (Kaggle path required)
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import load_huggingface_dataset, load_kaggle_dataset_from_folder, inspect_dataset_schema
from src.ingestion.converters import TurkishLawchatbotConverter, TurkishLawKaggleConverter
from src.ingestion.normalizer import get_text_hash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_output_dir():
    """Ensure output directory exists."""
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_jsonl(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Save data as JSONL."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} records to {output_path}")


def create_preview_file(data: List[Dict[str, Any]], output_path: Path, num_samples: int = 10) -> None:
    """Create a preview JSON file with sample records."""
    preview_data = {
        "total_records": len(data),
        "sample_size": min(num_samples, len(data)),
        "samples": data[:num_samples]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(preview_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created preview file: {output_path}")


def ingest_huggingface_lawchatbot():
    """Load and convert HuggingFace turkish-lawchatbot dataset."""
    logger.info("\n" + "="*80)
    logger.info("Loading HuggingFace dataset: Renicames/turkish-lawchatbot")
    logger.info("="*80)
    
    try:
        # Load raw data
        raw_data = load_huggingface_dataset(
            dataset_name='Renicames/turkish-lawchatbot',
            split='train'
        )
        
        if not raw_data:
            logger.error("Failed to load HuggingFace dataset")
            return None
        
        # Inspect schema
        logger.info(f"\nDataset schema inspection:")
        inspect_dataset_schema(raw_data, num_samples=2)
        
        # Convert to unified schema
        logger.info(f"\nConverting to unified QA schema...")
        converter = TurkishLawchatbotConverter()
        converted_data = converter.convert_batch(raw_data)
        
        # Save outputs
        output_dir = ensure_output_dir()
        
        # Save full dataset
        full_output = output_dir / 'huggingface_turkish_lawchatbot.jsonl'
        save_jsonl(converted_data, full_output)
        
        # Save preview
        preview_output = output_dir / 'huggingface_turkish_lawchatbot_preview.json'
        create_preview_file(converted_data, preview_output, num_samples=10)
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"HuggingFace Ingestion Summary:")
        logger.info(f"  Total records processed: {len(raw_data)}")
        logger.info(f"  Records kept: {len(converted_data)}")
        logger.info(f"  Records skipped: {len(raw_data) - len(converted_data)}")
        logger.info(f"  Output: {full_output}")
        logger.info(f"  Preview: {preview_output}")
        logger.info(f"{'='*80}\n")
        
        return converted_data
        
    except Exception as e:
        logger.error(f"Error ingesting HuggingFace dataset: {e}", exc_info=True)
        return None


def ingest_kaggle_turkishlaw(kaggle_path: str):
    """Load and convert Kaggle turkishlaw dataset."""
    logger.info("\n" + "="*80)
    logger.info(f"Loading Kaggle dataset from: {kaggle_path}")
    logger.info("="*80)
    
    kaggle_path = Path(kaggle_path)
    
    if not kaggle_path.exists():
        logger.error(f"Kaggle dataset path does not exist: {kaggle_path}")
        return None
    
    try:
        # Load raw data
        raw_data = load_kaggle_dataset_from_folder(str(kaggle_path))
        
        if not raw_data:
            logger.error("Failed to load Kaggle dataset")
            return None
        
        # Inspect schema
        logger.info(f"\nDataset schema inspection:")
        inspect_dataset_schema(raw_data, num_samples=2)
        
        # Convert to unified schema
        logger.info(f"\nConverting to unified QA schema...")
        converter = TurkishLawKaggleConverter()
        converted_data = converter.convert_batch(raw_data)
        
        # Save outputs
        output_dir = ensure_output_dir()
        
        # Save full dataset
        full_output = output_dir / 'kaggle_turkishlaw.jsonl'
        save_jsonl(converted_data, full_output)
        
        # Save preview
        preview_output = output_dir / 'kaggle_turkishlaw_preview.json'
        create_preview_file(converted_data, preview_output, num_samples=10)
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"Kaggle Ingestion Summary:")
        logger.info(f"  Total records processed: {len(raw_data)}")
        logger.info(f"  Records kept: {len(converted_data)}")
        logger.info(f"  Records skipped: {len(raw_data) - len(converted_data)}")
        logger.info(f"  Output: {full_output}")
        logger.info(f"  Preview: {preview_output}")
        logger.info(f"{'='*80}\n")
        
        return converted_data
        
    except Exception as e:
        logger.error(f"Error ingesting Kaggle dataset: {e}", exc_info=True)
        return None


def merge_and_save(hf_data: List[Dict], kg_data: List[Dict]) -> None:
    """Merge datasets and save combined version."""
    if not hf_data and not kg_data:
        logger.warning("No data to merge")
        return
    
    output_dir = ensure_output_dir()
    
    all_data = []
    if hf_data:
        all_data.extend(hf_data)
    if kg_data:
        all_data.extend(kg_data)
    
    # Save combined dataset
    combined_output = output_dir / 'combined_turkish_legal_qa.jsonl'
    save_jsonl(all_data, combined_output)
    
    # Find duplicates (by question lowercased)
    seen_questions = set()
    duplicates = 0
    for record in all_data:
        q_lower = record['question'].lower().strip()
        if q_lower in seen_questions:
            duplicates += 1
        else:
            seen_questions.add(q_lower)
    
    logger.info(f"\nCombined Dataset Summary:")
    logger.info(f"  Total records: {len(all_data)}")
    logger.info(f"  HuggingFace records: {len(hf_data) if hf_data else 0}")
    logger.info(f"  Kaggle records: {len(kg_data) if kg_data else 0}")
    logger.info(f"  Potential duplicates (same question): {duplicates}")
    logger.info(f"  Output: {combined_output}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Ingest Turkish legal datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load only HuggingFace dataset
  python ingest_turkish_datasets.py --hf
  
  # Load only Kaggle dataset
  python ingest_turkish_datasets.py --kaggle data/raw/kaggle_turkishlaw
  
  # Load both datasets
  python ingest_turkish_datasets.py --all data/raw/kaggle_turkishlaw
        """
    )
    
    parser.add_argument('--hf', action='store_true', help='Load HuggingFace dataset')
    parser.add_argument('--kaggle', type=str, help='Load Kaggle dataset from specified folder')
    parser.add_argument('--all', type=str, help='Load both HuggingFace and Kaggle (specify Kaggle path)')
    
    args = parser.parse_args()
    
    # Ensure we're running from the right directory
    if not Path('src/ingestion').exists():
        logger.error("Please run this script from the project root directory")
        sys.exit(1)
    
    hf_data = None
    kg_data = None
    
    # Load HuggingFace
    if args.hf or args.all:
        hf_data = ingest_huggingface_lawchatbot()
    
    # Load Kaggle
    if args.kaggle:
        kg_data = ingest_kaggle_turkishlaw(args.kaggle)
    elif args.all:
        if not args.all:
            logger.error("--all requires Kaggle path argument")
            sys.exit(1)
        kg_data = ingest_kaggle_turkishlaw(args.all)
    
    # Merge if both loaded
    if hf_data and kg_data:
        merge_and_save(hf_data, kg_data)
    
    # Log final status
    if not hf_data and not kg_data:
        logger.warning("No datasets were loaded. Use --hf, --kaggle, or --all")
        parser.print_help()
        sys.exit(1)
    
    logger.info("✓ Ingestion process completed successfully!")


if __name__ == '__main__':
    main()
