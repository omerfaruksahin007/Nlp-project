#!/usr/bin/env python3
"""
Demo script to test the Turkish legal dataset ingestion pipeline.

This script loads the example dataset files and demonstrates the ingestion process.

Usage:
    python ingest_demo.py
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import load_data, inspect_dataset_schema
from src.ingestion.converters import QAConverter, TurkishLawKaggleConverter
from src.ingestion.normalizer import normalize_turkish_text

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


def demo_ingest_example_lawchatbot():
    """Demo: Load and convert example lawchatbot dataset."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Loading Example Turkish Lawchatbot Dataset")
    logger.info("="*80)
    
    example_file = Path('data/raw/example_lawchatbot.json')
    
    if not example_file.exists():
        logger.error(f"Example file not found: {example_file}")
        return None
    
    try:
        # Load raw data
        logger.info(f"Loading data from: {example_file}")
        raw_data = load_data(example_file)
        logger.info(f"Loaded {len(raw_data)} records")
        
        # Inspect schema
        logger.info(f"\nDataset Schema:")
        inspect_dataset_schema(raw_data, num_samples=1)
        
        # Convert to unified schema
        logger.info(f"\nConverting to unified QA schema...")
        converter = QAConverter(field_mapping={
            'question': 'question',
            'answer': 'answer',
            'category': 'category'
        })
        converted_data = converter.convert_batch(raw_data, source_name='example:lawchatbot')
        
        # Save output
        output_dir = ensure_output_dir()
        output_file = output_dir / 'example_lawchatbot_processed.jsonl'
        save_jsonl(converted_data, output_file)
        
        # Display sample output
        logger.info(f"\nSample of converted data:")
        if converted_data:
            for i, record in enumerate(converted_data[:2]):
                logger.info(f"\nRecord {i+1}:")
                logger.info(f"  ID: {record['id']}")
                logger.info(f"  Question: {record['question'][:80]}...")
                logger.info(f"  Answer: {record['answer'][:80]}...")
                logger.info(f"  Source: {record['source']}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Lawchatbot Processing Summary:")
        logger.info(f"  Input records: {len(raw_data)}")
        logger.info(f"  Output records: {len(converted_data)}")
        logger.info(f"  Output file: {output_file}")
        logger.info(f"{'='*80}\n")
        
        return converted_data
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return None


def demo_ingest_example_kaggle():
    """Demo: Load and convert example Kaggle dataset."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Loading Example Kaggle Turkish Law Dataset")
    logger.info("="*80)
    
    example_file = Path('data/raw/example_kaggle_turkishlaw.json')
    
    if not example_file.exists():
        logger.error(f"Example file not found: {example_file}")
        return None
    
    try:
        # Load raw data
        logger.info(f"Loading data from: {example_file}")
        raw_data = load_data(example_file)
        logger.info(f"Loaded {len(raw_data)} records")
        
        # Inspect schema
        logger.info(f"\nDataset Schema:")
        inspect_dataset_schema(raw_data, num_samples=1)
        
        # Convert to unified schema
        logger.info(f"\nConverting to unified QA schema...")
        converter = TurkishLawKaggleConverter()
        converted_data = converter.convert_batch(raw_data)
        
        # Save output
        output_dir = ensure_output_dir()
        output_file = output_dir / 'example_kaggle_turkishlaw_processed.jsonl'
        save_jsonl(converted_data, output_file)
        
        # Display sample output
        logger.info(f"\nSample of converted data:")
        if converted_data:
            for i, record in enumerate(converted_data[:2]):
                logger.info(f"\nRecord {i+1}:")
                logger.info(f"  ID: {record['id']}")
                logger.info(f"  Question: {record['question'][:80]}...")
                logger.info(f"  Answer: {record['answer'][:80]}...")
                logger.info(f"  Source: {record['source']}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Kaggle Processing Summary:")
        logger.info(f"  Input records: {len(raw_data)}")
        logger.info(f"  Output records: {len(converted_data)}")
        logger.info(f"  Output file: {output_file}")
        logger.info(f"{'='*80}\n")
        
        return converted_data
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return None


def demo_text_normalization():
    """Demo: Show text normalization in action."""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Turkish Text Normalization")
    logger.info("="*80)
    
    test_texts = [
        "  Türkiye'de   hukuk   sistemi    nasıl çalışır?  ",
        'Medenî Kanun "Madde 23"  ve  İş Kanunu\'nda  yazsalı...',
        "Üniversite hukuk fakültesi öğrencilerine tavsiyeler.",
    ]
    
    for text in test_texts:
        normalized = normalize_turkish_text(text)
        logger.info(f"\nOriginal:  {repr(text)}")
        logger.info(f"Normalized: {repr(normalized)}")


def main():
    """Run all demos."""
    logger.info("="*80)
    logger.info("Turkish Legal Dataset Ingestion Pipeline - DEMO")
    logger.info("="*80)
    
    # Check if running from correct directory
    if not Path('src/ingestion').exists():
        logger.error("Please run this script from the project root directory")
        sys.exit(1)
    
    # Run demos
    demo_text_normalization()
    lawchatbot_data = demo_ingest_example_lawchatbot()
    kaggle_data = demo_ingest_example_kaggle()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info("Output files saved to: data/processed/")
    logger.info("  - example_lawchatbot_processed.jsonl")
    logger.info("  - example_kaggle_turkishlaw_processed.jsonl")
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()
