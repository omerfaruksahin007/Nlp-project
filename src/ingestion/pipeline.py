"""
Ingestion pipeline orchestrator.

Coordinates loading, normalization, deduplication, and conversion of datasets.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .loader import load_data
from .normalizer import normalize_turkish_text, get_text_hash, is_empty_or_whitespace
from .converters import QAConverter, LegalTextConverter

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Main pipeline for data ingestion and normalization."""
    
    def __init__(self, output_dir: Path = Path('data/processed')):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_loaded': 0,
            'duplicates_removed': 0,
            'invalid_records': 0,
            'final_count': 0,
        }
    
    def ingest_qa_dataset(
        self,
        input_path: Path,
        output_name: str,
        source_name: str = 'unknown',
        field_mapping: Optional[Dict[str, str]] = None,
        remove_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest QA-style dataset.
        
        Args:
            input_path: Path to input file
            output_name: Name for output file (without extension)
            source_name: Name of the data source
            field_mapping: Mapping from schema fields to input fields
            remove_duplicates: Whether to remove duplicate records
            
        Returns:
            dict with ingestion statistics
        """
        logger.info(f"Starting QA dataset ingestion from {input_path}")
        
        # Load data
        records = load_data(input_path)
        self.stats['total_loaded'] = len(records)
        
        # Convert to schema
        converter = QAConverter(field_mapping)
        converted = converter.convert_batch(records, source_name)
        
        # Remove duplicates if requested
        if remove_duplicates:
            converted, dup_count = self._remove_duplicates_qa(converted)
            self.stats['duplicates_removed'] = dup_count
        
        # Save
        output_path = self.output_dir / f"{output_name}.jsonl"
        self._save_jsonl(converted, output_path)
        
        self.stats['final_count'] = len(converted)
        logger.info(f"QA ingestion complete. Stats: {self.stats}")
        
        return self.stats
    
    def ingest_legal_text_dataset(
        self,
        input_path: Path,
        output_name: str,
        source_name: str = 'unknown',
        field_mapping: Optional[Dict[str, str]] = None,
        remove_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest legal text dataset.
        
        Args:
            input_path: Path to input file
            output_name: Name for output file (without extension)
            source_name: Name of the data source
            field_mapping: Mapping from schema fields to input fields
            remove_duplicates: Whether to remove duplicate records
            
        Returns:
            dict with ingestion statistics
        """
        logger.info(f"Starting legal text ingestion from {input_path}")
        
        # Load data
        records = load_data(input_path)
        self.stats['total_loaded'] = len(records)
        
        # Convert to schema
        converter = LegalTextConverter(field_mapping)
        converted = converter.convert_batch(records, source_name)
        
        # Remove duplicates if requested
        if remove_duplicates:
            converted, dup_count = self._remove_duplicates_legal(converted)
            self.stats['duplicates_removed'] = dup_count
        
        # Save
        output_path = self.output_dir / f"{output_name}.jsonl"
        self._save_jsonl(converted, output_path)
        
        self.stats['final_count'] = len(converted)
        logger.info(f"Legal text ingestion complete. Stats: {self.stats}")
        
        return self.stats
    
    def _remove_duplicates_qa(self, records: List[Dict[str, str]]) -> tuple:
        """
        Remove duplicate QA records based on question+answer hash.
        
        Args:
            records: List of QA records
            
        Returns:
            (deduplicated_records, num_removed)
        """
        seen_hashes = set()
        unique_records = []
        duplicates = 0
        
        for record in records:
            # Create hash from question + answer
            combined = f"{record['question']} {record['answer']}"
            hash_val = get_text_hash(combined)
            
            if hash_val not in seen_hashes:
                seen_hashes.add(hash_val)
                unique_records.append(record)
            else:
                duplicates += 1
        
        logger.info(f"Removed {duplicates} duplicate QA records")
        return unique_records, duplicates
    
    def _remove_duplicates_legal(self, records: List[Dict[str, str]]) -> tuple:
        """
        Remove duplicate legal text records based on text hash.
        
        Args:
            records: List of legal text records
            
        Returns:
            (deduplicated_records, num_removed)
        """
        seen_hashes = set()
        unique_records = []
        duplicates = 0
        
        for record in records:
            hash_val = get_text_hash(record['text'])
            
            if hash_val not in seen_hashes:
                seen_hashes.add(hash_val)
                unique_records.append(record)
            else:
                duplicates += 1
        
        logger.info(f"Removed {duplicates} duplicate legal text records")
        return unique_records, duplicates
    
    def _save_jsonl(self, records: List[Dict[str, Any]], output_path: Path):
        """
        Save records as JSONL.
        
        Args:
            records: List of records to save
            output_path: Path to output file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(records)} records to {output_path}")
