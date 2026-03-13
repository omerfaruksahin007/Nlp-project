"""Data loaders for different sources."""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


def load_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of dictionaries
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
        logger.info(f"Loaded {len(data)} rows from CSV: {filepath}")
    except Exception as e:
        logger.error(f"Error loading CSV {filepath}: {e}")
    
    return data


def load_json_lines(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file (one JSON object per line).
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")
        logger.info(f"Loaded {len(data)} rows from JSONL: {filepath}")
    except Exception as e:
        logger.error(f"Error loading JSONL {filepath}: {e}")
    
    return data


def load_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from JSON file (expects array of objects).
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of dictionaries
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            # Handle both array and single object
            if isinstance(content, list):
                data = content
            else:
                data = [content]
        logger.info(f"Loaded {len(data)} rows from JSON: {filepath}")
    except Exception as e:
        logger.error(f"Error loading JSON {filepath}: {e}")
    
    return data


def load_huggingface_dataset(dataset_id: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load data from Hugging Face Hub.
    
    Args:
        dataset_id: Dataset identifier (e.g., "Renicames/turkish-lawchatbot")
        split: Dataset split to load
        
    Returns:
        List of dictionaries with dataset rows
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        return []
    
    data = []
    try:
        dataset = load_dataset(dataset_id, split=split)
        data = [dict(row) for row in dataset]
        logger.info(f"Loaded {len(data)} rows from HuggingFace: {dataset_id}/{split}")
        logger.info(f"Columns: {list(dataset.column_names)}")
    except Exception as e:
        logger.error(f"Error loading HuggingFace dataset {dataset_id}: {e}")
    
    return data


def load_kaggle_dataset_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load Kaggle dataset from local folder by reading all CSV/JSON files.
    
    Args:
        folder_path: Path to folder containing dataset files
        
    Returns:
        Combined list of dictionaries from all files
    """
    data = []
    folder = Path(folder_path)
    
    if not folder.exists():
        logger.error(f"Folder does not exist: {folder_path}")
        return data
    
    # Load all CSV files
    for csv_file in folder.glob("*.csv"):
        logger.info(f"Loading CSV file: {csv_file.name}")
        csv_data = load_csv(str(csv_file))
        data.extend(csv_data)
    
    # Load all JSON files
    for json_file in folder.glob("*.json"):
        logger.info(f"Loading JSON file: {json_file.name}")
        json_data = load_json(str(json_file))
        data.extend(json_data)
    
    # Load all JSONL files
    for jsonl_file in folder.glob("*.jsonl"):
        logger.info(f"Loading JSONL file: {jsonl_file.name}")
        jsonl_data = load_json_lines(str(jsonl_file))
        data.extend(jsonl_data)
    
    logger.info(f"Total rows loaded from {folder_path}: {len(data)}")
    return data


def inspect_dataset_schema(data: List[Dict[str, Any]], num_samples: int = 3) -> None:
    """
    Print schema information about a dataset.
    
    Args:
        data: List of dictionaries
        num_samples: Number of sample rows to display
    """
    if not data:
        logger.info("Dataset is empty")
        return
    
    # Get all unique keys across all items
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    logger.info(f"\n=== Dataset Schema ===")
    logger.info(f"Total rows: {len(data)}")
    logger.info(f"Total columns: {len(all_keys)}")
    logger.info(f"Columns: {sorted(all_keys)}")
    
    logger.info(f"\n=== Sample rows (first {min(num_samples, len(data))}) ===")
    for i, row in enumerate(data[:num_samples]):
        logger.info(f"\nRow {i+1}:")
        for key, value in row.items():
            # Truncate long values for readability
            val_str = str(value)[:100]
            if len(str(value)) > 100:
                val_str += "..."
            logger.info(f"  {key}: {val_str}")
