"""
Data loader module for reading datasets from various formats.

Supports: CSV, JSON, JSONL, and Hugging Face datasets.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments to pass to pandas.read_csv
        
    Returns:
        List of dictionaries, one per row
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to load CSV from {file_path}: {e}")
        raise


def load_json(file_path: Union[str, Path]) -> Union[List[Dict], Dict]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data (list or dict)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {file_path}")
        
        # If it's a dict, try to extract a list of records
        if isinstance(data, dict):
            # Look for common keys that might contain a list
            for key in ['data', 'records', 'items', 'samples']:
                if key in data and isinstance(data[key], list):
                    logger.info(f"Extracted '{key}' field from JSON")
                    return data[key]
        
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        raise


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file (one JSON object per line).
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    try:
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
        
        logger.info(f"Loaded {len(records)} records from {file_path}")
        return records
    except Exception as e:
        logger.error(f"Failed to load JSONL from {file_path}: {e}")
        raise


def load_huggingface_dataset(dataset_name: str, split: str = 'train', **kwargs) -> List[Dict[str, Any]]:
    """
    Load data from Hugging Face datasets library.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'wikiqa', 'squad')
        split: Which split to load ('train', 'test', 'validation')
        **kwargs: Additional arguments to pass to load_dataset
        
    Returns:
        List of dictionaries
    """
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        logger.info(f"Loaded {len(dataset)} samples from HuggingFace dataset '{dataset_name}'")
        
        # Convert to list of dicts
        return dataset.to_dict(orient='records') if hasattr(dataset, 'to_dict') else [dict(item) for item in dataset]
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        raise
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset '{dataset_name}': {e}")
        raise


def load_data(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Auto-detect file format and load data.
    
    Args:
        file_path: Path to file
        
    Returns:
        List of dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.csv':
        return load_csv(file_path)
    elif suffix == '.json':
        result = load_json(file_path)
        return result if isinstance(result, list) else [result]
    elif suffix == '.jsonl':
        return load_jsonl(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_kaggle_dataset_from_folder(folder_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load Kaggle dataset from local folder by reading all CSV/JSON/JSONL files.
    
    Args:
        folder_path: Path to folder containing dataset files
        
    Returns:
        Combined list of dictionaries from all files
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        logger.error(f"Folder does not exist: {folder_path}")
        return []
    
    all_data = []
    
    # Load all CSV files
    for csv_file in folder.glob("*.csv"):
        logger.info(f"Loading CSV file: {csv_file.name}")
        try:
            csv_data = load_csv(csv_file)
            all_data.extend(csv_data)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")
    
    # Load all JSON files
    for json_file in folder.glob("*.json"):
        logger.info(f"Loading JSON file: {json_file.name}")
        try:
            json_data = load_json(json_file)
            if isinstance(json_data, list):
                all_data.extend(json_data)
            else:
                all_data.append(json_data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file.name}: {e}")
    
    # Load all JSONL files
    for jsonl_file in folder.glob("*.jsonl"):
        logger.info(f"Loading JSONL file: {jsonl_file.name}")
        try:
            jsonl_data = load_jsonl(jsonl_file)
            all_data.extend(jsonl_data)
        except Exception as e:
            logger.warning(f"Failed to load {jsonl_file.name}: {e}")
    
    logger.info(f"Total rows loaded from {folder_path}: {len(all_data)}")
    return all_data


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
        if isinstance(item, dict):
            all_keys.update(item.keys())
    
    logger.info(f"\n=== Dataset Schema ===")
    logger.info(f"Total rows: {len(data)}")
    logger.info(f"Total columns: {len(all_keys)}")
    logger.info(f"Columns: {sorted(all_keys)}")
    
    logger.info(f"\n=== Sample rows (first {min(num_samples, len(data))}) ===")
    for i, row in enumerate(data[:num_samples]):
        if isinstance(row, dict):
            logger.info(f"\nRow {i+1}:")
            for key, value in row.items():
                # Truncate long values for readability
                val_str = str(value)[:100]
                if len(str(value)) > 100:
                    val_str += "..."
                logger.info(f"  {key}: {val_str}")
