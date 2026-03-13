"""
Schema converters for normalizing datasets into unified formats.

Supports QA datasets and legal text datasets with flexible field mapping.
"""

import logging
from typing import Dict, List, Any, Optional
from uuid import uuid4

from .normalizer import normalize_turkish_text, is_empty_or_whitespace

logger = logging.getLogger(__name__)


class QAConverter:
    """Converts datasets to unified QA schema."""
    
    QA_SCHEMA = {
        'id': str,
        'question': str,
        'answer': str,
        'source': str,
        'category': str,
        'citation': str,
    }
    
    def __init__(self, field_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize converter with optional field mappings.
        
        Args:
            field_mapping: Map from schema fields to source field names
                          e.g., {'question': 'q', 'answer': 'a', 'source': 'origin'}
        """
        self.field_mapping = field_mapping or {}
    
    def convert_record(self, record: Dict[str, Any], source_name: str = 'unknown') -> Optional[Dict[str, str]]:
        """
        Convert a single record to QA schema.
        
        Args:
            record: Input record (dict)
            source_name: Name of data source
            
        Returns:
            Normalized record or None if invalid
        """
        try:
            # Extract fields using mappings
            question = self._extract_field(record, 'question')
            answer = self._extract_field(record, 'answer')
            
            # Both question and answer are required
            if is_empty_or_whitespace(question) or is_empty_or_whitespace(answer):
                logger.debug("Skipping record with missing question or answer")
                return None
            
            # Create normalized record
            qa_record = {
                'id': self._extract_field(record, 'id') or str(uuid4()),
                'question': normalize_turkish_text(question),
                'answer': normalize_turkish_text(answer),
                'source': self._extract_field(record, 'source') or source_name,
                'category': self._extract_field(record, 'category') or '',
                'citation': self._extract_field(record, 'citation') or '',
            }
            
            return qa_record
        except Exception as e:
            logger.warning(f"Failed to convert QA record: {e}")
            return None
    
    def _extract_field(self, record: Dict[str, Any], schema_field: str) -> Optional[str]:
        """
        Extract field from record using mapping or direct access.
        
        Args:
            record: Input record
            schema_field: Field name in schema
            
        Returns:
            Field value or None
        """
        # Use mapping if available
        source_field = self.field_mapping.get(schema_field, schema_field)
        
        # Try to find the field in record (case-insensitive)
        value = record.get(source_field)
        if value is not None:
            return str(value)
        
        # Try case-insensitive search
        for key, val in record.items():
            if key.lower() == source_field.lower():
                return str(val)
        
        return None
    
    def convert_batch(self, records: List[Dict[str, Any]], source_name: str = 'unknown') -> List[Dict[str, str]]:
        """
        Convert multiple records to QA schema.
        
        Args:
            records: List of input records
            source_name: Name of data source
            
        Returns:
            List of normalized records
        """
        converted = []
        for record in records:
            converted_record = self.convert_record(record, source_name)
            if converted_record:
                converted.append(converted_record)
        
        logger.info(f"Converted {len(converted)}/{len(records)} records to QA schema")
        return converted


class LegalTextConverter:
    """Converts datasets to unified legal text schema."""
    
    LEGAL_SCHEMA = {
        'doc_id': str,
        'title': str,
        'law_name': str,
        'article_no': str,
        'section': str,
        'text': str,
        'source': str,
    }
    
    def __init__(self, field_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize converter with optional field mappings.
        
        Args:
            field_mapping: Map from schema fields to source field names
        """
        self.field_mapping = field_mapping or {}
    
    def convert_record(self, record: Dict[str, Any], source_name: str = 'unknown') -> Optional[Dict[str, str]]:
        """
        Convert a single record to legal text schema.
        
        Args:
            record: Input record (dict)
            source_name: Name of data source
            
        Returns:
            Normalized record or None if invalid
        """
        try:
            # Text is required
            text = self._extract_field(record, 'text')
            if is_empty_or_whitespace(text):
                logger.debug("Skipping legal text record with missing text field")
                return None
            
            # Create normalized record
            legal_record = {
                'doc_id': self._extract_field(record, 'doc_id') or str(uuid4()),
                'title': normalize_turkish_text(self._extract_field(record, 'title') or ''),
                'law_name': normalize_turkish_text(self._extract_field(record, 'law_name') or ''),
                'article_no': self._extract_field(record, 'article_no') or '',
                'section': normalize_turkish_text(self._extract_field(record, 'section') or ''),
                'text': normalize_turkish_text(text),
                'source': self._extract_field(record, 'source') or source_name,
            }
            
            return legal_record
        except Exception as e:
            logger.warning(f"Failed to convert legal text record: {e}")
            return None
    
    def _extract_field(self, record: Dict[str, Any], schema_field: str) -> Optional[str]:
        """
        Extract field from record using mapping or direct access.
        
        Args:
            record: Input record
            schema_field: Field name in schema
            
        Returns:
            Field value or None
        """
        # Use mapping if available
        source_field = self.field_mapping.get(schema_field, schema_field)
        
        # Try to find the field in record (case-insensitive)
        value = record.get(source_field)
        if value is not None:
            return str(value)
        
        # Try case-insensitive search
        for key, val in record.items():
            if key.lower() == source_field.lower():
                return str(val)
        
        return None
    
    def convert_batch(self, records: List[Dict[str, Any]], source_name: str = 'unknown') -> List[Dict[str, str]]:
        """
        Convert multiple records to legal text schema.
        
        Args:
            records: List of input records
            source_name: Name of data source
            
        Returns:
            List of normalized records
        """
        converted = []
        for record in records:
            converted_record = self.convert_record(record, source_name)
            if converted_record:
                converted.append(converted_record)
        
        logger.info(f"Converted {len(converted)}/{len(records)} records to legal text schema")
        return converted


# ============================================================================
# Dataset-Specific Converters for Real Datasets
# ============================================================================

class TurkishLawchatbotConverter(QAConverter):
    """
    Specialized converter for Renicames/turkish-lawchatbot dataset.
    
    Expected structure:
    - question: str
    - answer: str
    - category: str (optional)
    
    This dataset from HuggingFace contains Turkish legal Q&A pairs.
    """
    
    def __init__(self):
        """Initialize with HuggingFace dataset field names."""
        # Default mapping for this dataset
        super().__init__(field_mapping={
            'question': 'question',
            'answer': 'answer',
            'category': 'category',
        })
    
    def convert_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Convert batch with dataset-specific logging.
        
        Args:
            records: List of raw records from HuggingFace
            
        Returns:
            List of QA records in unified schema
        """
        logger.info(f"Converting {len(records)} records from HuggingFace turkish-lawchatbot")
        conventional = super().convert_batch(records, source_name='huggingface:turkish-lawchatbot')
        
        # Log statistics
        skipped_count = len(records) - len(conventional)
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} records (missing question or answer)")
        
        return conventional


class TurkishLawKaggleConverter(QAConverter):
    """
    Specialized converter for batuhankalem/turkishlaw-dataset-for-llm-finetuning.
    
    This Kaggle dataset may have different column names depending on the structure.
    Common patterns: 'instruction'/'input'/'output' or 'question'/'answer'
    
    This converter tries to detect the column names flexibly.
    """
    
    def __init__(self):
        """Initialize with flexible field mapping for Kaggle dataset."""
        super().__init__(field_mapping={})
    
    def convert_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Convert batch with flexible column detection.
        
        Args:
            records: List of raw records from Kaggle
            
        Returns:
            List of QA records in unified schema
        """
        logger.info(f"Converting {len(records)} records from Kaggle turkishlaw")
        
        # First, detect the column structure
        if records:
            self._detect_columns(records[0])
        
        # Convert using detected columns
        conventional = []
        for record in records:
            converted_record = self.convert_record(record, source_name='kaggle:turkishlaw')
            if converted_record:
                conventional.append(converted_record)
        
        logger.info(f"Converted {len(conventional)}/{len(records)} records to QA schema")
        skipped_count = len(records) - len(conventional)
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} records (missing question or answer)")
        
        return conventional
    
    def _detect_columns(self, sample_record: Dict[str, Any]) -> None:
        """
        Detect question and answer columns from sample record.
        
        Args:
            sample_record: First record from dataset
        """
        record_keys = [k.lower() for k in sample_record.keys()]
        
        # Try to find question column
        question_candidates = ['question', 'instruction', 'prompt', 'soru', 'q']
        for cand in question_candidates:
            if cand in record_keys:
                self.field_mapping['question'] = cand
                logger.info(f"Detected question column: '{cand}'")
                break
        
        # Try to find answer column
        answer_candidates = ['answer', 'output', 'response', 'cevap', 'a']
        for cand in answer_candidates:
            if cand in record_keys:
                self.field_mapping['answer'] = cand
                logger.info(f"Detected answer column: '{cand}'")
                break
        
        # If no mapping found, log warning
        if 'question' not in self.field_mapping:
            logger.warning("Could not auto-detect question column. Available columns: " + 
                          str(list(sample_record.keys())))
        if 'answer' not in self.field_mapping:
            logger.warning("Could not auto-detect answer column. Available columns: " + 
                          str(list(sample_record.keys())))
