"""Unified data schemas for QA and legal documents."""

from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class QAPair:
    """Unified schema for question-answer pairs."""
    
    id: str
    question: str
    answer: str
    source: str  # kaggle, huggingface, etc
    category: Optional[str] = None
    citation: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LegalDocument:
    """Unified schema for legal documents/statutes."""
    
    doc_id: str
    title: str
    text: str
    source: str
    law_name: Optional[str] = None
    article_no: Optional[str] = None
    section: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def validate_qa_pair(qa: QAPair) -> bool:
    """
    Validate that a QA pair has required fields.
    
    Args:
        qa: QAPair object
        
    Returns:
        True if valid, False otherwise
    """
    return (
        qa.id and 
        qa.question and 
        qa.answer and 
        qa.source
    )


def validate_legal_document(doc: LegalDocument) -> bool:
    """
    Validate that a legal document has required fields.
    
    Args:
        doc: LegalDocument object
        
    Returns:
        True if valid, False otherwise
    """
    return (
        doc.doc_id and 
        doc.title and 
        doc.text and 
        doc.source
    )
