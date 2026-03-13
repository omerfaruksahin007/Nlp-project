"""Text normalization utilities for Turkish legal documents."""

import re
import unicodedata
from typing import Optional


def normalize_turkish_text(text: str) -> str:
    """
    Normalize Turkish text while preserving Turkish characters (ç, ğ, ı, ö, ş, ü, etc).
    
    Operations:
    - Strip leading/trailing whitespace
    - Normalize unicode
    - Remove multiple consecutive spaces
    - Remove form feeds and other control characters
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Cleaned and normalized text
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize to NFC form (canonical decomposition)
    text = unicodedata.normalize("NFC", text)
    
    # Remove control characters (but preserve newlines)
    text = "".join(char for char in text if unicodedata.category(char)[0] != "C" or char in "\n\r")
    
    # Replace multiple newlines with single newline
    text = re.sub(r"\n{2,}", "\n", text)
    
    # Strip each line and rejoin
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    # Remove multiple consecutive spaces
    text = re.sub(r"\s{2,}", " ", text)
    
    # Final strip
    text = text.strip()
    
    return text


def remove_duplicates(items: list, key_field: str = None) -> list:
    """
    Remove exact duplicates from a list of items.
    
    Args:
        items: List of dictionaries or strings
        key_field: If items are dicts, field to check for duplicates. If None, use entire dict.
        
    Returns:
        List with duplicates removed, preserving order
    """
    seen = set()
    result = []
    
    for item in items:
        if key_field:
            key = item.get(key_field, "")
        else:
            # For dicts, use json-serializable representation
            if isinstance(item, dict):
                key = tuple(sorted(item.items()))
            else:
                key = item
        
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result


def clean_qa_pair(question: Optional[str], answer: Optional[str]) -> tuple:
    """
    Clean question and answer pair.
    
    Args:
        question: Raw question text
        answer: Raw answer text
        
    Returns:
        Tuple of (cleaned_question, cleaned_answer)
    """
    question = normalize_turkish_text(question or "")
    answer = normalize_turkish_text(answer or "")
    
    return question, answer


def clean_legal_text(title: Optional[str], text: Optional[str]) -> tuple:
    """
    Clean title and legal text.
    
    Args:
        title: Document title
        text: Legal text content
        
    Returns:
        Tuple of (cleaned_title, cleaned_text)
    """
    title = normalize_turkish_text(title or "")
    text = normalize_turkish_text(text or "")
    
    return title, text
