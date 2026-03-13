"""
Turkish text normalization module.

Handles careful normalization while preserving Turkish characters.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Turkish-specific characters
TURKISH_CHARS = set('çğıöşüÇĞİÖŞÜ')


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: remove extra spaces, tabs, newlines.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def remove_control_characters(text: str) -> str:
    """
    Remove control characters while preserving printable text.
    
    Args:
        text: Input text
        
    Returns:
        Text with control characters removed
    """
    # Keep letters, digits, punctuation, spaces, and Turkish characters
    cleaned = ''.join(char for char in text if char.isprintable() or char in TURKISH_CHARS)
    return cleaned


def normalize_quotes(text: str) -> str:
    """
    Normalize different types of quotes to standard ASCII quotes.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized quotes
    """
    # Replace various quote styles with standard ASCII using string replacement
    # Double quotes: "", „, ‟ -> "
    text = text.replace('"', '"')  # U+201C LEFT DOUBLE QUOTATION MARK
    text = text.replace('"', '"')  # U+201D RIGHT DOUBLE QUOTATION MARK
    text = text.replace('„', '"')  # U+201E DOUBLE LOW-9 QUOTATION MARK
    text = text.replace('‟', '"')  # U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    text = text.replace('«', '"')  # U+00AB LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    text = text.replace('»', '"')  # U+00BB RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    
    # Single quotes: '', ', ` -> '
    text = text.replace(''', "'")  # U+2018 LEFT SINGLE QUOTATION MARK
    text = text.replace(''', "'")  # U+2019 RIGHT SINGLE QUOTATION MARK
    text = text.replace('`', "'")  # U+0060 GRAVE ACCENT
    
    return text


def normalize_turkish_text(text: Optional[str], preserve_case: bool = True) -> str:
    """
    Normalize Turkish text carefully without breaking Turkish characters.
    
    Process:
    1. Remove control characters
    2. Normalize whitespace
    3. Normalize quotes
    4. Optionally lowercase (keeping Turkish chars intact)
    
    Args:
        text: Input text (can be None)
        preserve_case: If False, convert to lowercase (Turkish-aware)
        
    Returns:
        Normalized text
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # Step 1: Remove control characters
    text = remove_control_characters(text)
    
    # Step 2: Normalize whitespace
    text = normalize_whitespace(text)
    
    # Step 3: Normalize quotes
    text = normalize_quotes(text)
    
    # Step 4: Lowercase if needed (Turkish-aware)
    if not preserve_case:
        # Python's lower() works fine with Turkish characters
        text = text.lower()
    
    return text


def get_text_hash(text: str) -> str:
    """
    Create a simple hash for duplicate detection.
    
    Uses normalized text to detect semantic duplicates.
    
    Args:
        text: Input text
        
    Returns:
        Hash string (first 16 characters of MD5)
    """
    import hashlib
    normalized = normalize_turkish_text(text, preserve_case=False)
    # Remove extra punctuation and spaces for deduplication
    simplified = re.sub(r'[^\w\s]', '', normalized)
    simplified = re.sub(r'\s+', ' ', simplified).strip()
    
    hash_obj = hashlib.md5(simplified.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def is_empty_or_whitespace(text: Optional[str]) -> bool:
    """
    Check if text is None, empty, or only whitespace.
    
    Args:
        text: Input text
        
    Returns:
        True if text is empty or whitespace only
    """
    return text is None or isinstance(text, str) and not text.strip()
