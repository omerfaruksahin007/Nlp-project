#!/usr/bin/env python3
"""
PROMPT 9 (Part 3): Citation Extraction and Formatting

Extracts and formats citations from generated answers.
Ensures consistency and traceability.

Features:
- Parse article numbers and law names
- Link citations back to source documents
- Validate citations against context
- Format citations for legal documents
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum


logger = logging.getLogger(__name__)


class CitationType(Enum):
    """Types of legal citations"""
    ARTICLE = "article"        # Madde
    SECTION = "section"        # Fıkra
    PARAGRAPH = "paragraph"    # Paragraf
    LAW = "law"                # Kanun
    REGULATION = "regulation"  # Yönetmelik
    DECREE = "decree"          # Kararname
    UNKNOWN = "unknown"


@dataclass
class Citation:
    """Single citation with metadata"""
    type: CitationType
    number: str                 # e.g., "141" or "5237"
    law_name: str              # e.g., "Türk Ceza Kanunu"
    full_text: Optional[str]   # Full citation text if available
    source_chunk_id: Optional[str]  # Which context chunk it came from
    confidence: float          # 0-1, how confident we are about this citation
    
    def __str__(self) -> str:
        """Format as string"""
        if self.type == CitationType.ARTICLE:
            return f"Madde {self.number}: {self.law_name}"
        elif self.type == CitationType.LAW:
            return self.law_name
        elif self.type == CitationType.SECTION:
            return f"Madde {self.number} - Fıkra {self.full_text}"
        else:
            return f"{self.type.value.title()} {self.number}: {self.law_name}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'type': self.type.value,
            'number': self.number,
            'law_name': self.law_name,
            'full_text': self.full_text,
            'source_chunk_id': self.source_chunk_id,
            'confidence': self.confidence
        }


class CitationExtractor:
    """
    Extracts citations from answer text.
    
    Patterns (Turkish legal citations):
    - "Madde 141" → Article 141
    - "5237 sayılı Türk Ceza Kanunu" → Law 5237 (Turkish Criminal Code)
    - "TCK Madde 50" → Criminal Code, Article 50
    - "Medeni Kanun Madde 1" → Civil Code, Article 1
    """
    
    # Common Turkish law abbreviations
    LAW_ABBREVIATIONS = {
        'TCK': 'Türk Ceza Kanunu',
        'MK': 'Medeni Kanun',
        'TK': 'Ticaret Kanunu',
        'ICEK': 'İcra ve İflas Kanunu',
        'CMK': 'Ceza Muhakemesi Kanunu',
        'HMK': 'Hukuk Muhakemesi Kanunı',
        'KVK': 'Kurumlar Vergisi Kanunu',
        'GVK': 'Gelir Vergisi Kanunu',
        'KDV': 'Katma Değer Vergisi',
        'İK': 'İş Kanunu',
        'SGK': 'Sosyal Güvenlik Kanunu',
    }
    
    # Patterns for citation detection
    PATTERNS = {
        'article_with_law': r'(?:Türk\s+)?([A-Za-z\s]+?)\s+(?:Madde|M\.)\s+(\d+)(?:/(\d+))?',
        'abbreviated_article': r'([A-Z]{2,4})\s+(?:Madde|M\.)\s+(\d+)',
        'law_number': r'(\d+)\s+sayılı\s+([A-Za-z\s]+)',
        'standalone_madde': r'Madde\s+(\d+)',
    }
    
    def __init__(self):
        """Initialize extractor"""
        self.logger = logging.getLogger(__name__)
    
    def extract_citations(
        self,
        text: str,
        context_chunks: Optional[List[Dict]] = None
    ) -> List[Citation]:
        """
        Extract all citations from text.
        
        Args:
            text: Answer text that may contain citations
            context_chunks: Optional context to validate citations against
        
        Returns:
            List of Citation objects
        """
        citations = []
        seen = set()  # Track unique citations
        
        # Try each pattern
        for pattern_name, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                citation = self._parse_match(pattern_name, match, text)
                if citation and citation.number not in seen:
                    citations.append(citation)
                    seen.add(citation.number)
        
        # Validate against context if provided
        if context_chunks:
            citations = self._validate_citations(citations, context_chunks)
        
        logger.info(f"Extracted {len(citations)} citations")
        return citations
    
    def _parse_match(self, pattern_name: str, match, text: str) -> Optional[Citation]:
        """Parse regex match into Citation object"""
        groups = match.groups()
        
        try:
            if pattern_name == 'article_with_law':
                law_name = groups[0].strip()
                article_num = groups[1]
                section_num = groups[2]
                
                return Citation(
                    type=CitationType.ARTICLE,
                    number=article_num,
                    law_name=law_name,
                    full_text=f"Fıkra {section_num}" if section_num else None,
                    source_chunk_id=None,
                    confidence=0.9
                )
            
            elif pattern_name == 'abbreviated_article':
                abbrev = groups[0].upper()
                article_num = groups[1]
                law_name = self.LAW_ABBREVIATIONS.get(abbrev, abbrev)
                
                return Citation(
                    type=CitationType.ARTICLE,
                    number=article_num,
                    law_name=law_name,
                    source_chunk_id=None,
                    confidence=0.85
                )
            
            elif pattern_name == 'law_number':
                law_num = groups[0]
                law_name = groups[1].strip()
                
                return Citation(
                    type=CitationType.LAW,
                    number=law_num,
                    law_name=law_name,
                    source_chunk_id=None,
                    confidence=0.95
                )
            
            elif pattern_name == 'standalone_madde':
                article_num = groups[0]
                
                # Try to infer law from context
                law_name = "Kanun"  # Default
                
                # Look backward in text for law name
                start = max(0, match.start() - 100)
                context_text = text[start:match.start()]
                
                for abbrev, full_name in self.LAW_ABBREVIATIONS.items():
                    if abbrev in context_text:
                        law_name = full_name
                        break
                
                return Citation(
                    type=CitationType.ARTICLE,
                    number=article_num,
                    law_name=law_name,
                    source_chunk_id=None,
                    confidence=0.7
                )
        
        except Exception as e:
            logger.debug(f"Failed to parse match: {e}")
            return None
    
    def _validate_citations(
        self,
        citations: List[Citation],
        context_chunks: List[Dict]
    ) -> List[Citation]:
        """
        Validate citations against context chunks.
        Increase confidence if found in context, mark if not found.
        """
        validated = []
        
        # Build index of articles mentioned in context
        context_articles = self._index_context_articles(context_chunks)
        
        for citation in citations:
            # Look for this article in context
            for chunk_id, articles_in_chunk in context_articles.items():
                if citation.number in articles_in_chunk:
                    citation.source_chunk_id = chunk_id
                    citation.confidence = min(0.99, citation.confidence + 0.1)
                    break
            else:
                # Not found in context, lower confidence
                citation.confidence *= 0.7
            
            validated.append(citation)
        
        return validated
    
    def _index_context_articles(self, chunks: List[Dict]) -> Dict[str, Set[str]]:
        """
        Index articles mentioned in each chunk.
        
        Returns:
            {chunk_id: set_of_article_numbers}
        """
        index = {}
        
        for chunk in chunks:
            chunk_id = chunk.get('id') or chunk.get('chunk_id', 'unknown')
            text = chunk.get('text', chunk.get('chunk_text', ''))
            
            # Find all article numbers
            article_matches = re.findall(r'(?:Madde|M\.)\s+(\d+)', text)
            index[chunk_id] = set(article_matches)
        
        return index


class CitationFormatter:
    """
    Formats citations for output.
    
    Styles:
    - inline: "Madde 141, Türk Ceza Kanunu"
    - harvard: "Kanunu, 5237, Madde 141"
    - footnote: "[1]" with footnote list
    - full_reference: Complete APA-style citation
    """
    
    @staticmethod
    def format_inline(citation: Citation) -> str:
        """Inline format"""
        if citation.type == CitationType.ARTICLE:
            base = f"Madde {citation.number}, {citation.law_name}"
            if citation.full_text:
                base += f" ({citation.full_text})"
            return base
        else:
            return str(citation)
    
    @staticmethod
    def format_harvard(citation: Citation) -> str:
        """Harvard-style format"""
        if citation.type == CitationType.LAW:
            return f"{citation.law_name}, {citation.number}"
        else:
            return f"{citation.law_name}, Madde {citation.number}"
    
    @staticmethod
    def format_footnote(citations: List[Citation]) -> Tuple[str, List[str]]:
        """
        Format as footnotes.
        
        Returns:
            (inline text with [1][2] markers, list of footnote texts)
        """
        footnotes = [CitationFormatter.format_inline(c) for c in citations]
        return "", footnotes
    
    @staticmethod
    def format_apa(citation: Citation) -> str:
        """APA-style format for legal citations"""
        # Simplified APA for Turkish legal documents
        if citation.type == CitationType.LAW:
            return f"{citation.law_name}. ({citation.number})"
        else:
            return f"{citation.law_name}. (Madde {citation.number})"


class AnswerWithCitations:
    """
    Answer with extracted and validated citations.
    """
    
    def __init__(
        self,
        original_answer: str,
        citations: List[Citation],
        confidence_level: str = "medium"
    ):
        """Initialize with answer and citations"""
        self.original_answer = original_answer
        self.citations = citations
        self.confidence_level = confidence_level  # high/medium/low/insufficient
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'answer': self.original_answer,
            'citations': [c.to_dict() for c in self.citations],
            'num_citations': len(self.citations),
            'confidence': self.confidence_level
        }
    
    def format_with_citations(self, style: str = 'inline') -> str:
        """
        Format answer with citations.
        
        Args:
            style: 'inline', 'harvard', 'footnote', 'apa'
        
        Returns:
            Formatted answer text
        """
        if style == 'inline':
            formatted_citations = [
                CitationFormatter.format_inline(c) for c in self.citations
            ]
        elif style == 'harvard':
            formatted_citations = [
                CitationFormatter.format_harvard(c) for c in self.citations
            ]
        elif style == 'apa':
            formatted_citations = [
                CitationFormatter.format_apa(c) for c in self.citations
            ]
        else:
            formatted_citations = [str(c) for c in self.citations]
        
        # Build formatted text
        text_parts = [
            self.original_answer,
            "",
            "───────────────────────────────",
            "KAYNAKLAR:",
        ]
        
        for i, citation in enumerate(formatted_citations, 1):
            text_parts.append(f"{i}. {citation}")
        
        return "\n".join(text_parts)


if __name__ == '__main__':
    # Example usage
    extractor = CitationExtractor()
    
    sample_answer = """
    Türk Ceza Kanunu Madde 141'e göre, başkasına ait bir malı, sahibini haksız yere yoksun etmek
    amacıyla alan kişi hırsızlık suçunu işlemiş olur. TCK Madde 142'de ise nitelikli hırsızlığın
    cezası belirtilmiştir. Medeni Kanun Madde 1'de ise kişiliye ilişkin hükümler yer almaktadır.
    """
    
    citations = extractor.extract_citations(sample_answer)
    
    print(f"Found {len(citations)} citations:")
    for c in citations:
        print(f"  - {c}")
