#!/usr/bin/env python3
"""
PROMPT 11: Citation Evaluation Module
Evaluate citation accuracy against gold references
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re

@dataclass
class CitationEvalResult:
    """Citation evaluation result"""
    precision: float  # Fraction of cited sources that are correct
    recall: float  # Fraction of gold citations that were cited
    f1: float  # Harmonic mean
    num_correct: int
    num_predicted: int
    num_gold: int

class CitationEvaluator:
    """Evaluate citation quality"""
    
    @staticmethod
    def extract_cited_articles(answer: str) -> Dict[str, str]:
        """
        Extract article citations from answer
        Format: "Madde 141", "TCK Madde 141", etc.
        
        Args:
            answer: Answer text
            
        Returns:
            Dict of {article_ref: full_text}
        """
        cited = {}
        
        # Pattern 1: "Madde XXX"
        for match in re.finditer(r'(Madde\s+(\d+))', answer, re.IGNORECASE):
            full_text = match.group(1)
            article_num = match.group(2)
            cited[f"Madde {article_num}"] = full_text
        
        # Pattern 2: Law abbreviations (TCK, TMK, etc)
        for match in re.finditer(r'((TCK|TMK|Ceza Kanunu|Medeni Kanun)\s+Madde\s+(\d+))', answer, re.IGNORECASE):
            full_text = match.group(1)
            law_name = match.group(2)
            article_num = match.group(3)
            key = f"{law_name} Madde {article_num}"
            cited[key] = full_text
        
        return cited
    
    @staticmethod
    def normalize_citation(citation: str) -> str:
        """
        Normalize citation for comparison
        E.g., "TCK Madde 141" -> "madde_141_tck"
        
        Args:
            citation: Citation string
            
        Returns:
            Normalized form
        """
        norm = citation.lower()
        # Extract just numbers and law name
        match = re.search(r'(\w+)\s+madde\s+(\d+)', norm)
        if match:
            return f"madde_{match.group(2)}_{match.group(1)}"
        return norm.replace(" ", "_")
    
    @staticmethod
    def evaluate(
        predicted_citations: List[str],
        gold_citations: List[str]
    ) -> CitationEvalResult:
        """
        Evaluate citation accuracy
        
        Args:
            predicted_citations: Citations extracted from model answer
            gold_citations: Gold standard citations (from ground truth)
            
        Returns:
            CitationEvalResult with precision, recall, F1
        """
        if len(gold_citations) == 0:
            # No gold citations provided
            if len(predicted_citations) == 0:
                return CitationEvalResult(
                    precision=1.0, recall=1.0, f1=1.0,
                    num_correct=0, num_predicted=0, num_gold=0
                )
            else:
                return CitationEvalResult(
                    precision=0.0, recall=0.0, f1=0.0,
                    num_correct=0, num_predicted=len(predicted_citations), num_gold=0
                )
        
        # Normalize for comparison
        pred_normalized = {CitationEvaluator.normalize_citation(c) for c in predicted_citations}
        gold_normalized = {CitationEvaluator.normalize_citation(c) for c in gold_citations}
        
        # Calculate overlap
        correct = pred_normalized & gold_normalized
        num_correct = len(correct)
        
        # Precision: of the citations we gave, how many were correct?
        if len(predicted_citations) == 0:
            precision = 0.0 if len(gold_citations) > 0 else 1.0
        else:
            precision = num_correct / len(predicted_citations)
        
        # Recall: of the gold citations, how many did we mention?
        recall = num_correct / len(gold_citations)
        
        # F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return CitationEvalResult(
            precision=precision,
            recall=recall,
            f1=f1,
            num_correct=num_correct,
            num_predicted=len(predicted_citations),
            num_gold=len(gold_citations)
        )
    
    @staticmethod
    def check_citation_support(
        citation: str,
        retrieved_docs: List[str]
    ) -> bool:
        """
        Check if a citation is actually supported by retrieved docs
        
        Args:
            citation: Citation to check
            retrieved_docs: Retrieved documents
            
        Returns:
            True if citation is mentioned in docs
        """
        combined = " ".join(retrieved_docs).lower()
        citation_lower = citation.lower()
        
        # Extract just the article number
        match = re.search(r'madde\s+(\d+)', citation_lower)
        if match:
            article_num = match.group(1)
            # Check for "Madde XXX" pattern in docs
            return f"madde {article_num}" in combined or f"Madde {article_num}" in combined.replace("madde", "Madde")
        
        return citation_lower in combined
