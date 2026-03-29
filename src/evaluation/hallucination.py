#!/usr/bin/env python3
"""
PROMPT 11: Hallucination Analysis Module
Detect unsupported claims, partial support, wrong citations, missing answers
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
import re

class HallucinationType(Enum):
    """Types of hallucinations/errors"""
    UNSUPPORTED_CLAIM = "unsupported_claim"  # Claim not in retrieved docs
    PARTIALLY_SUPPORTED = "partially_supported"  # Some support but incomplete
    WRONG_CITATION = "wrong_citation"  # Citation doesn't match content
    NO_ANSWER_DESPITE_EVIDENCE = "no_answer_despite_evidence"  # Evidence available but not used
    CORRECT = "correct"  # Properly supported answer

@dataclass
class HallucinationAnalysis:
    """Result of hallucination analysis"""
    hallucination_type: HallucinationType
    confidence: float  # 0-1, how confident is the detection
    explanation: str
    supporting_evidence: Optional[List[str]] = None
    unsupported_claims: Optional[List[str]] = None

class HallucinationDetector:
    """Detect hallucinations in RAG answers"""
    
    @staticmethod
    def extract_claims(answer: str) -> List[str]:
        """
        Extract main claims from answer
        Heuristic: Split by period, question mark, etc.
        
        Args:
            answer: Answer text
            
        Returns:
            List of claims
        """
        # Split by sentence terminators
        sentences = re.split(r'[.!?]+', answer)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    @staticmethod
    def extract_citations(answer: str) -> List[str]:
        """
        Extract citations from answer
        Heuristic: Look for "Madde X", "Kanun", article references
        
        Args:
            answer: Answer text
            
        Returns:
            List of cited references
        """
        citations = []
        
        # Pattern: "Madde XXX"
        madde_matches = re.findall(r'Madde\s+(\d+)', answer, re.IGNORECASE)
        citations.extend([f"Madde {m}" for m in madde_matches])
        
        # Pattern: "TCK", "TMK", "Kanun"
        law_matches = re.findall(r'(TCK|TMK|Ceza Kanunu|Medeni Kanun)', answer, re.IGNORECASE)
        citations.extend(law_matches)
        
        return list(set(citations))  # Remove duplicates
    
    @staticmethod
    def claim_coverage(claim: str, retrieved_docs: List[str]) -> Tuple[float, List[str]]:
        """
        Check how much a claim is covered by retrieved docs
        
        Args:
            claim: A claim statement
            retrieved_docs: List of retrieved document texts
            
        Returns:
            (coverage_score, supporting_docs)
        """
        claim_words = set(claim.lower().split())
        supporting_docs = []
        
        for doc in retrieved_docs:
            doc_words = set(doc.lower().split())
            overlap = len(claim_words & doc_words)
            coverage = overlap / len(claim_words) if len(claim_words) > 0 else 0.0
            
            if coverage > 0.3:  # At least 30% word overlap
                supporting_docs.append(doc)
        
        if len(supporting_docs) > 0:
            # Calculate average coverage
            avg_coverage = sum(
                len(set(claim.lower().split()) & set(doc.lower().split())) / len(claim_words)
                for doc in supporting_docs
            ) / len(supporting_docs)
            return avg_coverage, supporting_docs
        
        return 0.0, []
    
    @staticmethod
    def citation_accuracy(citations: List[str], retrieved_docs: List[str]) -> Tuple[float, Dict[str, bool]]:
        """
        Check if cited references are mentioned in retrieved docs
        
        Args:
            citations: List of citations from answer
            retrieved_docs: List of retrieved documents
            
        Returns:
            (accuracy_score, citation_validity_dict)
        """
        combined_text = " ".join(retrieved_docs).lower()
        citation_validity = {}
        
        for citation in citations:
            # Check if citation appears in docs
            is_valid = citation.lower() in combined_text
            citation_validity[citation] = is_valid
        
        if len(citations) == 0:
            return 1.0, {}
        
        accuracy = sum(citation_validity.values()) / len(citations)
        return accuracy, citation_validity
    
    @staticmethod
    def analyze(
        answer: str,
        retrieved_docs: List[str],
        gold_answer: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> HallucinationAnalysis:
        """
        Comprehensive hallucination analysis
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved context documents
            gold_answer: Gold standard answer (optional)
            confidence_score: Model's confidence score (optional)
            
        Returns:
            HallucinationAnalysis object
        """
        
        # Extract components
        claims = HallucinationDetector.extract_claims(answer)
        citations = HallucinationDetector.extract_citations(answer)
        
        # Check claim coverage
        unsupported = []
        supported = []
        partially_supported = []
        
        for claim in claims:
            coverage, docs = HallucinationDetector.claim_coverage(claim, retrieved_docs)
            
            if coverage == 0.0:
                unsupported.append(claim)
            elif coverage < 0.5:
                partially_supported.append(claim)
            else:
                supported.append(claim)
        
        # Check citation accuracy
        citation_accuracy, citation_validity = HallucinationDetector.citation_accuracy(
            citations, retrieved_docs
        )
        
        # Determine hallucination type
        detection_confidence = 0.5  # Base confidence
        
        if len(unsupported) > 0:
            hallucination_type = HallucinationType.UNSUPPORTED_CLAIM
            detection_confidence = 0.9
            explanation = f"Found {len(unsupported)} unsupported claim(s): {unsupported[0]}"
        
        elif len(partially_supported) > 0 and citation_accuracy < 0.8:
            hallucination_type = HallucinationType.PARTIALLY_SUPPORTED
            detection_confidence = 0.7
            explanation = f"Answer partially supported. {len(partially_supported)} claim(s) have <50% evidence."
        
        elif citation_accuracy < 0.5:
            hallucination_type = HallucinationType.WRONG_CITATION
            detection_confidence = 0.85
            invalid_citations = [c for c, v in citation_validity.items() if not v]
            explanation = f"Citations not found in docs: {invalid_citations[:2]}"
        
        elif len(retrieved_docs) > 0 and len(claims) == 0:
            hallucination_type = HallucinationType.NO_ANSWER_DESPITE_EVIDENCE
            detection_confidence = 0.6
            explanation = "Retrieved relevant docs but no clear answer provided"
        
        else:
            hallucination_type = HallucinationType.CORRECT
            detection_confidence = 0.95
            explanation = "Answer properly supported by retrieved documents"
        
        # Adjust confidence based on model's confidence if available
        if confidence_score is not None:
            if confidence_score == "low":
                detection_confidence = min(detection_confidence + 0.1, 1.0)
            elif confidence_score == "high" and hallucination_type != HallucinationType.CORRECT:
                detection_confidence = min(detection_confidence + 0.05, 1.0)
        
        return HallucinationAnalysis(
            hallucination_type=hallucination_type,
            confidence=detection_confidence,
            explanation=explanation,
            supporting_evidence=supported,
            unsupported_claims=unsupported if len(unsupported) > 0 else None
        )
