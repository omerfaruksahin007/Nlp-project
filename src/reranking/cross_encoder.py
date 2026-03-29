"""
PROMPT 8 (Part 1): Cross-Encoder Module

Implements cross-encoder reranking for query-document pairs.
The cross-encoder takes a query and document together to produce a relevance score,
which is more accurate than comparing embeddings separately (bi-encoders).

Architecture:
    Input: (query, document) pair
    ↓
    [Transformer - encode both together]
    ↓
    Output: Relevance score (0-1)

Compared to dense retrieval (bi-encoder):
    Bi-encoder: embed query, embed doc separately → cosine similarity
    Cross-encoder: process (query, doc) together → classification head
    
    Advantage: Cross-encoder is slower but more accurate for reranking top-k
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Single reranked result"""
    chunk_id: str
    chunk_text: str
    query: str
    relevance_score: float  # 0-1 from cross-encoder
    original_rank: int      # Original retrieval rank
    reranked_rank: int      # After reranking
    source: str            # Where it came from (dense/sparse/hybrid)


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.
    
    A cross-encoder processes a query and document together to produce
    a relevance score. This is more accurate than bi-encoder similarity
    for ranking tasks, but slower (can't batch embed documents offline).
    
    Best used for:
    - Reranking top-k results (e.g., top 20 → top 5)
    - Not for initial retrieval (too slow for large corpus)
    """
    
    def __init__(
        self,
        model_name: str = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        device: str = 'cpu',
        batch_size: int = 32
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model ID for cross-encoder
                Common options:
                - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (512 tokens)
                - 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1' (512 tokens, multilingual)
                - 'cross-encoder/qnli-distilroberta-base' (lightweight)
            device: 'cpu' or 'cuda'
            batch_size: Batch size for inference
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        
        self.model_name = model_name or 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading cross-encoder: {self.model_name}")
        self.model = CrossEncoder(self.model_name, device=device)
        self.model.max_length = 512
        
        logger.info(f"✅ Cross-encoder loaded on device: {device}")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        return_all: bool = False
    ) -> List[RerankedResult]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Input query string
            documents: List of dicts with keys:
                - chunk_text: Document text
                - chunk_id: Document ID
                - source: 'dense' or 'sparse' or 'hybrid'
                - rank: Original retrieval rank
            top_k: Return top-k documents
            return_all: If True, return all with scores; else only top-k
        
        Returns:
            List of RerankedResult sorted by relevance score (descending)
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        query_doc_pairs = []
        for doc in documents:
            # Support both 'text' and 'chunk_text' keys
            doc_text = doc.get('chunk_text') or doc.get('text', '')
            query_doc_pairs.append([query, doc_text])
        
        logger.info(f"Reranking {len(documents)} documents...")
        
        # Score all pairs (cross-encoder processes them together)
        scores = self.model.predict(
            query_doc_pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # scores shape: (num_docs,) in range [-inf, inf]
        # Normalize to [0, 1] using sigmoid
        relevance_scores = 1 / (1 + np.exp(-scores))
        
        # Create results with scores
        results = []
        for idx, (doc, score) in enumerate(zip(documents, relevance_scores)):
            chunk_id = doc.get('chunk_id') or doc.get('id', f'unknown_{idx}')
            chunk_text = doc.get('chunk_text') or doc.get('text', '')
            
            result = RerankedResult(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                query=query,
                relevance_score=float(score),
                original_rank=idx + 1,
                reranked_rank=0,  # Will be set after sorting
                source=doc.get('source', 'unknown')
            )
            results.append(result)
        
        # Sort by relevance score (descending)
        results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        # Update reranked ranks
        for idx, result in enumerate(results):
            result.reranked_rank = idx + 1
        
        # Return top-k or all
        if return_all:
            return results
        else:
            return results[:top_k]
    
    def score_pairs(
        self,
        query_doc_pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """
        Score raw query-document pairs.
        
        Args:
            query_doc_pairs: List of (query, document) tuples
        
        Returns:
            Array of scores, shape (num_pairs,)
        """
        logger.info(f"Scoring {len(query_doc_pairs)} pairs...")
        
        scores = self.model.predict(
            query_doc_pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return scores


class RerankerEnsemble:
    """
    Ensembles multiple rerankers or combines cross-encoder with
    retrieval scores for final ranking.
    
    Strategy: Combine dense, sparse, and reranker scores
        final_score = 0.3 * dense_score + 0.2 * sparse_score + 0.5 * cross_encoder
    """
    
    def __init__(
        self,
        cross_encoder: CrossEncoderReranker,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            cross_encoder: CrossEncoderReranker instance
            weights: Dict of {source: weight}
                e.g. {'dense': 0.3, 'sparse': 0.2, 'cross_encoder': 0.5}
        """
        self.cross_encoder = cross_encoder
        
        if weights is None:
            weights = {
                'dense': 0.3,
                'sparse': 0.2,
                'cross_encoder': 0.5
            }
        
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        
        logger.info(f"Ensemble weights: {self.weights}")
    
    def rerank_ensemble(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[RerankedResult]:
        """
        Rerank using ensemble of scores.
        
        Args:
            query: Input query
            documents: Retrieved documents with dense_score, sparse_score
            top_k: Return top-k
        
        Returns:
            Reranked results
        """
        if not documents:
            return []
        
        # Get cross-encoder scores
        reranked = self.cross_encoder.rerank(
            query,
            documents,
            top_k=len(documents),
            return_all=True
        )
        
        # Normalize scores (all to [0, 1])
        ce_scores = np.array([r.relevance_score for r in reranked])
        ce_scores = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-8)
        
        # Get other scores (if available)
        dense_scores = np.array([d.get('dense_score', 0.5) for d in documents])
        sparse_scores = np.array([d.get('sparse_score', 0.5) for d in documents])
        
        # Normalize
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)
        
        # Ensemble
        ensemble_scores = (
            self.weights.get('dense', 0.0) * dense_scores +
            self.weights.get('sparse', 0.0) * sparse_scores +
            self.weights.get('cross_encoder', 0.5) * ce_scores
        )
        
        # Re-sort by ensemble scores
        for result, score in zip(reranked, ensemble_scores):
            result.relevance_score = float(score)
        
        reranked = sorted(reranked, key=lambda x: x.relevance_score, reverse=True)
        
        # Update ranks
        for idx, result in enumerate(reranked):
            result.reranked_rank = idx + 1
        
        return reranked[:top_k]
