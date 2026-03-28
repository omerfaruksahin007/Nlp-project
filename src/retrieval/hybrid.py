#!/usr/bin/env python3
"""
PROMPT 5 - Hybrid Retrieval (Dense + Sparse Fusion)

Hybrid retrieval combines:
1. Dense retrieval (semantic similarity) - captures meaning
2. Sparse retrieval (keyword matching) - captures exact terms

Fusion strategy: RRF (Reciprocal Rank Fusion)
    
    Score(doc) = Σ 1 / (k + rank)
    
    where:
    - k = 60 (constant, typically)
    - rank = position in result list (1-based)
    
    Intuition: Combine rankings from multiple systems
    - If a document ranks high in both systems, it gets high score
    - If it ranks high in one system only, it still gets reasonable score
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from retrieval.dense import DenseRetriever, SearchResult as DenseSearchResult
from retrieval.sparse import SparseRetriever, SparseSearchResult


@dataclass
class HybridSearchResult:
    """Final hybrid search result"""
    chunk_id: str
    chunk_text: str
    source_record_id: str
    
    # Scores
    dense_score: float
    sparse_score: float
    hybrid_score: float
    
    # Ranks
    dense_rank: int
    sparse_rank: int
    hybrid_rank: int
    
    # Metadata
    metadata: Dict = None


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrievers.
    
    Attributes:
        dense_retriever: DenseRetriever instance
        sparse_retriever: SparseRetriever instance
        logger: Logger instance
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60
    ):
        """
        Initialize HybridRetriever.
        
        Args:
            dense_retriever: DenseRetriever instance (must be initialized)
            sparse_retriever: SparseRetriever instance (must be initialized)
            dense_weight: Weight for dense scores in weighted fusion
            sparse_weight: Weight for sparse scores in weighted fusion
            rrf_k: Constant for RRF formula (default 60)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"✅ HybridRetriever initialized "
            f"(RRF-k={rrf_k}, dense_w={dense_weight}, sparse_w={sparse_weight})"
        )
    
    def search(
        self,
        query: str,
        k_dense: int = 20,
        k_sparse: int = 20,
        k_final: int = 10,
        fusion_method: str = "rrf"
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Query text
            k_dense: Number of dense results to retrieve
            k_sparse: Number of sparse results to retrieve
            k_final: Number of final results to return
            fusion_method: "rrf" or "weighted"
            
        Returns:
            List of HybridSearchResult objects (top k_final)
        """
        try:
            # Get dense results
            self.logger.info(f"Searching dense index (k={k_dense})...")
            dense_results = self.dense_retriever.search(query, k=k_dense)
            
            # Get sparse results
            self.logger.info(f"Searching sparse index (k={k_sparse})...")
            sparse_results = self.sparse_retriever.search(query, k=k_sparse)
            
            self.logger.info(
                f"Got dense results: {len(dense_results)}, "
                f"sparse results: {len(sparse_results)}"
            )
            
            # Fuse results
            if fusion_method == "rrf":
                fused = self._rrf_fusion(dense_results, sparse_results, k_final)
            elif fusion_method == "weighted":
                fused = self._weighted_fusion(dense_results, sparse_results, k_final)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            return fused
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    def _rrf_fusion(
        self,
        dense_results: List[DenseSearchResult],
        sparse_results: List[SparseSearchResult],
        k_final: int
    ) -> List[HybridSearchResult]:
        """
        RRF (Reciprocal Rank Fusion) fusion.
        
        Formula: Score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        """
        # Create ranking dictionaries
        dense_ranking = {}  # chunk_id -> (rank, score)
        sparse_ranking = {}  # chunk_id -> (rank, score)
        
        for rank, result in enumerate(dense_results, 1):
            dense_ranking[result.chunk_id] = (rank, result.score)
        
        for rank, result in enumerate(sparse_results, 1):
            sparse_ranking[result.chunk_id] = (rank, result.score)
        
        # Combine all unique chunk IDs
        all_chunk_ids = set(dense_ranking.keys()) | set(sparse_ranking.keys())
        
        # Calculate RRF scores
        fused_scores = {}
        
        for chunk_id in all_chunk_ids:
            dense_rank, dense_score = dense_ranking.get(chunk_id, (len(dense_ranking) + 1, 0.0))
            sparse_rank, sparse_score = sparse_ranking.get(chunk_id, (len(sparse_ranking) + 1, 0.0))
            
            # RRF formula
            rrf_score = 1 / (self.rrf_k + dense_rank) + 1 / (self.rrf_k + sparse_rank)
            
            fused_scores[chunk_id] = {
                'chunk_id': chunk_id,
                'dense_rank': dense_rank,
                'dense_score': dense_score,
                'sparse_rank': sparse_rank,
                'sparse_score': sparse_score,
                'rrf_score': rrf_score
            }
        
        # Sort by RRF score
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )[:k_final]
        
        # Build output
        output = []
        for hybrid_rank, item in enumerate(sorted_results, 1):
            chunk_id = item['chunk_id']
            
            # Get chunk details from either retriever
            if chunk_id in {r.chunk_id for r in dense_results}:
                chunk_detail = next((r for r in dense_results if r.chunk_id == chunk_id), None)
            else:
                chunk_detail = next((r for r in sparse_results if r.chunk_id == chunk_id), None)
            
            if chunk_detail is None:
                continue
            
            result = HybridSearchResult(
                chunk_id=chunk_id,
                chunk_text=chunk_detail.chunk_text,
                source_record_id=chunk_detail.source_record_id,
                dense_score=item['dense_score'],
                sparse_score=item['sparse_score'],
                hybrid_score=item['rrf_score'],
                dense_rank=item['dense_rank'],
                sparse_rank=item['sparse_rank'],
                hybrid_rank=hybrid_rank,
                metadata=chunk_detail.metadata if hasattr(chunk_detail, 'metadata') else None
            )
            
            output.append(result)
        
        return output
    
    def _weighted_fusion(
        self,
        dense_results: List[DenseSearchResult],
        sparse_results: List[SparseSearchResult],
        k_final: int
    ) -> List[HybridSearchResult]:
        """
        Weighted fusion (simple averaging).
        
        Formula: Score = dense_weight * norm_dense_score + sparse_weight * norm_sparse_score
        """
        # Create dictionaries
        all_results = {}  # chunk_id -> {dense_score, sparse_score, ranks}
        
        # Add dense results
        for rank, result in enumerate(dense_results, 1):
            all_results[result.chunk_id] = {
                'chunk_text': result.chunk_text,
                'source_record_id': result.source_record_id,
                'metadata': result.metadata,
                'dense_score': result.score,
                'dense_rank': rank,
                'sparse_score': 0.0,
                'sparse_rank': len(sparse_results) + 1
            }
        
        # Add/update sparse results
        for rank, result in enumerate(sparse_results, 1):
            if result.chunk_id in all_results:
                all_results[result.chunk_id]['sparse_score'] = result.score
                all_results[result.chunk_id]['sparse_rank'] = rank
            else:
                all_results[result.chunk_id] = {
                    'chunk_text': result.chunk_text,
                    'source_record_id': result.source_record_id,
                    'metadata': result.metadata,
                    'dense_score': 0.0,
                    'dense_rank': len(dense_results) + 1,
                    'sparse_score': result.score,
                    'sparse_rank': rank
                }
        
        # Normalize scores to [0, 1]
        max_dense = max((r['dense_score'] for r in all_results.values()), default=1.0)
        max_sparse = max((r['sparse_score'] for r in all_results.values()), default=1.0)
        
        # Calculate weighted scores
        for chunk_id, item in all_results.items():
            norm_dense = item['dense_score'] / max_dense if max_dense > 0 else 0
            norm_sparse = item['sparse_score'] / max_sparse if max_sparse > 0 else 0
            
            item['weighted_score'] = (
                self.dense_weight * norm_dense + 
                self.sparse_weight * norm_sparse
            )
        
        # Sort and get top k
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1]['weighted_score'],
            reverse=True
        )[:k_final]
        
        # Build output
        output = []
        for hybrid_rank, (chunk_id, item) in enumerate(sorted_results, 1):
            result = HybridSearchResult(
                chunk_id=chunk_id,
                chunk_text=item['chunk_text'],
                source_record_id=item['source_record_id'],
                dense_score=item['dense_score'],
                sparse_score=item['sparse_score'],
                hybrid_score=item['weighted_score'],
                dense_rank=item['dense_rank'],
                sparse_rank=item['sparse_rank'],
                hybrid_rank=hybrid_rank,
                metadata=item['metadata']
            )
            
            output.append(result)
        
        return output
    
    def batch_search(
        self,
        queries: List[str],
        k_dense: int = 20,
        k_sparse: int = 20,
        k_final: int = 10
    ) -> List[List[HybridSearchResult]]:
        """
        Search multiple queries.
        
        Args:
            queries: List of query texts
            k_dense: Number of dense results per query
            k_sparse: Number of sparse results per query
            k_final: Number of final results per query
            
        Returns:
            List of result lists
        """
        results = []
        for query in queries:
            results.append(
                self.search(
                    query,
                    k_dense=k_dense,
                    k_sparse=k_sparse,
                    k_final=k_final
                )
            )
        return results
