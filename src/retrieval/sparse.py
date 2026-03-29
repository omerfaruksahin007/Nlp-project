#!/usr/bin/env python3
"""
PROMPT 5 - Sparse Retrieval using BM25

BM25 is a probabilistic ranking function that considers:
- Term frequency (TF): How often word appears in document
- Inverse document frequency (IDF): How rare the word is overall
- Document length normalization: Longer docs don't always rank higher

Algorithm:
    Score(d, q) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d| / avgdl))
    
    where:
    - qi = query term
    - d = document
    - f(qi, d) = frequency of qi in document d
    - |d| = document length
    - avgdl = average document length
    - k1, b = tuning parameters (default: k1=2.0, b=0.75)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
import re
from typing import List


@dataclass
class SparseSearchResult:
    """Sparse search result"""
    chunk_id: str
    chunk_text: str
    source_record_id: str
    score: float
    rank: int
    metadata: Dict = None


class SparseRetriever:
    """
    BM25-based sparse retrieval for keyword matching.
    
    Attributes:
        tokenizer: Tokenizer for preprocessing
        bm25: BM25Okapi index
        chunks: List of chunks
        logger: Logger instance
    """
    
    def __init__(
        self,
        tokenizer_name: str = None,
        device: str = "cpu"
    ):
        """
        Initialize SparseRetriever.
        
        Args:
            tokenizer_name: Tokenizer name (unused - using simple tokenizer)
            device: Compute device (cpu or cuda)
        """
        self.tokenizer_name = tokenizer_name
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Use simple custom tokenizer instead of HF model
        self.logger.info("Using simple whitespace tokenizer")
        self.tokenizer = self._simple_tokenizer
        
        # Will be set later
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: Optional[List[Dict]] = None
        self.tokenized_chunks: Optional[List[List[str]]] = None
        
        # BM25 parameters
        self.k1 = 2.0      # Term frequency scaling
        self.b = 0.75      # Length normalization
        
        self.logger.info(f"✅ SparseRetriever initialized (BM25: k1={self.k1}, b={self.b})")
    
    @staticmethod
    def _simple_tokenizer(text: str) -> List[str]:
        """
        Simple tokenizer: lowercase + split by whitespace + remove punctuation.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove non-alphanumeric characters (keep spaces)
        text = re.sub(r'[^a-zğıöşüçA-ZĞIÖŞÜÇ0-9\s]', ' ', text)
        
        # Split by whitespace
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def build_index(self, chunks: List[Dict]) -> bool:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dicts with 'chunk_text' field
            
        Returns:
            True if successful
        """
        try:
            if not chunks:
                self.logger.error("No chunks provided")
                return False
            
            self.logger.info(f"Building BM25 index from {len(chunks)} chunks...")
            
            # Store chunks
            self.chunks = chunks
            
            # Tokenize all chunks
            self.logger.info("Tokenizing chunks...")
            self.tokenized_chunks = []
            
            for i, chunk in enumerate(chunks):
                if i % 5000 == 0:
                    self.logger.info(f"  Tokenized: {i}/{len(chunks)}")
                
                text = chunk.get('chunk_text', '')
                
                # Tokenize: split by spaces and convert to lowercase
                # Using simple whitespace tokenization for speed
                tokens = text.lower().split()
                
                # Remove special characters and short tokens
                tokens = [t.strip('.,!?;:"\'-') for t in tokens]
                tokens = [t for t in tokens if len(t) > 1]  # Min 2 chars
                
                self.tokenized_chunks.append(tokens)
            
            self.logger.info(f"✅ Tokenized {len(self.tokenized_chunks)} chunks")
            
            # Build BM25 index
            self.logger.info("Building BM25 index...")
            self.bm25 = BM25Okapi(
                self.tokenized_chunks,
                k1=self.k1,
                b=self.b
            )
            
            self.logger.info(f"✅ BM25 index built successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {e}", exc_info=True)
            return False
    
    def load_bm25(self, index_dir: str, dense_metadata_dir: str = None) -> bool:
        """
        Load pre-trained BM25 index from disk.
        
        Args:
            index_dir: Directory containing bm25.pkl, tokenized_chunks.json, and metadata
            dense_metadata_dir: Optional directory containing dense_metadata.json with full chunks
            
        Returns:
            True if successful
        """
        try:
            import pickle
            
            index_path = Path(index_dir)
            
            # Load BM25 index
            bm25_file = index_path / "bm25.pkl"
            if not bm25_file.exists():
                self.logger.error(f"BM25 index file not found: {bm25_file}")
                return False
            
            self.logger.info(f"Loading BM25 index from {bm25_file}...")
            with open(bm25_file, 'rb') as f:
                self.bm25 = pickle.load(f)
            self.logger.info(f"✅ BM25 index loaded")
            
            # Load tokenized chunks
            tokenized_file = index_path / "tokenized_chunks.json"
            if tokenized_file.exists():
                self.logger.info(f"Loading tokenized chunks...")
                with open(tokenized_file, 'r', encoding='utf-8') as f:
                    tokenized_data = json.load(f)
                    # Can be either list of lists or dict with 'chunks' key
                    if isinstance(tokenized_data, dict):
                        self.tokenized_chunks = tokenized_data.get('chunks', [])
                        self.chunks = tokenized_data.get('metadata', [])
                    else:
                        self.tokenized_chunks = tokenized_data
                self.logger.info(f"✅ Loaded {len(self.tokenized_chunks)} tokenized chunks")
            
            # Load metadata (optional, for configuration)
            metadata_file = index_path / "bm25_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Update BM25 parameters if available
                    if 'k1' in metadata:
                        self.k1 = metadata['k1']
                    if 'b' in metadata:
                        self.b = metadata['b']
                    self.logger.info(f"✅ Loaded metadata (k1={self.k1}, b={self.b})")
            
            # Load full chunks from dense metadata if available
            if dense_metadata_dir:
                dense_meta_file = Path(dense_metadata_dir) / "dense_metadata.json"
                if dense_meta_file.exists():
                    self.logger.info(f"Loading chunks from dense metadata...")
                    with open(dense_meta_file, 'r', encoding='utf-8') as f:
                        dense_meta = json.load(f)
                        self.chunks = dense_meta.get('chunks', [])
                        self.logger.info(f"✅ Loaded {len(self.chunks)} full chunks from dense metadata")
            
            # If chunks still not loaded, try from chunks_reference.jsonl
            if not self.chunks:
                # Try in dense metadata directory first, then sparse directory
                chunks_ref_file = None
                if dense_metadata_dir:
                    chunks_ref_file = Path(dense_metadata_dir) / "chunks_reference.jsonl"
                
                if not chunks_ref_file or not chunks_ref_file.exists():
                    chunks_ref_file = index_path / "chunks_reference.jsonl"
                
                if chunks_ref_file and chunks_ref_file.exists():
                    self.logger.info(f"Loading chunks from chunks_reference.jsonl...")
                    try:
                        self.chunks = []
                        with open(chunks_ref_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                self.chunks.append(json.loads(line))
                        self.logger.info(f"✅ Loaded {len(self.chunks)} chunks from chunks_reference.jsonl")
                    except Exception as e:
                        self.logger.warning(f"Failed to load chunks_reference.jsonl: {e}")
                        self.chunks = []
                
                # Final fallback: create placeholder from tokenized chunks
                if not self.chunks:
                    self.logger.warning("Chunks metadata not found, using placeholder structure")
                    self.chunks = []
                    for i in range(len(self.tokenized_chunks)):
                        self.chunks.append({
                            'idx': i,
                            'chunk_id': f'chunk_{i}',
                            'law_name': 'Unknown',
                            'article_no': '',
                        })
            
            self.logger.info(f"✅ BM25 retriever ready with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index: {e}", exc_info=True)
            return False
    
    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[SparseSearchResult]:
        """
        Search using BM25.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of SparseSearchResult objects
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if self.chunks is None:
            raise ValueError("Chunks not loaded.")
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            query_tokens = [t.strip('.,!?;:"\'-') for t in query_tokens]
            query_tokens = [t for t in query_tokens if len(t) > 1]
            
            self.logger.debug(f"Query tokens: {query_tokens}")
            
            # Get scores from BM25
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Build results
            results = []
            
            for rank, idx in enumerate(top_indices, 1):
                score = float(scores[idx])
                
                # Skip zero scores
                if score == 0.0:
                    continue
                
                # Safety check: verify index is in range
                if idx < 0 or idx >= len(self.chunks):
                    self.logger.debug(f"Skipping out-of-range index {idx} (chunks: {len(self.chunks)})")
                    continue
                
                chunk = self.chunks[idx]
                
                result = SparseSearchResult(
                    chunk_id=chunk.get('chunk_id') or f"chunk_{idx}",
                    chunk_text=chunk.get('chunk_text') or "[No text available]",
                    source_record_id=chunk.get('source_record_id') or "unknown",
                    score=score,
                    rank=rank,
                    metadata={
                        'law_name': chunk.get('law_name', ''),
                        'article_no': chunk.get('article_no', ''),
                        'section': chunk.get('section', ''),
                        'source': chunk.get('source', ''),
                    }
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[List[SparseSearchResult]]:
        """
        Search multiple queries.
        
        Args:
            queries: List of query texts
            k: Number of results per query
            
        Returns:
            List of result lists
        """
        results = []
        for query in queries:
            results.append(self.search(query, k=k))
        return results


def setup_sparse_retriever(
    chunks_file: str,
    tokenizer_name: str = "distiluse-base-multilingual-cased-v2",
    device: str = "cpu"
) -> Tuple[Optional[SparseRetriever], int]:
    """
    Setup sparse retriever from chunks file.
    
    Args:
        chunks_file: Path to chunks JSONL file
        tokenizer_name: Tokenizer model name
        device: Compute device
        
    Returns:
        Tuple of (retriever, chunk_count) or (None, 0) if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize retriever
        retriever = SparseRetriever(
            tokenizer_name=tokenizer_name,
            device=device
        )
        
        # Load chunks
        logger.info(f"Loading chunks from {chunks_file}...")
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line))
        
        logger.info(f"✅ Loaded {len(chunks)} chunks")
        
        # Build index
        if retriever.build_index(chunks):
            return retriever, len(chunks)
        else:
            return None, 0
            
    except Exception as e:
        logger.error(f"Failed to setup sparse retriever: {e}")
        return None, 0
