#!/usr/bin/env python3
"""
PROMPT 5+: Trained Model Loader

Loads pre-trained dense embeddings and BM25 indexes from disk
and initializes hybrid retriever.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainedModelLoader:
    """Load pre-trained models (embeddings + indexes) from disk"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.embeddings_dir = self.model_dir / "fine_tuned_embeddings"
        self.dense_index_dir = self.model_dir / "retrieval_index"
        self.sparse_index_dir = self.model_dir / "sparse_index"
        
        logger.info(f"Model directory: {self.model_dir}")
        self._verify_model_files()
    
    def _verify_model_files(self):
        """Check all required model files exist"""
        required = [
            (self.embeddings_dir / "model.safetensors", "Fine-tuned embeddings"),
            (self.dense_index_dir / "dense.index", "Dense FAISS index"),
            (self.sparse_index_dir / "bm25.pkl", "BM25 sparse index"),
        ]
        
        for file_path, name in required:
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {name}: {size_mb:.1f} MB")
            else:
                logger.warning(f"  ⚠️  {name} NOT FOUND: {file_path}")
    
    def load_dense_retriever(self) -> Optional[DenseRetriever]:
        """Load DenseRetriever with fine-tuned embeddings"""
        try:
            logger.info("\n[1] Loading dense retriever (sentence-transformers)...")
            
            retriever = DenseRetriever(
                model_name=str(self.embeddings_dir),  # Use local fine-tuned model
                device="cpu"  # Change to "cuda" if GPU available
            )
            
            # Load the pre-trained FAISS index
            success = retriever.load_index()
            
            if success:
                logger.info(f"✅ Dense retriever loaded successfully")
                return retriever
            else:
                logger.error("Failed to load dense index from disk")
                return None
                
        except Exception as e:
            logger.error(f"Error loading dense retriever: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_sparse_retriever(self) -> Optional[SparseRetriever]:
        """Load SparseRetriever with BM25 index"""
        try:
            logger.info("\n[2] Loading sparse retriever (BM25)...")
            
            retriever = SparseRetriever(device="cpu")
            
            # Load the pre-trained BM25 index (pass directory, not file)
            # Also pass dense metadata directory for full chunks
            success = retriever.load_bm25(
                str(self.sparse_index_dir),
                dense_metadata_dir=str(self.dense_index_dir)
            )
            
            if success:
                logger.info(f"✅ Sparse retriever loaded successfully")
                return retriever
            else:
                logger.error("Failed to load BM25 index from disk")
                return None
                
        except Exception as e:
            logger.error(f"Error loading sparse retriever: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_hybrid_retriever(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ) -> Optional[HybridRetriever]:
        """Load HybridRetriever combining dense + sparse"""
        try:
            logger.info("\n[3] Setting up hybrid retriever...")
            
            # Load both retrievers
            dense = self.load_dense_retriever()
            if not dense:
                logger.error("Cannot initialize hybrid without dense retriever")
                return None
            
            sparse = self.load_sparse_retriever()
            if not sparse:
                logger.error("Cannot initialize hybrid without sparse retriever")
                return None
            
            # Combine them
            hybrid = HybridRetriever(
                dense_retriever=dense,
                sparse_retriever=sparse,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
            
            logger.info(f"✅ Hybrid retriever ready")
            logger.info(f"   Dense weight: {dense_weight}, Sparse weight: {sparse_weight}")
            
            return hybrid
            
        except Exception as e:
            logger.error(f"Error loading hybrid retriever: {e}")
            import traceback
            traceback.print_exc()
            return None


def load_trained_models(
    model_dir: str = "models",
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4
) -> Optional[HybridRetriever]:
    """
    Convenience function: Load all trained models and return hybrid retriever
    
    Args:
        model_dir: Directory containing trained models
        dense_weight: Weight for dense retrieval in fusion
        sparse_weight: Weight for sparse retrieval in fusion
    
    Returns:
        HybridRetriever instance or None if failed
    """
    loader = TrainedModelLoader(model_dir)
    return loader.load_hybrid_retriever(dense_weight, sparse_weight)


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING TRAINED MODEL LOADER")
    logger.info("=" * 70)
    
    # Test loading
    retriever = load_trained_models()
    
    if retriever:
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL MODELS LOADED SUCCESSFULLY!")
        logger.info("=" * 70)
        
        # Test a query
        logger.info("\nTesting retrieval...")
        test_query = "ceza kanunu nedir"
        results = retriever.search(test_query, k_final=3)
        
        if results:
            logger.info(f"Found {len(results)} results:")
            for result in results:
                logger.info(f"  • {result.chunk_text[:60]}... (score: {result.hybrid_score:.4f})")
        else:
            logger.error("No results found!")
    else:
        logger.error("\n" + "=" * 70)
        logger.error("❌ FAILED TO LOAD MODELS")
        logger.error("=" * 70)
