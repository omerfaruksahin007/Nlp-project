"""
PROMPT 4: Dense Retrieval Module

Embed chunks and create FAISS index for fast similarity search.

Classes:
    DenseRetriever: Handles embedding and indexing
    IndexManager: Manages FAISS index persistence
    SearchResult: Return type for search queries
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class SearchResult:
    """Result of a single retrieved chunk"""
    chunk_id: str
    source_record_id: str
    chunk_text: str
    score: float  # Similarity score or distance
    rank: int     # Rank in results (1st, 2nd, etc.)
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'source_record_id': self.source_record_id,
            'chunk_text': self.chunk_text,
            'score': float(self.score),
            'rank': self.rank,
            'metadata': self.metadata or {}
        }


class IndexManager:
    """
    Manages FAISS index persistence and loading.
    
    Attributes:
        index_dir: Directory to store index files
        logger: Logger instance
    """
    
    def __init__(self, index_dir: str = "models/retrieval_index"):
        """
        Initialize IndexManager.
        
        Args:
            index_dir: Directory for storing index files
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # File locations
        self.index_file = self.index_dir / "dense.index"
        self.metadata_file = self.index_dir / "dense_metadata.json"
        self.config_file = self.index_dir / "dense_config.json"
    
    def save_index(
        self,
        index: 'faiss.Index',
        chunks: List[Dict],
        config: Dict
    ) -> bool:
        """
        Save FAISS index and metadata.
        
        Args:
            index: FAISS index object
            chunks: List of chunk dicts (for metadata)
            config: Configuration dict (model name, dimensions, etc.)
            
        Returns:
            True if successful
        """
        try:
            # Save FAISS index using pickle (more reliable than faiss.write_index)
            import pickle
            with open(self.index_file, 'wb') as f:
                pickle.dump(index, f)
            self.logger.info(f"✅ Index saved: {self.index_file}")
            
            # Save metadata (chunk info for retrieval)
            metadata = {
                'chunk_count': len(chunks),
                'timestamp': datetime.now().isoformat(),
                'chunks': [
                    {
                        'index': i,
                        'chunk_id': chunk.get('chunk_id'),
                        'chunk_text': chunk.get('chunk_text'),
                        'source_record_id': chunk.get('source_record_id'),
                        'law_name': chunk.get('law_name'),
                        'article_no': chunk.get('article_no'),
                        'section': chunk.get('section'),
                        'source': chunk.get('source'),
                    }
                    for i, chunk in enumerate(chunks)
                ]
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Metadata saved: {self.metadata_file}")
            
            # Save config
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Config saved: {self.config_file}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self) -> Tuple[Optional['faiss.Index'], Optional[Dict], Optional[Dict]]:
        """
        Load FAISS index and metadata.
        
        Returns:
            Tuple of (index, metadata, config) or (None, None, None) if failed
        """
        try:
            # Check for index file - PRIORITY: faiss.index (new) > dense.index (old)
            import faiss
            import pickle
            
            faiss_index_path = self.index_dir / "faiss.index"
            dense_index_path = self.index_file
            index_path = None
            index = None
            
            # Try NEW FAISS binary format FIRST (41k+ vectors)
            if faiss_index_path.exists():
                try:
                    index = faiss.read_index(str(faiss_index_path))
                    index_path = faiss_index_path
                    self.logger.info(f"✅ Index loaded (FAISS binary): {faiss_index_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load FAISS binary: {e}")
            
            # Fallback to OLD pickle format if FAISS failed
            if index is None and dense_index_path.exists():
                try:
                    with open(dense_index_path, 'rb') as f:
                        index = pickle.load(f)
                    index_path = dense_index_path
                    self.logger.info(f"✅ Index loaded (pickle): {dense_index_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load pickle: {e}")
            
            # If both failed, error
            if index is None:
                self.logger.warning(f"Index files not found: {faiss_index_path} or {dense_index_path}")
                return None, None, None
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.logger.info(f"✅ Metadata loaded: {self.metadata_file}")
            else:
                metadata = None
            
            # Load config
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"✅ Config loaded: {self.config_file}")
            else:
                config = None
            
            return index, metadata, config
        
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return None, None, None
    
    def index_exists(self) -> bool:
        """Check if index files exist"""
        return self.index_file.exists() and self.metadata_file.exists()


class DenseRetriever:
    """
    Dense retrieval system using embedding model and FAISS index.
    
    Workflow:
        1. Load embedding model (sentence-transformers)
        2. Embed all chunks
        3. Create FAISS index
        4. Search: Query → embed → search in index → retrieve chunks
    
    Attributes:
        model_name: HuggingFace model name
        model: Loaded SentenceTransformer model
        index_manager: IndexManager instance
        chunks: List of chunks
        index: FAISS index
        logger: Logger instance
    """
    
    def __init__(
        self,
        model_name: str = "distiluse-base-multilingual-cased-v2",
        index_dir: str = "models/retrieval_index",
        device: str = "cpu"
    ):
        """
        Initialize DenseRetriever.
        
        Args:
            model_name: HuggingFace sentence-transformers model
            index_dir: Directory for storing FAISS index
            device: Compute device ('cpu' or 'cuda')
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")
        
        self.model_name = model_name
        self.device = device
        
        # Load model
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.logger.info(f"✅ Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Initialize index manager
        self.index_manager = IndexManager(index_dir)
        
        # Placeholders
        self.chunks = None
        self.metadata = None
        self.index = None
        self.config = None
    
    def embed_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed all chunk texts.
        
        Args:
            chunks: List of chunk dicts (must have 'chunk_text' field)
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Embeddings array (num_chunks, embedding_dim)
        """
        # Extract chunk texts
        chunk_texts = [chunk.get('chunk_text', '') for chunk in chunks]
        
        if not chunk_texts:
            raise ValueError("No chunk texts found")
        
        self.logger.info(f"Embedding {len(chunk_texts)} chunks...")
        
        # Encode with batching
        embeddings = self.model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Use normalized vectors for L2
        )
        
        self.logger.info(f"✅ Embeddings created: {embeddings.shape}")
        
        return embeddings
    
    def create_index(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        index_type: str = "flat"
    ) -> 'faiss.Index':
        """
        Create FAISS index from embeddings.
        
        Args:
            chunks: List of chunk dicts (for metadata)
            embeddings: Embeddings array (num_chunks, embedding_dim)
            index_type: Type of index ('flat' or 'ivf')
            
        Returns:
            FAISS index object
        """
        dimension = embeddings.shape[1]
        
        self.logger.info(f"Creating FAISS index (type={index_type}, dim={dimension})...")
        
        # Create index
        if index_type.lower() == 'flat':
            # Simple brute-force index
            index = faiss.IndexFlatL2(dimension)
        elif index_type.lower() == 'ivf':
            # Inverted file index (for large scale)
            nlist = int(np.sqrt(len(embeddings)))  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings.astype('float32'))
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add vectors
        index.add(embeddings.astype('float32'))
        
        self.logger.info(f"✅ Index created with {index.ntotal} vectors")
        
        # Store for later use
        self.chunks = chunks
        self.index = index
        
        # Save index
        config = {
            'model_name': self.model_name,
            'embedding_dimension': dimension,
            'index_type': index_type,
            'num_vectors': index.ntotal,
            'chunk_count': len(chunks),
            'created_at': datetime.now().isoformat()
        }
        
        self.index_manager.save_index(index, chunks, config)
        
        return index
    
    def load_index(self) -> bool:
        """
        Load pre-built FAISS index.
        
        Returns:
            True if successful
        """
        index, metadata, config = self.index_manager.load_index()
        
        if index is None:
            self.logger.warning("Failed to load index")
            return False
        
        self.index = index
        self.metadata = metadata
        self.config = config
        
        # Load chunks from metadata or chunks_reference.jsonl
        if metadata and 'chunks' in metadata:
            self.chunks = metadata['chunks']
            self.logger.info(f"✅ Loaded {len(self.chunks)} chunks from metadata")
        else:
            # Try loading from chunks_reference.jsonl (new format)
            chunks_ref_file = self.index_manager.index_dir / 'chunks_reference.jsonl'
            if chunks_ref_file.exists():
                try:
                    chunks = []
                    with open(chunks_ref_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            chunks.append(json.loads(line))
                    self.chunks = chunks
                    self.logger.info(f"✅ Loaded {len(self.chunks)} chunks from chunks_reference.jsonl")
                except Exception as e:
                    self.logger.warning(f"Failed to load chunks_reference.jsonl: {e}")
                    self.chunks = None
            else:
                self.logger.warning("No chunks found in metadata or chunks_reference.jsonl")
                self.chunks = None
        
        self.logger.info(f"✅ Index loaded. Vectors: {index.ntotal}")
        
        return True
    
    def search(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            k: Number of results to return
            return_scores: Include similarity scores
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() or create_index() first.")
        
        if self.chunks is None:
            raise ValueError("Chunks not loaded.")
        
        # Embed query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            k=min(k, self.index.ntotal)
        )
        
        # Build results
        results = []
        
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Convert L2 distance to similarity score (0-1)
            # For normalized vectors: similarity = 1 - (L2_distance / 2)
            similarity_score = max(0.0, 1.0 - (distance / 2.0))
            
            result = SearchResult(
                chunk_id=chunk.get('chunk_id'),
                source_record_id=chunk.get('source_record_id'),
                chunk_text=chunk.get('chunk_text'),
                score=similarity_score,
                rank=rank,
                metadata={
                    'law_name': chunk.get('law_name'),
                    'article_no': chunk.get('article_no'),
                    'section': chunk.get('section'),
                    'source': chunk.get('source'),
                    'distance': float(distance)
                }
            )
            
            results.append(result)
        
        self.logger.debug(f"Search returned {len(results)} results for query")
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries.
        
        Args:
            queries: List of query texts
            k: Number of results per query
            
        Returns:
            List of result lists
        """
        all_results = []
        
        for query in queries:
            results = self.search(query, k=k)
            all_results.append(results)
        
        self.logger.info(f"Batch search complete: {len(queries)} queries")
        
        return all_results


def setup_dense_retriever(
    chunks_file: str,
    model_name: str = "distiluse-base-multilingual-cased-v2",
    index_dir: str = "models/retrieval_index",
    device: str = "cpu",
    force_rebuild: bool = False
) -> DenseRetriever:
    """
    Create or load a DenseRetriever instance.
    
    Args:
        chunks_file: Path to chunked JSONL file
        model_name: Embedding model name
        index_dir: Index directory
        device: Compute device
        force_rebuild: Rebuild index even if exists
        
    Returns:
        Configured DenseRetriever
    """
    retriever = DenseRetriever(model_name=model_name, index_dir=index_dir, device=device)
    
    # Check if index exists
    if retriever.index_manager.index_exists() and not force_rebuild:
        retriever.logger.info("Loading existing index...")
        success = retriever.load_index()
        if success:
            return retriever
    
    # Build new index
    retriever.logger.info("Building new index...")
    
    # Load chunks
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line.strip())
            chunks.append(chunk)
    
    retriever.logger.info(f"Loaded {len(chunks)} chunks")
    
    # Embed
    embeddings = retriever.embed_chunks(chunks)
    
    # Create index
    retriever.create_index(chunks, embeddings)
    
    return retriever
