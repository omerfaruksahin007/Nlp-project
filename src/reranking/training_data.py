"""
PROMPT 8 (Part 2): Training Data for Cross-Encoder

Builds training data specifically for fine-tuning cross-encoders.

Cross-encoder format differs from bi-encoder:
    Bi-encoder triplets: (anchor, positive, negative) - embed separately
    Cross-encoder triples: (query, positive_doc, negative_doc) - pairs for scoring

Strategy:
    1. Positive pairs: (query, relevant_chunk)
    2. Negative pairs: (query, irrelevant_chunk)
    3. Hard negatives: chunks from same domain but irrelevant
    
Output format for cross-encoder training (sentence-transformers):
    {
        "texts": ["query text", "document text"],
        "label": 1.0  (1.0 for relevant, 0.0 for irrelevant)
    }
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import random

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CrossEncoderTrainingExample:
    """Single training example for cross-encoder"""
    query: str
    document: str
    label: float  # 1.0 = relevant, 0.0 = irrelevant
    pair_type: str  # "positive" or "negative" (for analysis)
    difficulty: float  # 0.0-1.0 (higher = harder negative)
    document_id: str
    query_id: str
    
    def to_sentence_transformers_format(self) -> Dict:
        """Convert to sentence-transformers training format"""
        return {
            "texts": [self.query, self.document],
            "label": self.label
        }


class CrossEncoderTrainingDataBuilder:
    """
    Builds training data for cross-encoder fine-tuning.
    
    Unlike bi-encoders (which embed query and document separately),
    cross-encoders process the pair together, making them more accurate
    for ranking but slower.
    """
    
    def __init__(
        self,
        chunk_file: Path,
        qa_file: Path,
        device: str = 'cpu'
    ):
        """
        Initialize builder.
        
        Args:
            chunk_file: Path to chunked JSONL
            qa_file: Path to QA pairs JSONL
            device: 'cpu' or 'cuda'
        """
        self.chunk_file = Path(chunk_file)
        self.qa_file = Path(qa_file)
        self.device = device
        
        # Load data
        self.chunks = self._load_chunks()
        self.qa_pairs = self._load_qa_pairs()
        
        # Build indices
        self.chunk_by_id = {c['chunk_id']: c for c in self.chunks}
        self.chunks_by_source = defaultdict(list)
        for chunk in self.chunks:
            source_id = chunk.get('source_record_id', 'unknown')
            self.chunks_by_source[source_id].append(chunk)
        
        # For hard negative mining
        self.embedding_model = None
        self.embeddings = None
        self.faiss_index = None
        
        logger.info(f"✅ CrossEncoderTrainingDataBuilder initialized")
        logger.info(f"   Chunks: {len(self.chunks)}")
        logger.info(f"   QA pairs: {len(self.qa_pairs)}")
    
    def _load_chunks(self) -> List[Dict]:
        """Load chunks"""
        chunks = []
        with open(self.chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def _load_qa_pairs(self) -> List[Dict]:
        """Load QA pairs"""
        pairs = []
        with open(self.qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                pairs.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(pairs)} QA pairs")
        return pairs
    
    def _load_embedding_model(self) -> None:
        """Load model for hard negative mining"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required")
        
        logger.info("Loading embedding model for hard negative mining...")
        self.embedding_model = SentenceTransformer(
            'distiluse-base-multilingual-cased-v2',
            device=self.device
        )
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index for hard negative search"""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu required")
        
        if self.embedding_model is None:
            self._load_embedding_model()
        
        logger.info("Building FAISS index...")
        
        texts = [c['chunk_text'] for c in self.chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create index
        self.embeddings = embeddings.astype('float32')
        self.faiss_index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings)
        
        logger.info(f"✅ FAISS index built: {self.faiss_index.ntotal} vectors")
    
    def generate_training_data(
        self,
        positive_ratio: float = 0.6,
        hard_negative_ratio: float = 0.3,
        random_negative_ratio: float = 0.1,
        num_negatives_per_query: int = 3
    ) -> List[CrossEncoderTrainingExample]:
        """
        Generate training examples for cross-encoder.
        
        Args:
            positive_ratio: Fraction of positive pairs
            hard_negative_ratio: Fraction of hard negatives
            random_negative_ratio: Fraction of random negatives
            num_negatives_per_query: How many negatives per query
        
        Returns:
            List of training examples
        """
        examples = []
        
        # 1. Generate positive pairs (query, relevant_document)
        logger.info(f"\n[1/3] Generating positive pairs ({positive_ratio*100:.0f}%)...")
        
        positive_examples = []
        for qa in self.qa_pairs:
            query = qa.get('question', '')
            qa_id = qa.get('id', 'unknown')
            
            if not query:
                continue
            
            # Get relevant documents
            relevant_chunks = self.chunks_by_source.get(qa_id, [])
            
            for chunk in relevant_chunks:
                example = CrossEncoderTrainingExample(
                    query=query,
                    document=chunk['chunk_text'],
                    label=1.0,  # Positive
                    pair_type='positive',
                    difficulty=0.0,  # Not difficult
                    document_id=chunk.get('chunk_id', 'unknown'),
                    query_id=qa_id
                )
                positive_examples.append(example)
        
        logger.info(f"✅ Generated {len(positive_examples)} positive examples")
        
        # 2. Generate hard negatives (using embeddings)
        logger.info(f"\n[2/3] Generating hard negatives ({hard_negative_ratio*100:.0f}%)...")
        
        hard_negative_examples = []
        
        if hard_negative_ratio > 0:
            if self.faiss_index is None:
                self._build_faiss_index()
            
            # For each query, find hard negatives
            for qa in self.qa_pairs[:min(len(self.qa_pairs), 20)]:  # Limit to 20 queries
                query = qa.get('question', '')
                qa_id = qa.get('id', 'unknown')
                
                if not query:
                    continue
                
                # Embed query
                query_emb = self.embedding_model.encode(
                    [query],
                    convert_to_numpy=True
                ).astype('float32')
                
                # Find similar chunks
                distances, indices = self.faiss_index.search(query_emb, k=50)
                
                # Get relevant chunk IDs (to exclude)
                relevant_ids = {
                    c['chunk_id'] for c in self.chunks_by_source.get(qa_id, [])
                }
                
                # Find hard negatives (similar to query but not relevant)
                neg_count = 0
                for idx in indices[0]:
                    if neg_count >= num_negatives_per_query:
                        break
                    
                    chunk = self.chunks[idx]
                    if chunk['chunk_id'] not in relevant_ids:
                        # Calculate difficulty (inverse distance)
                        distance = distances[0][neg_count]
                        difficulty = min(1.0 - (distance / 50), 0.99)
                        
                        example = CrossEncoderTrainingExample(
                            query=query,
                            document=chunk['chunk_text'],
                            label=0.0,  # Negative
                            pair_type='hard_negative',
                            difficulty=difficulty,
                            document_id=chunk['chunk_id'],
                            query_id=qa_id
                        )
                        hard_negative_examples.append(example)
                        neg_count += 1
        
        logger.info(f"✅ Generated {len(hard_negative_examples)} hard negatives")
        
        # 3. Generate random negatives (easy sampling)
        logger.info(f"\n[3/3] Generating random negatives ({random_negative_ratio*100:.0f}%)...")
        
        random_negative_examples = []
        
        if random_negative_ratio > 0:
            for qa in self.qa_pairs:
                query = qa.get('question', '')
                qa_id = qa.get('id', 'unknown')
                
                if not query:
                    continue
                
                # Get relevant chunk IDs
                relevant_ids = {
                    c['chunk_id'] for c in self.chunks_by_source.get(qa_id, [])
                }
                
                # Sample random chunks (not relevant)
                irrelevant_chunks = [
                    c for c in self.chunks if c['chunk_id'] not in relevant_ids
                ]
                
                sampled = random.sample(
                    irrelevant_chunks,
                    min(num_negatives_per_query, len(irrelevant_chunks))
                )
                
                for chunk in sampled:
                    example = CrossEncoderTrainingExample(
                        query=query,
                        document=chunk['chunk_text'],
                        label=0.0,  # Negative
                        pair_type='random_negative',
                        difficulty=0.1,  # Easy negative
                        document_id=chunk['chunk_id'],
                        query_id=qa_id
                    )
                    random_negative_examples.append(example)
        
        logger.info(f"✅ Generated {len(random_negative_examples)} random negatives")
        
        # Combine and balance
        total_target = len(positive_examples) // 2  # Total examples budget
        
        pos_count = int(total_target * positive_ratio)
        hard_neg_count = int(total_target * hard_negative_ratio)
        random_neg_count = int(total_target * random_negative_ratio)
        
        examples += random.sample(positive_examples, min(pos_count, len(positive_examples)))
        examples += random.sample(hard_negative_examples, min(hard_neg_count, len(hard_negative_examples)))
        examples += random.sample(random_negative_examples, min(random_neg_count, len(random_negative_examples)))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Generated {len(examples)} total training examples")
        logger.info(f"   Positive: {sum(1 for e in examples if e.pair_type == 'positive')}")
        logger.info(f"   Hard negatives: {sum(1 for e in examples if e.pair_type == 'hard_negative')}")
        logger.info(f"   Random negatives: {sum(1 for e in examples if e.pair_type == 'random_negative')}")
        logger.info(f"{'='*80}")
        
        return examples
    
    def save_to_jsonl(
        self,
        examples: List[CrossEncoderTrainingExample],
        output_path: Path
    ) -> None:
        """Save examples to JSONL format for sentence-transformers"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(examples)} examples to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                # sentence-transformers format
                record = {
                    "texts": [example.query, example.document],
                    "label": example.label,
                    "pair_type": example.pair_type,
                    "difficulty": example.difficulty,
                    "query_id": example.query_id,
                    "document_id": example.document_id
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Saved to {output_path}")
    
    def save_statistics(
        self,
        examples: List[CrossEncoderTrainingExample],
        output_path: Path
    ) -> None:
        """Save training data statistics"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        difficulties = [e.difficulty for e in examples]
        
        stats = {
            "total_examples": len(examples),
            "positive_examples": sum(1 for e in examples if e.label == 1.0),
            "negative_examples": sum(1 for e in examples if e.label == 0.0),
            "pair_distribution": {
                "positive": sum(1 for e in examples if e.pair_type == 'positive'),
                "hard_negative": sum(1 for e in examples if e.pair_type == 'hard_negative'),
                "random_negative": sum(1 for e in examples if e.pair_type == 'random_negative')
            },
            "difficulty_stats": {
                "min": float(np.min(difficulties)) if difficulties else 0,
                "max": float(np.max(difficulties)) if difficulties else 0,
                "mean": float(np.mean(difficulties)) if difficulties else 0,
                "median": float(np.median(difficulties)) if difficulties else 0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Statistics saved to {output_path}")
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("Training Data Summary:")
        logger.info(f"  Total: {stats['total_examples']}")
        logger.info(f"  Positive: {stats['positive_examples']}")
        logger.info(f"  Negative: {stats['negative_examples']}")
        logger.info(f"  Ratio: {stats['positive_examples']/stats['total_examples']*100:.1f}% positive")
        logger.info(f"{'='*80}")
