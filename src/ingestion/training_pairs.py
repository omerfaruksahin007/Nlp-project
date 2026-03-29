"""
PROMPT 6: Training Pairs Generator for Embedding Fine-tuning

Generates triplet training pairs (anchor, positive, negative) for fine-tuning
sentence transformers on Turkish legal domain.

Strategies:
1. Positive pairs (60%): Q-A matching + semantic similarity
2. Hard negatives (30%): Embedding-based similarity mining
3. Random negatives (10%): Easy negatives for balance

Usage:
    builder = TrainingPairsBuilder(chunk_file, qa_file)
    pairs = builder.generate_pairs(
        positive_ratio=0.6,
        hard_negative_ratio=0.3,
        random_negative_ratio=0.1
    )
    builder.save_pairs(output_path, pairs)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingPair:
    """Single training pair for embedding fine-tuning"""
    anchor_text: str
    positive_text: str
    negative_text: str
    anchor_id: str
    positive_id: str
    negative_id: str
    pair_type: str  # "qa_pair" | "semantic_related" | "hard_negative" | "random_negative"
    difficulty: float  # 0.0-1.0 (higher = harder)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class PairStatistics:
    """Statistics about generated pairs"""
    total_pairs: int
    positive_pairs: int
    hard_negative_pairs: int
    random_negative_pairs: int
    
    qa_pair_count: int
    semantic_related_count: int
    
    min_difficulty: float
    max_difficulty: float
    mean_difficulty: float
    std_difficulty: float
    median_difficulty: float
    
    anchor_text_length_mean: int
    positive_text_length_mean: int
    negative_text_length_mean: int
    
    duplicate_count: int = 0
    completeness: float = 1.0


class TrainingPairsBuilder:
    """
    Build training pairs for embedding fine-tuning using triplet loss.
    
    Combines multiple strategies:
    - QA-matching (direct question-answer pairs)
    - Semantic similarity (same article/law)
    - Hard negative mining (embedding-based)
    - Random negatives (easy negatives)
    """
    
    def __init__(
        self,
        chunk_file: Path,
        qa_file: Path,
        device: str = 'cpu',
        embedding_model: str = 'distiluse-base-multilingual-cased-v2'
    ):
        """
        Initialize builder with chunk and QA data.
        
        Args:
            chunk_file: Path to chunked JSONL file
            qa_file: Path to processed QA pairs JSONL file
            device: 'cpu' or 'cuda'
            embedding_model: Model name for hard negative mining
        """
        self.chunk_file = Path(chunk_file)
        self.qa_file = Path(qa_file)
        self.device = device
        self.embedding_model_name = embedding_model
        
        # Load data
        self.chunks = self._load_chunks()
        self.qa_pairs = self._load_qa_pairs()
        
        # Build indices for fast lookup
        self.chunk_by_source = self._build_source_index()
        self.chunks_by_article = self._build_article_index()
        
        # Embedding model (lazy loaded for hard negatives)
        self.embedding_model = None
        self.embeddings = None
        self.faiss_index = None
        
        logger.info(f"✅ TrainingPairsBuilder initialized")
        logger.info(f"   Chunks loaded: {len(self.chunks)}")
        logger.info(f"   QA pairs loaded: {len(self.qa_pairs)}")
    
    def _load_chunks(self) -> List[Dict]:
        """Load chunks from JSONL file"""
        chunks = []
        with open(self.chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
        logger.info(f"Loaded {len(chunks)} chunks from {self.chunk_file}")
        return chunks
    
    def _load_qa_pairs(self) -> List[Dict]:
        """Load QA pairs from JSONL file"""
        qa_pairs = []
        with open(self.qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                qa = json.loads(line.strip())
                qa_pairs.append(qa)
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {self.qa_file}")
        return qa_pairs
    
    def _build_source_index(self) -> Dict[str, List[Dict]]:
        """Build index: source_record_id → list of chunks"""
        index = defaultdict(list)
        for chunk in self.chunks:
            source_id = chunk.get('source_record_id', 'unknown')
            index[source_id].append(chunk)
        return dict(index)
    
    def _build_article_index(self) -> Dict[str, List[Dict]]:
        """Build index: (law_name, article_no) → list of chunks"""
        index = defaultdict(list)
        for chunk in self.chunks:
            law_name = chunk.get('law_name', 'unknown')
            article_no = chunk.get('article_no', 'unknown')
            key = f"{law_name}#{article_no}"
            index[key].append(chunk)
        return dict(index)
    
    def _load_embedding_model(self) -> None:
        """Load embedding model for hard negative mining"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )
        logger.info("✅ Embedding model loaded")
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index from chunk embeddings"""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss not installed")
        
        if self.embedding_model is None:
            self._load_embedding_model()
        
        logger.info("Building FAISS index from chunks...")
        
        # Embed all chunks
        chunk_texts = [c['chunk_text'] for c in self.chunks]
        self.embeddings = self.embedding_model.encode(
            chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        logger.info(f"✅ FAISS index built: {len(self.chunks)} vectors, dim={dimension}")
    
    def generate_pairs(
        self,
        positive_ratio: float = 0.6,
        hard_negative_ratio: float = 0.3,
        random_negative_ratio: float = 0.1,
        num_positives_per_anchor: int = 3,
        num_negatives_per_anchor: int = 2,
        seed: int = 42
    ) -> List[TrainingPair]:
        """
        Generate training pairs with specified distribution.
        
        Args:
            positive_ratio: Fraction of positive pairs
            hard_negative_ratio: Fraction of hard negatives
            random_negative_ratio: Fraction of random negatives
            num_positives_per_anchor: Positive samples per anchor
            num_negatives_per_anchor: Negative samples per anchor
            seed: Random seed for reproducibility
            
        Returns:
            List of TrainingPair objects
        """
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info("=" * 80)
        logger.info("GENERATING TRAINING PAIRS")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  positive_ratio: {positive_ratio:.2%}")
        logger.info(f"  hard_negative_ratio: {hard_negative_ratio:.2%}")
        logger.info(f"  random_negative_ratio: {random_negative_ratio:.2%}")
        
        # Generate positive pairs
        logger.info("\n[1/4] Generating positive pairs...")
        positive_pairs = self._generate_positive_pairs()
        logger.info(f"✅ Generated {len(positive_pairs)} positive pairs")
        
        # Generate hard negatives
        logger.info("\n[2/4] Generating hard negatives...")
        hard_negative_pairs = self._generate_hard_negatives(
            len(positive_pairs),
            num_negatives_per_anchor
        )
        logger.info(f"✅ Generated {len(hard_negative_pairs)} hard negative pairs")
        
        # Generate random negatives
        logger.info("\n[3/4] Generating random negatives...")
        random_negative_pairs = self._generate_random_negatives(
            len(positive_pairs),
            num_negatives_per_anchor
        )
        logger.info(f"✅ Generated {len(random_negative_pairs)} random negative pairs")
        
        # Combine and shuffle
        logger.info("\n[4/4] Merging and shuffling...")
        all_pairs = positive_pairs + hard_negative_pairs + random_negative_pairs
        
        random.shuffle(all_pairs)
        logger.info(f"✅ Total pairs: {len(all_pairs)}")
        
        # Verify distribution
        self._verify_distribution(all_pairs)
        
        return all_pairs
    
    def _generate_positive_pairs(self) -> List[TrainingPair]:
        """
        Strategy 1: QA-matching + semantic-related pairs (60%)
        
        For each QA pair:
            - Find answer chunks (from same source)
            - Create (question, answer_chunk, random_other) pairs
            
        Also create semantic-related pairs (same article)
        """
        pairs = []
        
        # QA-matching pairs
        for qa in self.qa_pairs:
            qa_id = qa.get('id', 'unknown')
            question = qa.get('question', '')
            
            # Find answer chunks for this QA
            answer_chunks = self.chunk_by_source.get(qa_id, [])
            
            if answer_chunks:
                for answer_chunk in answer_chunks:
                    pair = TrainingPair(
                        anchor_text=question,
                        positive_text=answer_chunk['chunk_text'],
                        negative_text='',  # Will be filled later
                        anchor_id=qa_id,
                        positive_id=answer_chunk.get('chunk_id', 'unknown'),
                        negative_id='',  # Will be filled later
                        pair_type='qa_pair',
                        difficulty=0.0
                    )
                    pairs.append(pair)
        
        # Semantic-related pairs (same article)
        for article_key, chunks in self.chunks_by_article.items():
            if len(chunks) > 1:
                # Create pairs between chunks in same article
                for i in range(len(chunks)):
                    for j in range(i + 1, min(i + 3, len(chunks))):  # Limit to avoid explosion
                        pair = TrainingPair(
                            anchor_text=chunks[i]['chunk_text'],
                            positive_text=chunks[j]['chunk_text'],
                            negative_text='',  # Will be filled later
                            anchor_id=chunks[i].get('chunk_id', 'unknown'),
                            positive_id=chunks[j].get('chunk_id', 'unknown'),
                            negative_id='',  # Will be filled later
                            pair_type='semantic_related',
                            difficulty=0.0
                        )
                        pairs.append(pair)
        
        return pairs
    
    def _generate_hard_negatives(
        self,
        num_anchors: int,
        num_per_anchor: int = 2
    ) -> List[TrainingPair]:
        """
        Strategy 2: Hard negative mining using embeddings
        
        For each anchor:
            1. Embed anchor
            2. Find top-k similar chunks
            3. Filter: exclude same article
            4. Select hardest (closest but wrong article)
        """
        if self.faiss_index is None:
            self._build_faiss_index()
        
        pairs = []
        
        # Sample anchors (to avoid too many negatives)
        all_anchors = []
        for qa in self.qa_pairs:
            qa_id = qa.get('id', 'unknown')
            question = qa.get('question', '')
            answer_chunks = self.chunk_by_source.get(qa_id, [])
            
            for answer_chunk in answer_chunks:
                all_anchors.append({
                    'anchor_text': question,
                    'anchor_id': qa_id,
                    'article_key': f"{answer_chunk.get('law_name', 'unknown')}#{answer_chunk.get('article_no', 'unknown')}"
                })
        
        # Sample subset
        sampled_anchors = random.sample(all_anchors, min(num_anchors, len(all_anchors)))
        
        for anchor in sampled_anchors:
            # Embed anchor
            anchor_emb = self.embedding_model.encode(
                [anchor['anchor_text']],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Find similar chunks
            distances, indices = self.faiss_index.search(
                anchor_emb.astype('float32'),
                k=20
            )
            
            # Find hard negatives (different article)
            hard_neg_candidates = []
            for idx in indices[0]:
                chunk = self.chunks[idx]
                chunk_article_key = f"{chunk.get('law_name', 'unknown')}#{chunk.get('article_no', 'unknown')}"
                
                if chunk_article_key != anchor['article_key']:
                    hard_neg_candidates.append((idx, distances[0][len(hard_neg_candidates)]))
            
            # Select hardest negatives
            hard_neg_candidates = sorted(hard_neg_candidates, key=lambda x: x[1])
            
            for idx, distance in hard_neg_candidates[:num_per_anchor]:
                chunk = self.chunks[idx]
                
                # Calculate difficulty (0-1, higher = harder)
                difficulty = min(1.0 - (distance / 2.0), 0.99)
                
                pair = TrainingPair(
                    anchor_text=anchor['anchor_text'],
                    positive_text='',  # Placeholder
                    negative_text=chunk['chunk_text'],
                    anchor_id=anchor['anchor_id'],
                    positive_id='',  # Not used for negatives
                    negative_id=chunk.get('chunk_id', 'unknown'),
                    pair_type='hard_negative',
                    difficulty=difficulty
                )
                pairs.append(pair)
        
        return pairs
    
    def _generate_random_negatives(
        self,
        num_anchors: int,
        num_per_anchor: int = 2
    ) -> List[TrainingPair]:
        """
        Strategy 3: Random negative sampling (easy negatives)
        
        For each anchor, randomly select chunks from different articles
        """
        pairs = []
        
        # Sample anchors
        all_anchors = []
        for qa in self.qa_pairs:
            qa_id = qa.get('id', 'unknown')
            question = qa.get('question', '')
            answer_chunks = self.chunk_by_source.get(qa_id, [])
            
            for answer_chunk in answer_chunks:
                all_anchors.append({
                    'anchor_text': question,
                    'anchor_id': qa_id,
                    'article_key': f"{answer_chunk.get('law_name', 'unknown')}#{answer_chunk.get('article_no', 'unknown')}"
                })
        
        sampled_anchors = random.sample(all_anchors, min(num_anchors, len(all_anchors)))
        
        for anchor in sampled_anchors:
            # Random chunks from different articles
            negative_candidates = [
                c for c in self.chunks
                if f"{c.get('law_name', 'unknown')}#{c.get('article_no', 'unknown')}" != anchor['article_key']
            ]
            
            if negative_candidates:
                selected_negatives = random.sample(
                    negative_candidates,
                    min(num_per_anchor, len(negative_candidates))
                )
                
                for neg_chunk in selected_negatives:
                    pair = TrainingPair(
                        anchor_text=anchor['anchor_text'],
                        positive_text='',
                        negative_text=neg_chunk['chunk_text'],
                        anchor_id=anchor['anchor_id'],
                        positive_id='',
                        negative_id=neg_chunk.get('chunk_id', 'unknown'),
                        pair_type='random_negative',
                        difficulty=0.05  # Easy
                    )
                    pairs.append(pair)
        
        return pairs
    
    def _verify_distribution(self, pairs: List[TrainingPair]) -> None:
        """Verify pair distribution is balanced"""
        by_type = defaultdict(int)
        difficulties = []
        
        for pair in pairs:
            by_type[pair.pair_type] += 1
            difficulties.append(pair.difficulty)
        
        logger.info("\nPair distribution:")
        for pair_type, count in by_type.items():
            pct = 100 * count / len(pairs)
            logger.info(f"  {pair_type}: {count} ({pct:.1f}%)")
        
        difficulties = np.array(difficulties)
        logger.info(f"\nDifficulty statistics:")
        logger.info(f"  Min: {difficulties.min():.2f}")
        logger.info(f"  Max: {difficulties.max():.2f}")
        logger.info(f"  Mean: {difficulties.mean():.2f}")
        logger.info(f"  Std: {difficulties.std():.2f}")
    
    def save_pairs(
        self,
        output_path: Path,
        pairs: List[TrainingPair]
    ) -> None:
        """Save training pairs to JSONL file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                record = pair.to_dict()
                # Convert numpy floats to Python floats
                record['difficulty'] = float(record['difficulty'])
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"\n✅ Saved {len(pairs)} pairs to {output_path}")
    
    def get_statistics(self, pairs: List[TrainingPair]) -> PairStatistics:
        """Compute statistics about generated pairs"""
        difficulties = [p.difficulty for p in pairs]
        difficulties_arr = np.array(difficulties)
        
        pair_types = defaultdict(int)
        text_lengths = {'anchor': [], 'positive': [], 'negative': []}
        
        for pair in pairs:
            pair_types[pair.pair_type] += 1
            
            if pair.anchor_text:
                text_lengths['anchor'].append(len(pair.anchor_text.split()))
            if pair.positive_text:
                text_lengths['positive'].append(len(pair.positive_text.split()))
            if pair.negative_text:
                text_lengths['negative'].append(len(pair.negative_text.split()))
        
        stats = PairStatistics(
            total_pairs=len(pairs),
            positive_pairs=len([p for p in pairs if p.difficulty == 0.0]),
            hard_negative_pairs=len([p for p in pairs if p.pair_type == 'hard_negative']),
            random_negative_pairs=len([p for p in pairs if p.pair_type == 'random_negative']),
            qa_pair_count=pair_types.get('qa_pair', 0),
            semantic_related_count=pair_types.get('semantic_related', 0),
            min_difficulty=float(difficulties_arr.min()),
            max_difficulty=float(difficulties_arr.max()),
            mean_difficulty=float(difficulties_arr.mean()),
            std_difficulty=float(difficulties_arr.std()),
            median_difficulty=float(np.median(difficulties_arr)),
            anchor_text_length_mean=int(np.mean(text_lengths['anchor'])) if text_lengths['anchor'] else 0,
            positive_text_length_mean=int(np.mean(text_lengths['positive'])) if text_lengths['positive'] else 0,
            negative_text_length_mean=int(np.mean(text_lengths['negative'])) if text_lengths['negative'] else 0,
        )
        
        return stats
