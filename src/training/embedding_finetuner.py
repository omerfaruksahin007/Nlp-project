#!/usr/bin/env python3
"""
PROMPT 7: Embedding Fine-tuning Script for Turkish Legal RAG

Fine-tunes sentence-transformers on Turkish legal domain using triplet loss.

Usage:
    from src.training.embedding_finetuner import EmbeddingFineTuner
    
    tuner = EmbeddingFineTuner(
        model_name='distiluse-base-multilingual-cased-v2',
        output_dir='models/finetuned_embedding'
    )
    tuner.train(
        training_pairs_file='data/processed/training_pairs.jsonl',
        num_epochs=3,
        batch_size=64
    )
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import random

import torch
from sentence_transformers import (
    SentenceTransformer, 
    losses, 
    InputExample,
    models,
    util
)
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from datasets import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EmbeddingFineTuner:
    """
    Fine-tune sentence-transformers on Turkish legal domain.
    
    Strategy:
    - Load pre-trained multilingual model
    - Train on triplet pairs (anchor, positive, negative)
    - Use in-batch negatives + triplet loss
    - Save checkpoints and final model
    """
    
    def __init__(
        self,
        model_name: str = 'distiluse-base-multilingual-cased-v2',
        output_dir: str = 'models/finetuned_embedding',
        device: str = None
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model_name: HuggingFace model identifier
            output_dir: Directory to save outputs
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        logger.info(f"✅ Model loaded on {device}")
        logger.info(f"Model dimensions: {self.model.get_sentence_embedding_dimension()}")
    
    def _load_training_pairs(self, pairs_file: Path) -> List[InputExample]:
        """Load triplet pairs from JSONL file."""
        examples = []
        
        with open(pairs_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                pair = json.loads(line)
                
                # InputExample: (texts, label=None) or (texts=[anchor, positive, negative])
                example = InputExample(
                    texts=[
                        pair['anchor'],
                        pair['positive'],
                        pair['negative']
                    ]
                )
                examples.append(example)
        
        logger.info(f"✅ Loaded {len(examples)} training examples from {pairs_file}")
        return examples
    
    def train(
        self,
        training_pairs_file: str,
        num_epochs: int = 3,
        batch_size: int = 32,
        warmup_steps: int = 500,
        learning_rate: float = 2e-5,
        validation_split: float = 0.1
    ) -> Dict:
        """
        Fine-tune model on training pairs.
        
        Args:
            training_pairs_file: Path to training_pairs.jsonl
            num_epochs: Number of training epochs
            batch_size: Batch size
            warmup_steps: Warmup steps
            learning_rate: Learning rate
            validation_split: Fraction for validation
        
        Returns:
            Dictionary with training metrics
        """
        pairs_file = Path(training_pairs_file)
        if not pairs_file.exists():
            raise FileNotFoundError(f"Training pairs file not found: {pairs_file}")
        
        # Load training data
        train_examples = self._load_training_pairs(pairs_file)
        
        # Split into train/val
        n_val = int(len(train_examples) * validation_split)
        random.shuffle(train_examples)
        val_examples = train_examples[:n_val]
        train_examples = train_examples[n_val:]
        
        logger.info(f"Train split: {len(train_examples)} examples")
        logger.info(f"Validation split: {len(val_examples)} examples")
        
        # Create DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )
        
        # Use TripletLoss for contrastive learning
        train_loss = losses.TripletLoss(model=self.model)
        
        # Training configuration
        logger.info("\n=== TRAINING START ===")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Device: {self.device}")
        
        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            checkpoint_path=str(self.output_dir / 'checkpoint'),
            checkpoint_save_total_limit=2,
            show_progress_bar=True,
            use_amp=True  # Mixed precision for faster training
        )
        
        # Save final model
        model_path = self.output_dir / 'final_model'
        logger.info(f"\n✅ Saving final model to {model_path}")
        self.model.save(str(model_path))
        
        # Save training metadata
        metadata = {
            'model_name': self.model_name,
            'training_date': datetime.now().isoformat(),
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'warmup_steps': warmup_steps,
            'total_training_pairs': len(train_examples),
            'total_validation_pairs': len(val_examples),
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'device': self.device
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Metadata saved to {metadata_path}")
        logger.info(f"\n=== TRAINING COMPLETE ===")
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Embedding dimension: {metadata['embedding_dimension']}")
        
        return metadata
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def get_model_path(self) -> Path:
        """Get path to fine-tuned model."""
        return self.output_dir / 'final_model'


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune embedding model')
    parser.add_argument(
        '--training-pairs',
        default='data/processed/training_pairs.jsonl',
        help='Path to training pairs file'
    )
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument(
        '--output-dir',
        default='models/finetuned_embedding',
        help='Output directory'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='Device (auto-detect if not specified)'
    )
    
    args = parser.parse_args()
    
    tuner = EmbeddingFineTuner(
        output_dir=args.output_dir,
        device=args.device
    )
    
    tuner.train(
        training_pairs_file=args.training_pairs,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
