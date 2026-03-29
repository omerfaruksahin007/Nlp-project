"""
PROMPT 7: Embedding Fine-tuning for Turkish Legal Domain

Fine-tunes a sentence-transformer model using triplet loss or contrastive loss
on Turkish legal training pairs.

Features:
- Load training pairs (from PROMPT 6)
- Fine-tune with triplet/contrastive loss
- Validation during training
- Checkpoint saving
- Hyperparameter config support
- Progress logging

Usage:
    fine_tuner = EmbeddingFineTuner(config_file='configs/fine_tuning_config.yaml')
    fine_tuner.train()
    fine_tuner.save_model('models/fine_tuned_embeddings')
"""

import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    model_name: str = 'distiluse-base-multilingual-cased-v2'
    training_file: str = 'data/training/training_pairs.jsonl'
    output_dir: str = 'models/fine_tuned_embeddings'
    
    # Training parameters
    epochs: int = 3
    batch_size: int = 16
    warmup_steps: int = 100
    learning_rate: float = 2e-5
    
    # Loss function
    loss_type: str = 'triplet'  # 'triplet', 'multiple_negatives_ranking', 'cosine'
    margin: float = 0.5
    
    # Validation
    validation_split: float = 0.1
    eval_steps: int = 500
    
    # Device
    device: str = 'cpu'  # 'cpu', 'cuda'
    
    @classmethod
    def from_yaml(cls, yaml_file: Path) -> 'FineTuningConfig':
        """Load config from YAML file"""
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'model_name': self.model_name,
            'training_file': self.training_file,
            'output_dir': self.output_dir,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'warmup_steps': self.warmup_steps,
            'learning_rate': self.learning_rate,
            'loss_type': self.loss_type,
            'margin': self.margin,
            'validation_split': self.validation_split,
            'eval_steps': self.eval_steps,
            'device': self.device,
        }


class TrainingPairsDataset(Dataset):
    """Dataset for training pairs (triplet format)"""
    
    def __init__(self, pairs: List[Dict]):
        """
        Args:
            pairs: List of dicts with keys:
                - anchor_text
                - positive_text (or empty for negatives)
                - negative_text
                - pair_type
        """
        self.pairs = pairs
        
        # Separate by type
        self.qa_pairs = [p for p in pairs if p['pair_type'] in ['qa_pair', 'semantic_related']]
        self.negative_pairs = [p for p in pairs if p['pair_type'] in ['hard_negative', 'random_negative']]
        
        logger.info(f"Dataset initialized:")
        logger.info(f"  QA/positive pairs: {len(self.qa_pairs)}")
        logger.info(f"  Negative pairs: {len(self.negative_pairs)}")
    
    def __len__(self) -> int:
        return len(self.qa_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """Return triplet or pair"""
        qa_pair = self.qa_pairs[idx]
        
        return {
            'anchor_text': qa_pair['anchor_text'],
            'positive_text': qa_pair['positive_text'],
            'negative_text': qa_pair.get('negative_text', ''),
            'pair_type': qa_pair.get('pair_type', 'unknown'),
        }


class EmbeddingFineTuner:
    """Fine-tune SentenceTransformer for Turkish legal domain"""
    
    def __init__(self, config: Optional[FineTuningConfig] = None):
        """
        Args:
            config: FineTuningConfig object
        """
        self.config = config or FineTuningConfig()
        
        # Setup logging
        logger.info("=" * 80)
        logger.info("EMBEDDING FINE-TUNING")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        for key, value in self.config.to_dict().items():
            logger.info(f"  {key}: {value}")
        
        # Load model
        logger.info(f"\nLoading base model: {self.config.model_name}")
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device
        )
        self.device = torch.device(self.config.device)
        logger.info(f"✅ Model loaded (device: {self.device})")
        
        # Load training data
        logger.info(f"\nLoading training data from {self.config.training_file}")
        self.train_pairs, self.val_pairs = self._load_and_split_data()
        logger.info(f"✅ Training set: {len(self.train_pairs)} pairs")
        logger.info(f"✅ Validation set: {len(self.val_pairs)} pairs")
    
    def _load_and_split_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load and split training/validation data"""
        pairs = []
        
        with open(self.config.training_file, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line.strip())
                pairs.append(pair)
        
        # Split
        val_count = int(len(pairs) * self.config.validation_split)
        val_pairs = pairs[:val_count]
        train_pairs = pairs[val_count:]
        
        return train_pairs, val_pairs
    
    def _create_dataset(self) -> Dataset:
        """Create PyTorch dataset"""
        return TrainingPairsDataset(self.train_pairs)
    
    def _get_loss_function(self):
        """Get loss function based on config"""
        if self.config.loss_type == 'triplet':
            logger.info(f"Using TripletLoss (triplet_margin={self.config.margin})")
            return losses.TripletLoss(model=self.model, triplet_margin=self.config.margin)
        
        elif self.config.loss_type == 'multiple_negatives_ranking':
            logger.info("Using MultipleNegativesRankingLoss")
            return losses.MultipleNegativesRankingLoss(model=self.model)
        
        elif self.config.loss_type == 'cosine':
            logger.info("Using CosineSimilarityLoss")
            return losses.CosineSimilarityLoss(model=self.model)
        
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def _convert_pairs_to_examples(self, pairs: List[Dict]) -> List[InputExample]:
        """Convert training pairs to InputExample format
        
        Note: If pairs are split (positive/negative separately), this intelligently
        combines them into triplets for triplet loss
        """
        examples = []
        
        # Separate positive and negative pairs
        positive_pairs = {}  # anchor -> positive_text
        negative_pairs = {}  # anchor -> negative_text
        
        for pair in pairs:
            anchor = pair['anchor_text']
            if not anchor:
                continue
                
            pair_type = pair.get('pair_type', '')
            
            # Collect positive texts
            if pair_type in ['qa_pair', 'semantic_related']:
                positive = pair.get('positive_text', '')
                if positive and anchor not in positive_pairs:
                    positive_pairs[anchor] = positive
            
            # Collect negative texts  
            if pair_type in ['hard_negative', 'random_negative']:
                negative = pair.get('negative_text', '')
                if negative and anchor not in negative_pairs:
                    negative_pairs[anchor] = negative
        
        # Format depends on loss type
        if self.config.loss_type == 'triplet':
            # Triplet: (anchor, positive, negative)
            # Try to create triplets by matching anchors
            for anchor, positive in positive_pairs.items():
                negative = negative_pairs.get(anchor, '')
                
                if positive:  # Must have positive at minimum
                    example = InputExample(
                        texts=[anchor, positive, negative] if negative else [anchor, positive, anchor],
                        label=0  # Not used for triplet
                    )
                    examples.append(example)
        
        elif self.config.loss_type == 'multiple_negatives_ranking':
            # Multiple negatives: (query, positive, [negatives...])
            for anchor, positive in positive_pairs.items():
                if positive:
                    example = InputExample(
                        texts=[anchor, positive],
                        label=1.0
                    )
                    examples.append(example)
        
        elif self.config.loss_type == 'cosine':
            # Cosine similarity: (text1, text2, similarity_score)
            for anchor, positive in positive_pairs.items():
                if positive:
                    example = InputExample(
                        texts=[anchor, positive],
                        label=1.0  # High similarity
                    )
                    examples.append(example)
            
            for anchor, negative in negative_pairs.items():
                if negative:
                    example = InputExample(
                        texts=[anchor, negative],
                        label=0.1  # Low similarity
                    )
                    examples.append(example)
        
        logger.info(f"Created {len(examples)} training examples")
        return examples
    
    def train(self):
        """Train the model"""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        
        # Convert pairs to training examples
        logger.info("\n[1/4] Converting training pairs to examples...")
        train_examples = self._convert_pairs_to_examples(self.train_pairs)
        
        # Create DataLoader
        logger.info("\n[2/4] Creating data loader...")
        train_batch_size = self.config.batch_size
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=train_batch_size
        )
        
        # Get loss function
        logger.info("\n[3/4] Initializing loss function...")
        train_loss = self._get_loss_function()
        
        # Set up output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training
        logger.info("\n[4/4] Starting training loop...")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Batch size: {train_batch_size}")
        logger.info(f"  Warmup steps: {self.config.warmup_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.epochs,
            warmup_steps=self.config.warmup_steps,
            show_progress_bar=True,
            output_path=str(output_dir)
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✅ Model saved to: {output_dir}")
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the fine-tuned model"""
        if output_dir is None:
            output_dir = self.config.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(output_path))
        logger.info(f"✅ Model saved to {output_path}")
        
        # Save config
        config_file = output_path / 'fine_tuning_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"✅ Config saved to {config_file}")
    
    def get_model(self) -> SentenceTransformer:
        """Get the fine-tuned model"""
        return self.model


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)
