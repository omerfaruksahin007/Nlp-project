"""
LoRA/QLoRA Fine-tuning for Turkish Legal Answer Generation

Implements efficient fine-tuning using LoRA (Low-Rank Adaptation)
and QLoRA (Quantized LoRA) for resource-constrained environments.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    # LoRA hyperparameters
    r: int = 16                          # LoRA rank
    lora_alpha: int = 32                 # LoRA alpha (scaling)
    lora_dropout: float = 0.05           # Dropout in LoRA layers
    bias: str = "none"                   # 'none', 'all', or 'lora_only'
    task_type: str = "CAUSAL_LM"         # Task type
    
    # Module selection (which layers to apply LoRA to)
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    @property
    def peft_config(self):
        """Get PEFT LoRA config"""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.target_modules
        )


@dataclass
class QLoRAConfig(LoRAConfig):
    """QLoRA configuration (LoRA + Quantization)"""
    # Quantization bits (4 for QLoRA, 8 for semi-quantization)
    bits: int = 4
    use_bnb_4bit: bool = True


class LoRAFinetuner:
    """
    LoRA/QLoRA fine-tuner for language models
    
    Supports:
    - LoRA: Add trainable low-rank matrices
    - QLoRA: LoRA + 4-bit quantization (reduced memory)
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        use_qlora: bool = True,
        lora_config: Optional[LoRAConfig] = None,
        device_map: str = "auto"
    ):
        """
        Initialize fine-tuner
        
        Parameters:
        -----------
        model_name: str
            HuggingFace model ID
        use_qlora: bool
            Use QLoRA (quantized) vs regular LoRA
        lora_config: Optional[LoRAConfig]
            LoRA configuration (uses default if None)
        device_map: str
            Device mapping strategy ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.use_qlora = use_qlora
        self.lora_config = lora_config or LoRAConfig()
        self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        logger.info(f"Initialized LoRA{'Q' if use_qlora else ''}Finetuner")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  LoRA rank: {self.lora_config.r}")
        logger.info(f"  LoRA alpha: {self.lora_config.lora_alpha}")
    
    def load_model_and_tokenizer(self) -> tuple:
        """
        Load model and tokenizer from HuggingFace
        
        Returns:
        --------
        tuple: (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if self.use_qlora:
            # QLoRA: 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
                attn_implementation="eager"  # For stability
            )
            logger.info("✅ Loaded with QLoRA (4-bit quantization)")
        else:
            # Regular LoRA (full precision)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info("✅ Loaded with LoRA (float16)")
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config.peft_config)
        
        logger.info(f"✅ Applied LoRA (rank={self.lora_config.r})")
        logger.info(f"Trainable params: {self._count_trainable_params():,}")
        
        return self.model, self.tokenizer
    
    def _count_trainable_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def prepare_dataset(
        self,
        train_path: Path,
        val_path: Optional[Path] = None,
        max_length: int = 512
    ) -> Dict[str, Dataset]:
        """
        Prepare training dataset
        
        Parameters:
        -----------
        train_path: Path
            Path to training JSONL file
        val_path: Optional[Path]
            Path to validation JSONL file
        max_length: int
            Maximum sequence length
        
        Returns:
        --------
        Dict[str, Dataset]: {split_name: dataset}
        """
        logger.info("Preparing dataset...")
        
        def load_jsonl(path: Path) -> Dataset:
            """Load JSONL file as HuggingFace Dataset"""
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            dataset = Dataset.from_dict({
                'text': [
                    f"Instruction: {d['instruction']}\nContext: {d['context']}\n"
                    f"Response: {d['output']}"
                    for d in data
                ]
            })
            return dataset
        
        # Load datasets
        train_dataset = load_jsonl(train_path)
        val_dataset = load_jsonl(val_path) if val_path else None
        
        logger.info(f"Loaded {len(train_dataset)} training examples")
        if val_dataset:
            logger.info(f"Loaded {len(val_dataset)} validation examples")
        
        # Tokenize
        def tokenize_fn(example):
            result = self.tokenizer(
                example['text'],
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            result['labels'] = result['input_ids'].copy()
            return result
        
        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=False,
            remove_columns=['text']
        )
        
        if val_dataset:
            val_dataset = val_dataset.map(
                tokenize_fn,
                batched=False,
                remove_columns=['text']
            )
        
        datasets = {'train': train_dataset}
        if val_dataset:
            datasets['eval'] = val_dataset
        
        logger.info("✅ Dataset prepared")
        return datasets
    
    def train(
        self,
        datasets: Dict[str, Dataset],
        output_dir: Path,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        save_strategy: str = "epoch",
        logging_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Fine-tune model using LoRA
        
        Parameters:
        -----------
        datasets: Dict[str, Dataset]
            Training and validation datasets
        output_dir: Path
            Directory to save model and results
        num_epochs: int
            Number of training epochs
        batch_size: int
            Training batch size
        learning_rate: float
            Learning rate
        warmup_steps: int
            Number of warmup steps
        save_strategy: str
            Save strategy ('steps', 'epoch', 'no')
        logging_steps: int
            Logging frequency
        
        Returns:
        --------
        Dict[str, Any]: Training results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*70}")
        logger.info("STARTING LoRA FINE-TUNING")
        logger.info(f"{'='*70}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=100 if save_strategy == "steps" else None,
            save_total_limit=2,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            seed=42,
            logging_dir=str(output_dir / "logs"),
            logging_first_step=True,
            max_grad_norm=1.0,
            fp16=True,
            push_to_hub=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets.get('eval'),
            data_collator=data_collator,
            callbacks=[]
        )
        
        # Train
        logger.info("Starting training loop...")
        train_result = self.trainer.train()
        
        logger.info(f"\n{'='*70}")
        logger.info("TRAINING COMPLETED")
        logger.info(f"{'='*70}")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        return {
            'training_loss': train_result.training_loss,
            'training_time': train_result.training_steps
        }
    
    def save_model(self, output_dir: Path):
        """
        Save fine-tuned model and tokenizer
        
        Parameters:
        -----------
        output_dir: Path
            Directory to save model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        
        # Save LoRA weights
        self.model.save_pretrained(str(output_dir))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_dir))
        
        logger.info("✅ Model saved")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate text using fine-tuned model
        
        Parameters:
        -----------
        prompt: str
            Input prompt
        max_length: int
            Maximum generation length
        temperature: float
            Sampling temperature
        top_p: float
            Top-p sampling
        num_return_sequences: int
            Number of sequences to generate
        
        Returns:
        --------
        str: Generated text
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return generated_texts[0] if num_return_sequences == 1 else generated_texts


if __name__ == "__main__":
    # Test
    print("LoRA Fine-tuner loaded successfully")
