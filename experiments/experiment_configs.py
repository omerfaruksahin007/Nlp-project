#!/usr/bin/env python3
"""
PROMPT 12 - Ablation Experiment Runner
Turkish Legal RAG - Experiment Configuration System

This module defines all experiment configurations for ablation studies.
Each experiment configuration is independent and reproducible.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List
from pathlib import Path


@dataclass
class RetrieverConfig:
    """Configuration for retriever component"""
    dense_enabled: bool = True
    sparse_enabled: bool = True
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    k_dense: int = 20
    k_sparse: int = 20
    k_final: int = 10
    fusion_method: str = "rrf"  # 'rrf' or 'weighted'


@dataclass
class RerankerConfig:
    """Configuration for reranker component"""
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-base"
    top_k: int = 5
    threshold: float = 0.5


@dataclass
class LLMConfig:
    """Configuration for LLM fine-tuning and generation"""
    enabled: bool = False
    fine_tuned: bool = False
    model_name: str = "meta-llama/Llama-2-7b"
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fine_tuned: bool = False
    fine_tuned_path: str = "models/fine_tuned_embeddings"
    dimension: int = 512


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str
    description: str
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    batch_size: int = 32
    num_test_queries: int = 100
    random_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            description=data['description'],
            retriever=RetrieverConfig(**data.get('retriever', {})),
            reranker=RerankerConfig(**data.get('reranker', {})),
            llm=LLMConfig(**data.get('llm', {})),
            embedding=EmbeddingConfig(**data.get('embedding', {})),
            batch_size=data.get('batch_size', 32),
            num_test_queries=data.get('num_test_queries', 100),
            random_seed=data.get('random_seed', 42),
        )


# ==============================================================================
# ABLATION STUDY EXPERIMENTS (5 variations)
# ==============================================================================

# [1] BASELINE RAG - Dense + Sparse retrieval only
BASELINE_RAG = ExperimentConfig(
    name="01_baseline_rag",
    description="Baseline: Dense (60%) + Sparse (40%) RRF fusion retrieval",
    retriever=RetrieverConfig(
        dense_enabled=True,
        sparse_enabled=True,
        dense_weight=0.6,
        sparse_weight=0.4,
        k_dense=20,
        k_sparse=20,
        k_final=10,
    ),
    reranker=RerankerConfig(enabled=False),
    llm=LLMConfig(enabled=False),
    embedding=EmbeddingConfig(fine_tuned=False),
)

# [2] BASELINE + EMBEDDING TUNING
BASELINE_EMBEDDING = ExperimentConfig(
    name="02_baseline_embedding_tuning",
    description="Baseline RAG + Fine-tuned embedding model",
    retriever=RetrieverConfig(
        dense_enabled=True,
        sparse_enabled=True,
        dense_weight=0.6,
        sparse_weight=0.4,
        k_dense=20,
        k_sparse=20,
        k_final=10,
    ),
    reranker=RerankerConfig(enabled=False),
    llm=LLMConfig(enabled=False),
    embedding=EmbeddingConfig(
        fine_tuned=True,
        fine_tuned_path="models/fine_tuned_embeddings"
    ),
)

# [3] BASELINE + RERANKER
BASELINE_RERANKER = ExperimentConfig(
    name="03_baseline_reranker",
    description="Baseline RAG + Cross-encoder reranker",
    retriever=RetrieverConfig(
        dense_enabled=True,
        sparse_enabled=True,
        dense_weight=0.6,
        sparse_weight=0.4,
        k_dense=20,
        k_sparse=20,
        k_final=10,
    ),
    reranker=RerankerConfig(
        enabled=True,
        model_name="BAAI/bge-reranker-base",
        top_k=5,
        threshold=0.5
    ),
    llm=LLMConfig(enabled=False),
    embedding=EmbeddingConfig(fine_tuned=False),
)

# [4] BASELINE + LLM FINE-TUNING
BASELINE_LLM_FINETUNING = ExperimentConfig(
    name="04_baseline_llm_finetuning",
    description="Baseline RAG + Fine-tuned LLM for answer generation",
    retriever=RetrieverConfig(
        dense_enabled=True,
        sparse_enabled=True,
        dense_weight=0.6,
        sparse_weight=0.4,
        k_dense=20,
        k_sparse=20,
        k_final=10,
    ),
    reranker=RerankerConfig(enabled=False),
    llm=LLMConfig(
        enabled=True,
        fine_tuned=True,
        model_name="meta-llama/Llama-2-7b",
        temperature=0.7,
        max_tokens=512
    ),
    embedding=EmbeddingConfig(fine_tuned=False),
)

# [5] FULLY OPTIMIZED - All components enabled
FULLY_OPTIMIZED = ExperimentConfig(
    name="05_fully_optimized",
    description="Full system: Fine-tuned embeddings + Reranker + Fine-tuned LLM",
    retriever=RetrieverConfig(
        dense_enabled=True,
        sparse_enabled=True,
        dense_weight=0.65,  # Slightly higher weight for fine-tuned embeddings
        sparse_weight=0.35,
        k_dense=20,
        k_sparse=20,
        k_final=10,
    ),
    reranker=RerankerConfig(
        enabled=True,
        model_name="BAAI/bge-reranker-base",
        top_k=5,
        threshold=0.5
    ),
    llm=LLMConfig(
        enabled=True,
        fine_tuned=True,
        model_name="meta-llama/Llama-2-7b",
        temperature=0.7,
        max_tokens=512
    ),
    embedding=EmbeddingConfig(
        fine_tuned=True,
        fine_tuned_path="models/fine_tuned_embeddings"
    ),
)


# Registry of all experiments
EXPERIMENTS = {
    "baseline": BASELINE_RAG,
    "embedding": BASELINE_EMBEDDING,
    "reranker": BASELINE_RERANKER,
    "llm": BASELINE_LLM_FINETUNING,
    "full": FULLY_OPTIMIZED,
}

EXPERIMENT_NAMES = {
    "01_baseline_rag": BASELINE_RAG,
    "02_baseline_embedding_tuning": BASELINE_EMBEDDING,
    "03_baseline_reranker": BASELINE_RERANKER,
    "04_baseline_llm_finetuning": BASELINE_LLM_FINETUNING,
    "05_fully_optimized": FULLY_OPTIMIZED,
}


def get_experiment(name: str) -> ExperimentConfig:
    """Get experiment by short name or full name"""
    if name in EXPERIMENTS:
        return EXPERIMENTS[name]
    if name in EXPERIMENT_NAMES:
        return EXPERIMENT_NAMES[name]
    raise ValueError(f"Unknown experiment: {name}")


def list_experiments() -> List[str]:
    """List all available experiment names"""
    return list(EXPERIMENTS.keys())


def save_experiment_config(config: ExperimentConfig, path: Path) -> None:
    """Save experiment configuration to JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(config.to_json())


if __name__ == "__main__":
    print("=" * 70)
    print("ABLATION STUDY - EXPERIMENT CONFIGURATIONS")
    print("=" * 70)
    
    for name, config in EXPERIMENTS.items():
        print(f"\n[{config.name}] {config.description}")
        print(f"  - Embedding fine-tuned: {config.embedding.fine_tuned}")
        print(f"  - Reranker enabled: {config.reranker.enabled}")
        print(f"  - LLM fine-tuned: {config.llm.fine_tuned}")
        print(f"  - Dense/Sparse ratio: {config.retriever.dense_weight}/{config.retriever.sparse_weight}")
