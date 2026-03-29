#!/usr/bin/env python3
"""
PROMPT 12 - Metrics Collection Module
Collects and stores evaluation metrics for each experiment
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval performance"""
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    recall_at_k: Dict[int, float]  # Recall@1, @5, @10
    precision_at_k: Dict[int, float]  # Precision@1, @5, @10
    avg_rank: float  # Average rank of relevant document
    mean_retrieval_time: float  # Milliseconds


@dataclass
class GenerationMetrics:
    """Metrics for answer generation quality"""
    bleu_score: float
    rouge_l: float
    bertscore_f1: float
    answer_length_mean: float
    answer_length_std: float


@dataclass
class HallucinationMetrics:
    """Metrics for hallucination and faithfulness"""
    hallucination_rate: float  # % of hallucinated content
    faithfulness_score: float  # 0-1 how faithful to retrieved docs
    citation_rate: float  # % of statements with citations


@dataclass
class ExperimentMetrics:
    """Complete metrics for one experiment run"""
    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Retrieval metrics
    retrieval: RetrievalMetrics = field(default_factory=lambda: RetrievalMetrics(
        mrr=0.0, ndcg=0.0, recall_at_k={}, precision_at_k={}, 
        avg_rank=0.0, mean_retrieval_time=0.0
    ))
    
    # Generation metrics (if LLM enabled)
    generation: Optional[GenerationMetrics] = None
    
    # Hallucination metrics  
    hallucination: Optional[HallucinationMetrics] = None
    
    # Reranking metrics (if reranker enabled)
    reranking_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Meta info
    num_queries: int = 0
    total_time_seconds: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "num_queries": self.num_queries,
            "total_time_seconds": self.total_time_seconds,
            "notes": self.notes,
            "retrieval": asdict(self.retrieval),
        }
        
        if self.generation:
            data["generation"] = asdict(self.generation)
        if self.hallucination:
            data["hallucination"] = asdict(self.hallucination)
        if self.reranking_metrics:
            data["reranking"] = self.reranking_metrics
            
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class MetricsCollector:
    """Collects and stores metrics during experiment run"""
    
    def __init__(self, experiment_name: str, output_dir: Path):
        """
        Initialize metrics collector
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = ExperimentMetrics(experiment_name=experiment_name)
        self.query_times = []
        self.retrieval_scores = []
        
    def record_retrieval_metrics(
        self,
        mrr: float,
        ndcg: float,
        recall_at_k: Dict[int, float],
        precision_at_k: Dict[int, float],
        avg_rank: float,
        retrieval_time_ms: float,
    ) -> None:
        """Record retrieval metrics"""
        self.metrics.retrieval = RetrievalMetrics(
            mrr=mrr,
            ndcg=ndcg,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            avg_rank=avg_rank,
            mean_retrieval_time=retrieval_time_ms
        )
        logger.info(f"Recorded retrieval metrics: MRR={mrr:.3f}, NDCG={ndcg:.3f}")
    
    def record_generation_metrics(
        self,
        bleu: float,
        rouge_l: float,
        bertscore: float,
        answer_lengths: List[int],
    ) -> None:
        """Record generation metrics"""
        self.metrics.generation = GenerationMetrics(
            bleu_score=bleu,
            rouge_l=rouge_l,
            bertscore_f1=bertscore,
            answer_length_mean=float(np.mean(answer_lengths)),
            answer_length_std=float(np.std(answer_lengths)),
        )
        logger.info(f"Recorded generation metrics: BLEU={bleu:.3f}, ROUGE-L={rouge_l:.3f}")
    
    def record_hallucination_metrics(
        self,
        hallucination_rate: float,
        faithfulness_score: float,
        citation_rate: float,
    ) -> None:
        """Record hallucination/faithfulness metrics"""
        self.metrics.hallucination = HallucinationMetrics(
            hallucination_rate=hallucination_rate,
            faithfulness_score=faithfulness_score,
            citation_rate=citation_rate,
        )
        logger.info(f"Recorded hallucination metrics: rate={hallucination_rate:.1%}, faithfulness={faithfulness_score:.3f}")
    
    def record_reranking_metrics(self, metrics: Dict[str, float]) -> None:
        """Record reranking-specific metrics"""
        self.metrics.reranking_metrics = metrics
        logger.info(f"Recorded reranking metrics: {metrics}")
    
    def record_query_time(self, time_ms: float) -> None:
        """Record individual query execution time"""
        self.query_times.append(time_ms)
    
    def record_retrieval_score(self, score: float) -> None:
        """Record individual retrieval score"""
        self.retrieval_scores.append(score)
    
    def finalize(
        self,
        num_queries: int,
        total_time_seconds: float,
        notes: str = ""
    ) -> None:
        """Finalize metrics collection"""
        self.metrics.num_queries = num_queries
        self.metrics.total_time_seconds = total_time_seconds
        self.metrics.notes = notes
        
        logger.info(f"Metrics finalized: {num_queries} queries in {total_time_seconds:.1f}s")
    
    def save(self) -> Path:
        """Save metrics to JSON file"""
        output_file = (
            self.output_dir / f"{self.experiment_name}_metrics.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(self.metrics.to_json())
        
        logger.info(f"Saved metrics to {output_file}")
        return output_file
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of metrics for comparison"""
        summary = {
            "experiment": self.experiment_name,
            "timestamp": self.metrics.timestamp,
            "mrr": self.metrics.retrieval.mrr,
            "ndcg": self.metrics.retrieval.ndcg,
            "mean_retrieval_time_ms": self.metrics.retrieval.mean_retrieval_time,
        }
        
        if self.metrics.retrieval.recall_at_k:
            summary["recall@5"] = self.metrics.retrieval.recall_at_k.get(5, 0.0)
            summary["recall@10"] = self.metrics.retrieval.recall_at_k.get(10, 0.0)
        
        if self.metrics.generation:
            summary["bleu"] = self.metrics.generation.bleu_score
            summary["rouge_l"] = self.metrics.generation.rouge_l
            summary["bertscore"] = self.metrics.generation.bertscore_f1
        
        if self.metrics.hallucination:
            summary["hallucination_rate"] = self.metrics.hallucination.hallucination_rate
            summary["faithfulness"] = self.metrics.hallucination.faithfulness_score
        
        return summary


def load_metrics(path: Path) -> ExperimentMetrics:
    """Load metrics from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metrics = ExperimentMetrics(
        experiment_name=data['experiment_name'],
        timestamp=data.get('timestamp', '')
    )
    
    if 'retrieval' in data:
        ret_data = data['retrieval']
        metrics.retrieval = RetrievalMetrics(
            mrr=ret_data.get('mrr', 0.0),
            ndcg=ret_data.get('ndcg', 0.0),
            recall_at_k=ret_data.get('recall_at_k', {}),
            precision_at_k=ret_data.get('precision_at_k', {}),
            avg_rank=ret_data.get('avg_rank', 0.0),
            mean_retrieval_time=ret_data.get('mean_retrieval_time', 0.0),
        )
    
    if 'generation' in data:
        gen_data = data['generation']
        metrics.generation = GenerationMetrics(
            bleu_score=gen_data.get('bleu_score', 0.0),
            rouge_l=gen_data.get('rouge_l', 0.0),
            bertscore_f1=gen_data.get('bertscore_f1', 0.0),
            answer_length_mean=gen_data.get('answer_length_mean', 0.0),
            answer_length_std=gen_data.get('answer_length_std', 0.0),
        )
    
    if 'hallucination' in data:
        hall_data = data['hallucination']
        metrics.hallucination = HallucinationMetrics(
            hallucination_rate=hall_data.get('hallucination_rate', 0.0),
            faithfulness_score=hall_data.get('faithfulness_score', 0.0),
            citation_rate=hall_data.get('citation_rate', 0.0),
        )
    
    metrics.num_queries = data.get('num_queries', 0)
    metrics.total_time_seconds = data.get('total_time_seconds', 0.0)
    metrics.notes = data.get('notes', '')
    metrics.reranking_metrics = data.get('reranking', {})
    
    return metrics
