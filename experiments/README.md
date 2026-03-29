# experiments/ - Ablation Study Framework

This directory contains the complete ablation study framework for systematic evaluation of the Turkish Legal RAG system.

## Files

- **`ablation_runner.py`** - Main CLI orchestrator for running experiments
- **`experiment_configs.py`** - Definitions of 5 ablation experiments
- **`metrics_collector.py`** - Metrics collection and storage system
- **`results_visualizer.py`** - Comparison table and report generation
- **`__init__.py`** - Package initialization

## Quick Start

```bash
# Run all 5 experiments
python ablation_runner.py --all

# Run specific experiment
python ablation_runner.py --exp baseline

# Generate comparison reports
python ablation_runner.py --compare

# List available experiments
python ablation_runner.py --list
```

## The 5 Experiments

1. **Baseline RAG** - Dense + Sparse retrieval only
2. **+ Embedding Tuning** - With fine-tuned embeddings
3. **+ Reranker** - With cross-encoder reranker
4. **+ LLM Fine-tuning** - With fine-tuned answer generation
5. **Fully Optimized** - All components enabled

## Results

Results are saved to `ablation_results/` directory with automatic comparison reports:
- `COMPARISON_RESULTS.csv` - Comparison table (can open in Excel)
- `ABLATION_RESULTS.md` - Full markdown report

## Documentation

See `../PROMPT_12_ABLATION_GUIDE.md` for complete documentation.

## Configuration

To customize experiments, edit `experiment_configs.py`:

```python
BASELINE_RAG = ExperimentConfig(
    name="01_baseline_rag",
    retriever=RetrieverConfig(
        dense_weight=0.6,  # Customize weights
        k_dense=20,        # Customize k values
        # ...
    ),
    # ...
)
```

## Understanding Results

Key metrics:
- **MRR** - Mean reciprocal rank (how quickly relevant docs appear)
- **NDCG** - Overall ranking quality
- **BLEU/ROUGE-L** - Answer generation quality
- **Hallucination Rate** - % of incorrect/contradicting content

See documentation for detailed interpretation guide.

---

**Status:** ✅ Production Ready  
**Version:** 1.0.0
