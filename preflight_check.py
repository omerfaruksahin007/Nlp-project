#!/usr/bin/env python3
"""
Pre-flight checklist for COLAB_ABLATION_FULL deployment
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

print('=== PRE-FLIGHT CHECKLIST ===\n')

checks = {
    'Training pairs': False,
    'embedding_finetuner': False,
    'ablation_runner': False,
    'experiment_configs': False,
    'retrieval_modules': False,
    'answer_generator': False,
    'reranker': False,
    'evaluation': False,
    'config_files': False,
    'FAISS_baseline': False,
    'BM25_baseline': False,
    'Notebook': False
}

# 1. Training pairs
try:
    with open('data/processed/training_pairs.jsonl', 'r', encoding='utf-8') as f:
        pair_count = sum(1 for _ in f)
    print(f'✅ Training pairs: {pair_count} pairs')
    checks['Training pairs'] = True
except Exception as e:
    print(f'❌ Training pairs: {str(e)[:50]}')

# 2. Embedding finetuner
try:
    from src.training.embedding_finetuner import EmbeddingFineTuner
    print('✅ embedding_finetuner.py: imported')
    checks['embedding_finetuner'] = True
except Exception as e:
    print(f'❌ embedding_finetuner: {str(e)[:50]}')

# 3. Ablation runner
try:
    from experiments.ablation_runner import AblationRunner
    print('✅ ablation_runner.py: imported')
    checks['ablation_runner'] = True
except Exception as e:
    print(f'❌ ablation_runner: {str(e)[:50]}')

# 4. Experiment configs
try:
    from experiments.experiment_configs import (
        BASELINE_RAG, BASELINE_EMBEDDING, BASELINE_RERANKER,
        BASELINE_LLM_FINETUNING, FULLY_OPTIMIZED
    )
    print('✅ experiment_configs: all 5 experiments')
    checks['experiment_configs'] = True
except Exception as e:
    print(f'❌ experiment_configs: {str(e)[:50]}')

# 5. Retrieval
try:
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.sparse import SparseRetriever
    from src.retrieval.hybrid import HybridRetriever
    print('✅ Retrieval: dense, sparse, hybrid')
    checks['retrieval_modules'] = True
except Exception as e:
    print(f'❌ Retrieval: {str(e)[:50]}')

# 6. Generation
try:
    from src.generation.answer_generator import AnswerGenerator
    print('✅ answer_generator.py: imported')
    checks['answer_generator'] = True
except Exception as e:
    print(f'❌ answer_generator: {str(e)[:50]}')

# 7. Reranker
try:
    from src.reranking.cross_encoder import CrossEncoderReranker
    print('✅ cross_encoder.py: imported')
    checks['reranker'] = True
except Exception as e:
    print(f'❌ reranker: {str(e)[:50]}')

# 8. Evaluation
try:
    from src.evaluation.framework import RAGEvaluationFramework
    from src.evaluation.metrics import RetrievalEvaluator
    print('✅ evaluation: framework + metrics')
    checks['evaluation'] = True
except Exception as e:
    print(f'❌ evaluation: {str(e)[:50]}')

# 9. Config files
missing = []
for cfg in ['configs/generation_config.yaml', 'configs/retrieval_config.yaml',
            'configs/training_config.yaml', 'configs/ingestion_config.yaml']:
    if not Path(cfg).exists():
        missing.append(cfg)

if not missing:
    print('✅ Config files: all 4 present')
    checks['config_files'] = True
else:
    print(f'⚠️  Missing configs: {", ".join(missing)}')

# 10. FAISS + BM25
if Path('models/faiss_index.index').exists():
    print('✅ FAISS baseline index: exists')
    checks['FAISS_baseline'] = True
else:
    print('⚠️  FAISS baseline: will be built in Colab hücre 6')

if Path('models/bm25_index.pkl').exists():
    print('✅ BM25 baseline index: exists')
    checks['BM25_baseline'] = True
else:
    print('⚠️  BM25 baseline: will be built in Colab hücre 6')

# 11. Notebook
if Path('COLAB_ABLATION_FULL.ipynb').exists():
    print('✅ COLAB_ABLATION_FULL.ipynb: exists')
    checks['Notebook'] = True
else:
    print('❌ COLAB_ABLATION_FULL.ipynb: MISSING')

# Summary
print('\n=== SUMMARY ===')
passed = sum(1 for v in checks.values() if v)
total = len(checks)
print(f'Passed: {passed}/{total}')

critical = ['Training pairs', 'embedding_finetuner', 'ablation_runner', 
            'experiment_configs', 'retrieval_modules', 'evaluation', 'Notebook']
critical_passed = sum(1 for k in critical if checks[k])
print(f'Critical checks: {critical_passed}/{len(critical)}')

if critical_passed == len(critical):
    print('\n🚀 READY FOR COLAB!')
else:
    print(f'\n⚠️  Missing {len(critical) - critical_passed} critical components')
