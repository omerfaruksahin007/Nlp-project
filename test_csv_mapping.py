#!/usr/bin/env python3
"""Quick test for Turkish CSV with field mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.ingestion.pipeline import IngestionPipeline

# Turkish CSV has: soru, cevap, veri türü, kaynak, context, Score
field_mapping = {
    'question': 'soru',
    'answer': 'cevap',
    'source': 'kaynak',
    'category': 'veri türü'
}

pipeline = IngestionPipeline(output_dir=Path('data/processed'))
stats = pipeline.ingest_qa_dataset(
    input_path=Path('data/raw/turkish_law_dataset.csv'),
    output_name='turkish_law_mapped',
    source_name='turkish_legal_corpus',
    field_mapping=field_mapping,
    remove_duplicates=True,
)

print(f'✅ Ingestion complete!')
print(f'   Total loaded: {stats["total_loaded"]}')
print(f'   Duplicates removed: {stats["duplicates_removed"]}')
print(f'   Final count: {stats["final_count"]}')
