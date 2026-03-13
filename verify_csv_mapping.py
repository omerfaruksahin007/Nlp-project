#!/usr/bin/env python3
"""Verify CSV field mapping and conversion."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.ingestion.loader import load_csv
from src.ingestion.converters import QAConverter

# Step 1: Load CSV with field mapping
print("="*80)
print("STEP 1: Loading CSV with Turkish field mapping")
print("="*80)

csv_path = Path('data/raw/turkish_law_dataset.csv')
raw_data = load_csv(csv_path)

print(f"✅ Loaded {len(raw_data)} rows from CSV")
print(f"   CSV columns: {list(raw_data[0].keys())}")

# Step 2: Convert with custom Turkish field mapping
print("\n" + "="*80)
print("STEP 2: Converting with Turkish field mapping")
print("="*80)

field_mapping = {
    'question': 'soru',
    'answer': 'cevap',
    'source': 'kaynak',
    'category': 'veri türü'
}

converter = QAConverter(field_mapping=field_mapping)
converted_data = converter.convert_batch(raw_data, source_name='turkish_legal_corpus')

print(f"✅ Converted {len(converted_data)}/{len(raw_data)} records")
print(f"   Skipped: {len(raw_data) - len(converted_data)} rows")

# Step 3: Save output
print("\n" + "="*80)
print("STEP 3: Saving normalized output")
print("="*80)

output_path = Path('data/processed/turkish_law_dataset_verified.jsonl')
with open(output_path, 'w', encoding='utf-8') as f:
    for record in converted_data:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"✅ Saved to: {output_path}")
print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

# Step 4: Show first 3 records
print("\n" + "="*80)
print("STEP 4: First 3 normalized records")
print("="*80)

for i, record in enumerate(converted_data[:3], 1):
    print(f"\nRecord {i}:")
    print(f"  id:       {record['id']}")
    print(f"  question: {record['question'][:60]}...")
    print(f"  answer:   {record['answer'][:60]}...")
    print(f"  source:   {record['source']}")
    print(f"  category: {record.get('category', 'N/A')}")
    print(f"  citation: {record.get('citation', 'N/A')}")

# Step 5: Validate no empty fields
print("\n" + "="*80)
print("STEP 5: Field validation")
print("="*80)

empty_questions = sum(1 for r in converted_data if not r.get('question', '').strip())
empty_answers = sum(1 for r in converted_data if not r.get('answer', '').strip())
broken_records = sum(1 for r in converted_data if not r.get('id') or not r.get('source'))

print(f"✅ Empty questions: {empty_questions}")
print(f"✅ Empty answers: {empty_answers}")
print(f"✅ Broken records (missing id or source): {broken_records}")

if empty_questions == 0 and empty_answers == 0 and broken_records == 0:
    print("\n✅ ALL RECORDS VALIDATED - No empty or broken fields!")
else:
    print("\n⚠️ ISSUES FOUND - Review validation above")

# Step 6: Summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Total rows loaded:     {len(raw_data):,}")
print(f"Total rows converted:  {len(converted_data):,}")
print(f"Total rows skipped:    {len(raw_data) - len(converted_data):,}")
print(f"Output file:           {output_path}")
print(f"Status:                ✅ SUCCESS")
print("="*80)
