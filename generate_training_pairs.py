#!/usr/bin/env python3
"""
Generate 13,954 training pairs from Turkish legal chunks
Format: (question, answer, random_answer) triplets
"""

import json
import random
from pathlib import Path

print('⏳ Loading 13,954 chunks...')
chunks = []
with open('data/processed/turkish_law.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))

print(f'✅ Loaded {len(chunks)} chunks')

# Strategy: Her chunk → anchor(question) + positive(answer) + negative(random)
print('⏳ Generating training pairs...')
training_pairs = []

for idx, chunk in enumerate(chunks):
    if idx % 2000 == 0:
        print(f'  Progress: {idx}/{len(chunks)}')
    
    question = chunk.get('question', '')
    answer = chunk.get('answer', '')
    
    if not question or not answer:
        continue
    
    # Positive: this chunk's answer
    positive = answer
    
    # Negative: random different answer
    neg_chunk = random.choice(chunks)
    while neg_chunk.get('id') == chunk.get('id'):
        neg_chunk = random.choice(chunks)
    negative = neg_chunk.get('answer', '')
    
    if negative:
        pair = {
            'anchor': question,
            'positive': positive,
            'negative': negative,
            'pair_id': f'{chunk.get("id")}_train',
            'pair_type': 'qa_pair',
            'difficulty': 0.5
        }
        training_pairs.append(pair)

print(f'✅ Generated {len(training_pairs)} training pairs')

# Save
output_path = Path('data/processed/training_pairs.jsonl')
print(f'⏳ Saving to {output_path}...')
with open(output_path, 'w', encoding='utf-8') as f:
    for pair in training_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

print(f'✅ SAVED!')
print(f'\n=== TRAINING PAIRS SUMMARY ===')
print(f'Total pairs: {len(training_pairs)}')
print(f'Anchor: Turkish legal question')
print(f'Positive: Relevant answer')
print(f'Negative: Random answer (hard negative)')
print(f'Format: One triplet per line (JSONL)')
print(f'File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB')
