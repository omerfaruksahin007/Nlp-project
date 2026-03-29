import json

# Check the verified dataset with citations
print('🔍 Citation Field Comparison:')
print('='*80)

for filename in ['turkish_law_dataset_verified.jsonl', 'turkish_law_mapped.jsonl']:
    print(f'\n{filename}:')
    print('-'*80)
    
    with open(f'data/processed/{filename}', 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f if line.strip()][:10]
    
    for i, doc in enumerate(docs[:3], 1):
        print(f'\nDoc {i}:')
        print(f'  question: {doc.get("question", "")[:60]}...')
        print(f'  citation: {doc.get("citation", "NONE")}')
        print(f'  source: {doc.get("source", "NONE")}')
        print(f'  category: {doc.get("category", "NONE")}')
