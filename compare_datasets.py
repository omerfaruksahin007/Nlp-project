import json
import os

print('📂 Dataset Comparison:')
print('='*80)

for root, dirs, files in os.walk('data/processed'):
    for file in sorted(files):
        if file.endswith('.jsonl'):
            path = os.path.join(root, file)
            size = os.path.getsize(path) / 1024 / 1024
            
            # Count docs and check structure
            with open(path, 'r', encoding='utf-8') as f:
                docs = [json.loads(line) for line in f if line.strip()]
            
            if docs:
                first = docs[0]
                has_article = sum(1 for d in docs if d.get('article_no'))
                
                print(f'\n{file} ({size:.1f}MB, {len(docs)} docs)')
                print(f'  law_name: {first.get("law_name", "NONE")[:30]}')
                print(f'  article_no: {first.get("article_no", "NONE")} (present in {has_article} docs)')
                print(f'  Fields: {list(first.keys())}')
