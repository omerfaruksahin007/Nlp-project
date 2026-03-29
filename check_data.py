import json
from pathlib import Path

# Check if file exists
doc_path = Path('data/processed/turkish_law.jsonl')
print(f'📁 File exists: {doc_path.exists()}')

if doc_path.exists():
    size_mb = doc_path.stat().st_size / (1024*1024)
    print(f'📊 File size: {size_mb:.2f} MB')
    
    # Check first few lines
    with open(doc_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:
                try:
                    doc = json.loads(line)
                    print(f'\n📄 Document {i+1}:')
                    print(f'   Keys: {list(doc.keys())[:5]}')
                    if 'answer' in doc:
                        preview = doc['answer'][:100]
                        print(f'   Answer: {preview}...')
                    elif 'text' in doc:
                        preview = doc['text'][:100]
                        print(f'   Text: {preview}...')
                    else:
                        preview = str(doc)[:100]
                        print(f'   Content: {preview}...')
                except Exception as e:
                    print(f'   Error parsing: {e}')
            else:
                break
        
        # Count total lines
        f.seek(0)
        total = sum(1 for _ in f)
        print(f'\n📊 Total documents: {total}')
else:
    print('❌ File not found at:', doc_path.absolute())
    print('\n📂 Available files in data/processed/:')
    
    if Path('data/processed').exists():
        files = list(Path('data/processed').glob('*.jsonl'))
        if files:
            for f in files:
                size = f.stat().st_size / (1024*1024)
                print(f'   {f.name}: {size:.2f} MB')
        else:
            print('   No JSONL files found')
            print('\n📂 All files in data/processed/:')
            for f in Path('data/processed').glob('*'):
                print(f'   {f.name}')
    else:
        print('   Directory not found')
