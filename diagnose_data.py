import json

# Check data quality
with open('data/processed/turkish_law.jsonl', 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f if line.strip()]

print('📊 Data Quality Diagnosis:')
print('='*80)
print(f'Total documents: {len(docs)}')
print()

# Sample analysis
print('Sample Documents (first 5):')
print('-'*80)
for i, doc in enumerate(docs[:5], 1):
    print(f'\nDoc {i}:')
    print(f'  law_name: {doc.get("law_name", "MISSING")}')
    print(f'  article_no: {doc.get("article_no", "MISSING")}')
    print(f'  citation: {doc.get("citation", "MISSING")}')
    print(f'  answer len: {len(doc.get("answer", ""))} chars')
    print(f'  question: {doc.get("question", "MISSING")[:60]}...')

# Field presence analysis
print('\n' + '='*80)
print('Field Presence Analysis:')
fields = ['id', 'question', 'answer', 'law_name', 'article_no', 'citation']
for field in fields:
    present = sum(1 for doc in docs if field in doc and doc[field])
    percent = (present / len(docs) * 100) if docs else 0
    print(f'  {field}: {present}/{len(docs)} ({percent:.1f}%)')

# Citation field check
print('\n' + '='*80)
print('Citation Field Status:')
has_citation = sum(1 for doc in docs if 'citation' in doc and doc['citation'])
print(f'  Documents with citation: {has_citation}/{len(docs)}')

# Sample citations
print('\n  Sample citations:')
for i, doc in enumerate(docs[:5]):
    citation = doc.get('citation', 'MISSING')
    print(f'    {i+1}. {citation}')
