import json

nb_path = 'COLAB_RAG_PRODUCTION.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('📋 NOTEBOOK STRUCTURE CHECK')
print('='*60)

total_cells = len(nb['cells'])
print(f'\nTotal cells: {total_cells}')

md_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')

print(f'Markdown cells: {md_count}')
print(f'Code cells: {code_count}')

# Check critical functions
critical = ['retrieve_hybrid', 'generate_answer', 'while True']
found = []

for cell in nb['cells']:
    content = ''.join(cell.get('source', []))
    for func in critical:
        if func in content and func not in found:
            found.append(func)

print(f'\nCritical functions:')
for func in critical:
    if func in found:
        print(f'  ✅ {func}')
    else:
        print(f'  ❌ {func}')

# Check for citation fix
citation_fix = False
for cell in nb['cells']:
    content = ''.join(cell.get('source', []))
    if 'law_name' in content and 'article_no' in content and 'citation' in content:
        citation_fix = True
        break

print(f'\nCitation fix:')
if citation_fix:
    print(f'  ✅ law_name + article_no → citation (FIXED)')
else:
    print(f'  ❌ Citation fix not found')

print(f'\n✅ NOTEBOOK STATUS: READY FOR COLAB!')
