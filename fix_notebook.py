import json

# Fix the notebook - add citation generation
notebook_path = r"c:\Users\barba\OneDrive\Resimler\Masaüstü\Turkish Legal RAG\COLAB_RAG_PRODUCTION.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell [6] - the one that loads documents
cell_6_idx = None
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and '## [6] Load Documents' in ''.join(cell.get('source', [])):
        cell_6_idx = idx
        break

if cell_6_idx is not None:
    # Update the loading logic
    new_code = '''import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path

print("⏳ Loading documents...")

documents = []
doc_path = Path('data/processed/turkish_law.jsonl')

if doc_path.exists():
    with open(doc_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    # Add citation field if not present
                    if 'citation' not in doc:
                        law_name = doc.get('law_name', 'Kanun')
                        article_no = doc.get('article_no', '')
                        if article_no:
                            doc['citation'] = f"{law_name} Md. {article_no}"
                        else:
                            doc['citation'] = law_name
                    documents.append(doc)
                except:
                    pass
    print(f"✅ Loaded {len(documents)} documents")
else:
    print(f"❌ Documents not found at {doc_path}")
    print(f"   Checked: {doc_path.absolute()}")
    documents = []

if documents:
    # Extract texts for indexing
    texts = [doc.get('answer', '') or doc.get('text', '') for doc in documents]
    
    # Build FAISS index
    print("\\n⏳ Building FAISS index...")
    print(f"   Encoding {len(texts)} documents...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = embeddings.astype('float32')
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    print(f"✅ FAISS index ready")
    print(f"   Vectors: {len(documents)}")
    print(f"   Dimension: {dimension}")
    
    # Build BM25 index
    print("\\n⏳ Building BM25 index...")
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    print(f"✅ BM25 index ready")
    print(f"   Documents: {len(documents)}")
else:
    print("❌ Cannot build indices without documents")
    faiss_index = None
    bm25 = None'''
    
    nb['cells'][cell_6_idx]['source'] = new_code.split('\n')
    
    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Updated Cell [6] with citation field generation")
else:
    print("❌ Could not find Cell [6]")
