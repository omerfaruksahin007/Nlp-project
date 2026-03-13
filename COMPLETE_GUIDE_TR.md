# Turkish Legal RAG Projesi - Tam Dokümantasyon

## 📋 İçindekiler
1. Proje Tanımı
2. Proje Yapısı Kurulumu
3. Virtual Environment Kurulumu
4. Bağımlılıklar (Requirements)
5. Veri İçe Aktarım (Ingestion) Pipeline
6. Testler ve Sonuçlar
7. Son Durum ve İleri Adımlar

---

## 1. PROJE TANIMI

### Amaç
**Turkish Legal RAG** (Retrieval-Augmented Generation), Türk hukuk alanında soruların cevaplanmasını sağlayan yapay zeka sistemi.

### Ana Özellikler
- 📥 Türkçe hukuk belgelerini yükleme ve normalizasyon
- 🔍 Anlamsal arama (semantic search) ile belgeler arama
- 🎯 Sıralama (reranking) ile sonuçları iyileştirme
- 🔧 Dil modellerini (LLM) hukuk alanına uyarlama
- 📊 Sistem performansını değerlendirme
- 🌐 Web arayüzü (Gradio demo)

### Teknoloji Stack
```
Python 3.13.9
├── pandas, numpy           (veri işleme)
├── sentence-transformers   (gömülü vektörler/embeddings)
├── faiss-cpu              (vektör indeksleme)
├── rank-bm25              (BM25 sıralaması)
├── transformers           (LLM modeller)
├── peft                   (parametre-verimli fine-tuning)
├── datasets               (HuggingFace dataset API)
├── scikit-learn           (makine öğrenmesi)
├── gradio                 (demo web arayüzü)
└── tqdm                   (ilerleme göstergeleri)
```

---

## 2. PROJE YAPISI KURULUMU

### Adım 1: Örnek Proje Yapısı
Projemiz şu klasör yapısını kullana:

```
turkish-legal-rag/
├── data/                          # Veri depolama
│   ├── raw/                      # Ham veriler (indirilenler)
│   ├── processed/                # Temizlenmiş veriler
│   ├── chunked/                  # Metin parçaları (retrieval için)
│   └── gold/                     # Değerlendirme setleri
│
├── src/                           # Ana kaynak kodu
│   ├── ingestion/                # Veri yükleme ve normalizasyon
│   ├── retrieval/                # Anlamsal arama
│   ├── reranking/                # Sonuç sıralaması
│   ├── llm/                      # Model ince ayarı
│   ├── evaluation/               # Performans değerlendirmesi
│   └── utils/                    # Yardımcı fonksiyonlar
│
├── scripts/                       # Çalıştırılabilir komut dosyaları
├── configs/                       # Yapılandırma dosyaları
├── notebooks/                     # Jupyter defterleri
├── experiments/                   # Deney sonuçları
├── reports/                       # Raporlar
├── app/                          # Web demo arayüzü
│
├── requirements.txt              # Python bağımlılıkları
├── README.md                     # Proje açıklaması
└── venv/                         # Virtual environment

```

### Adım 2: Klasörleri Oluşturma
Tüm gerekli klasörler otomatik olarak oluşturuldu. Terminal'de doğrulama:

```bash
# Tüm klasörlerin var olup olmadığını kontrol et
ls -la
ls -la data/
ls -la src/
ls -la scripts/
```

✅ **Sonuç**: 15 ana klasör başarıyla oluşturuldu

---

## 3. VIRTUAL ENVIRONMENT KURULUMU

### Adım 1: Python Sürümü Kontrolü
```bash
python --version
# Output: Python 3.13.9 ✅
```

### Adım 2: Virtual Environment Oluşturma
```bash
python -m venv venv
```

Nedir?
- `venv` = Python'un kendi virtual environment aracı
- Bu, projeye özel bir Python ortamı oluşturur
- Her proje bağımsız olarak çalışır, paketler izole olur

### Adım 3: Virtual Environment Aktivasyon
```bash
# PowerShell'de (Windows)
venv\Scripts\Activate.ps1

# İlk çalıştırmada izin vermesi gerekebilir:
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
venv\Scripts\Activate.ps1

# Başarılı olursa, terminal'de şu göözür:
# (venv) PS C:\...
```

Nedir?
- Virtual environment aktif oldu
- Şimdi kurduğumuz paketler sadece bu projede kullanılır
- Sistem Python'u etkilenmez

✅ **Sonuç**: Virtual environment aktif ve hazır

---

## 4. BAĞLILIKLAR (REQUIREMENTS.TXT)

### Adım 1: Requirements.txt İçeriği
```
pandas                     # CSV/Excel veri manipülasyonu
numpy                      # Sayısal hesaplamalar
sentence-transformers      # Türkçe metni vektöre çevirme (embedding)
faiss-cpu                  # Vektör araması (indeksleme)
rank-bm25                  # BM25 sıralama algoritması
transformers               # Transformer modellerini başlatma
datasets                   # HuggingFace datasets API
peft                       # LoRA gibi efficient fine-tuning
accelerate                 # Dağıtık eğitim
scikit-learn               # Metrik hesaplama
tqdm                       # İlerleme barı
gradio                     # Web demo arayüzü
```

### Adım 2: Paketleri Yükleme
```bash
# Virtual environment aktif olmalı!
pip install -r requirements.txt

# Bu işlem ilk kez ~5-10 dakika alabilir
# Tüm paketleri indirir ve kurar
```

### Adım 3: Kurulumu Doğrulama
```bash
python -c "import pandas; import torch; print('✅ Tüm paketler kuruldu')"
```

✅ **Sonuç**: Tüm 12 paket başarıyla kuruldu

---

## 5. VERİ İÇE AKTARIM (INGESTION) PIPELINE

Ingestion = Ham veriler → Temiz, standart format veri

### A. İnceleme Modülleri (src/ingestion/)

#### 1. **normalizer.py** - Metin Normalizasyonu
```python
# Ne yapar?
- Türkçe karakterleri korur (ç, ğ, ı, ö, ş, ü)
- Boşlukları düzenler (fazla boşluğu temizler)
- Kontrol karakterlerini kaldırır
- Alıntı işaretlerini standardize eder

# Örnek:
Input:  "  Türkiye'de   hukuk  sistemi  nasıl  çalışır?  "
Output: "Türkiye'de hukuk sistemi nasıl çalışır?"
```

#### 2. **loader.py** - Veri Yükleme
```python
# Ne yapar?
- CSV dosyaları yükler (pandas)
- JSON dosyaları yükler
- JSONL dosyaları yükler (satır satır JSON)
- HuggingFace datasets API entegrasyonu
- Kaggle datasets yerel klasörden yükleme

# Örnek:
raw_data = load_data('data/raw/my_dataset.json')
# Output: Liste dictionaryler [(dict1), (dict2), ...]
```

#### 3. **schema.py** - Veri Şemaları
```python
# Standart veri formatları tanımlar

# QA Pair (Soru-Cevap) Şeması
{
  "id": "unique_id",
  "question": "Soru metni",
  "answer": "Cevap metni",
  "source": "Veri kaynağı adı",
  "category": "Hukuk kategorisi (isteğe bağlı)",
  "citation": "Kaynak (isteğe bağlı)"
}

# Legal Document (Hukuk Belgesi) Şeması
{
  "doc_id": "document_id",
  "title": "Başlık",
  "law_name": "Kanun adı",
  "article_no": "Madde numarası",
  "section": "Bölüm",
  "text": "Tam metin",
  "source": "Kaynak"
}
```

#### 4. **converters.py** - Format Dönüştürücüler
```python
# Farklı kaynaklardan gelen verileri standart şemaya çevirir

class QAConverter:
  # Esnek alan eşleştirmesi (field mapping)
  # Örneğin: soru -> question, cevap -> answer
  
class TurkishLawchatbotConverter:
  # HuggingFace türü veriler için özel dönüştürücü
  
class TurkishLawKaggleConverter:
  # Kaggle türü veriler için özel dönüştürücü
  # Otomatik alan deteksiyonu
```

#### 5. **pipeline.py** - Orkestraçyon
```python
# Tüm adımları koordine eder
# 1. Verileri yükle
# 2. Normaliz et
# 3. Şemaya dönüştür
# 4. Çoğaltmaaları kaldır
# 5. JSONL olarak kaydet
```

### B. Veri Kaynakları (data/raw/)

#### 1. **example_lawchatbot.json** (1.6 KB)
HuggingFace dataset formatında örnek:
```json
[
  {
    "question": "Türkçe soru",
    "answer": "Türkçe cevap",
    "category": "Hukuk kategorisi"
  }
]
```

#### 2. **example_kaggle_turkishlaw.json** (2.1 KB)
Kaggle Instruction format'ında örnek:
```json
[
  {
    "instruction": "Talimat",
    "input": "Giriş",
    "output": "Çıkış"
  }
]
```

#### 3. **sample_legal_texts.json** (2.1 KB)
Hukuk belgesi formatında örnek:
```json
[
  {
    "doc_id": "law_001",
    "title": "Kanun başlığı",
    "law_name": "Kanun adı",
    "article_no": "141",
    "section": "Bölüm",
    "text": "Tam metin",
    "source": "Kaynak"
  }
]
```

#### 4. **sample_qa_dataset.json** (1.7 KB)
Nested yapıdaki QA formato:
```json
{
  "data": [
    {
      "id": "qa_001",
      "question": "...",
      "answer": "...",
      "source": "...",
      "category": "..."
    }
  ]
}
```

#### 5. **turkish_law_dataset.csv** (100 MB) ⭐ Ana veri seti
Türkçe alan adlarıyla:
```
soru,cevap,veri türü,kaynak,context,Score
"Anayasa, Türk Vatanı...","Anayasa, Türk Vatanı...","hukuk","Anayasa","...",8
```

### C. İnceleme Komut Dosyaları (scripts/)

#### 1. **ingest_demo.py** - Demo (Test)
```bash
python scripts/ingest_demo.py
```

Ne yapar?
- Örnek dosyalarını yükler
- Metin normalizasyonunu gösterir
- HuggingFace formatını dönüştürür
- Kaggle formatını dönüştürür
- data/processed/ dizinine kaydet

Çıktı:
- example_lawchatbot_processed.jsonl (5 kayıt)
- example_kaggle_turkishlaw_processed.jsonl (5 kayıt)

#### 2. **ingest_data.py** - Ana Script
```bash
python scripts/ingest_data.py --input <file> --output <name> --type {qa,legal}
```

Örnekler:
```bash
# QA verisi ile
python scripts/ingest_data.py \
  --input data/raw/sample_qa_dataset.json \
  --output sample_qa \
  --type qa

# Hukuk belgesi ile
python scripts/ingest_data.py \
  --input data/raw/sample_legal_texts.json \
  --output sample_legal_texts \
  --type legal

# CSV -Türkçe alan adları ile (alan eşleştirmesi gerekli)
# İndie yazılı verify_csv_mapping.py ile
```

#### 3. **ingest_turkish_datasets.py** - Gerçek Dataset'ler
```bash
# Sadece HuggingFace
python scripts/ingest_turkish_datasets.py --hf

# Sadece Kaggle (indirildikten sonra)
python scripts/ingest_turkish_datasets.py --kaggle data/raw/kaggle_turkishlaw

# Her ikisi
python scripts/ingest_turkish_datasets.py --all data/raw/kaggle_turkishlaw
```

---

## 6. TESTLER VE SONUÇLAR

### Test 1: Demo Script (✅ BAŞARILI)
```bash
python scripts/ingest_demo.py
```

**Sonuç:**
- example_lawchatbot.json → 5 kayıt ✅
- example_kaggle_turkishlaw.json → 5 kayıt ✅
- Türkçe karakterler korundu ✅
- Çıktı: data/processed/ ✅

### Test 2: Hukuk Belgesi (✅ BAŞARILI)
```bash
python scripts/ingest_data.py \
  --input data/raw/sample_legal_texts.json \
  --output sample_legal_texts \
  --type legal
```

**Sonuç:**
- 3/3 kayıt dönüştürüldü ✅
- Çoğaltmalar: 0 ✅
- Çıktı dosyası: sample_legal_texts.jsonl ✅

### Test 3: Nested QA Formatı (✅ BAŞARILI)
```bash
python scripts/ingest_data.py \
  --input data/raw/sample_qa_dataset.json \
  --output sample_qa \
  --type qa
```

**Sonuç:**
- 4/4 kayıt dönüştürüldü ✅
- Nested "data" array otomatik çıkarıldı ✅
- İD'ler otomatik oluşturuldu ✅

### Test 4: Büyük CSV (100 MB) (✅ BAŞARILI - Alan Eşleştirmesiyle)
```bash
python verify_csv_mapping.py
```

**Sorunu:** CSV'nin alan adları Türkçe (soru, cevap, kaynak)
**Çözüm:** Alan eşleştirmesi (field mapping) ile:
```python
field_mapping = {
    'question': 'soru',
    'answer': 'cevap',
    'source': 'kaynak',
    'category': 'veri türü'
}
```

**Sonuç:**
- 13,954 kayıt yüklendi ✅
- 13,954 kayıt dönüştürüldü ✅
- 0 kayıt atlandı ✅
- Çıktı: turkish_law_dataset_verified.jsonl (7.86 MB) ✅

---

## 7. SON DURUM VE İLeri ADIMLAR

### Tamamlanan Görevler ✅

| Görev | Durum | Dosya/Çıktı |
|-------|-------|------------|
| Proje yapısı | ✅ | 15 klasör + dosyalar |
| Virtual env | ✅ | venv/ |
| Bağımlılıklar | ✅ | requirements.txt |
| Normalizasyon | ✅ | src/ingestion/normalizer.py |
| Veri yükleme | ✅ | src/ingestion/loader.py |
| Şema tanımı | ✅ | src/ingestion/schema.py |
| Dönüştürücüler | ✅ | src/ingestion/converters.py |
| Pipeline | ✅ | src/ingestion/pipeline.py |
| Demo script | ✅ | scripts/ingest_demo.py |
| Ana script | ✅ | scripts/ingest_data.py |
| HF/Kaggle script | ✅ | scripts/ingest_turkish_datasets.py |
| Testler | ✅ | 4/4 başarılı |
| Sonuç verisi | ✅ | 13,610 kayıt |

### Veriler (data/processed/)
```
✅ example_lawchatbot_processed.jsonl (5 kayıt)
✅ example_kaggle_turkishlaw_processed.jsonl (5 kayıt)
✅ sample_legal_texts.jsonl (3 kayıt)
✅ sample_qa.jsonl (4 kayıt)
✅ turkish_law_dataset_verified.jsonl (13,954 kayıt) ⭐ ana seti
───────────────────────────────────────────────
   TOPLAM: 13,971 kayıt hazır!
```

### Yapılacak İşler

#### Faz 1: Retrieval (Arama) Modülü
```
Hedef: src/retrieval/
Görevler:
- Sentence-transformers ile Türkçe embedding modeli seç
- Processed verileri vektörlere dönüştür
- FAISS indeksi oluştur (hızlı vektör araması)
- Anlamsal arama fonksiyonu implement
```

#### Faz 2: Reranking (Sıralama) Modülü
```
Hedef: src/reranking/
Görevler:
- Cross-encoder modeli seç (Türkçe)
- Arama sonuçlarını sırala
- BM25 sıramayı implement
- Hybrid arama (EL + BM25)
```

#### Faz 3: LLM Fine-tuning Modülü
```
Hedef: src/llm/
Görevler:
- Türkçe LLM modeli seç (Tüik/LLaMA/GPT vb)
- PEFT (LoRA) ile ince ayar
- QA pair'ları eğitim formatına dönüştür
- Fine-tuning pipeline
```

#### Faz 4: Değerlendirme Modülü
```
Hedef: src/evaluation/
Görevler:
- Retrieval metriklerini hesapla (MRR, NDCG, MAP)
- Generation metriklerini hesapla (BLEU, ROUGE)
- Custom Türkçe metrikler
- Performans raporları
```

#### Faz 5: Demo Uygulaması
```
Hedef: app/
Görevler:
- Gradio web arayüzü
- Soru giriş alanı
- Cevap çıkışı
- Kaynak gösterme
- Deploy hazırlığı
```

---

## 📚 Önemli Dosyalar

### Dokümantasyon
- **README.md** - Proje oklamaya başlayın
- **INGESTION.md** - Veri ingestion detaylı dokümantasyon
- **HEALTH_CHECK_REPORT.md** - Proje durum raporu
- **TEST_RESULTS.md** - Test sonuçları
- **IMPLEMENTATION_SUMMARY.md** - Teknik detaylar

### Yazılı Dosyalar
- **requirements.txt** - Python paketleri
- **verify_csv_mapping.py** - CSV alan eşleştirmesi doğrulama
- **test_csv_mapping.py** - CSV test script'i
- **.gitignore** - Git ignore kuralları

### Veri Dosyaları
- **data/raw/** - Ham veriler (5 dosya, 100 MB+)
- **data/processed/** - Temiz veriler (Ş JSONL dosyası, ~8 MB)
- **data/chunked/** - Boş (parçalama için)
- **data/gold/** - Boş (değerlendirme seti için)

---

## 🚀 SONUÇ

✅ **Proje Durumu: HAZIR**

İnternet kaynaklı bilgisi başarıyla kuruldu:
1. ✅ 15 klasörlü modüler yapı
2. ✅ Python 3.13.9 + 12 paket bağımlılığı
3. ✅ Veri ingestion pipeline (5 modül + 7 script)
4. ✅ Türkçe metin normalizasyonu
5. ✅ Esnek alan eşleştirmesi sistemi
6. ✅ 13,971 kayıt temiz veri
7. ✅ 4/4 tamamı başarılı test

**Sonraki Adım:** 
Retrieval modülüne başlamaya hazır. Sentence-transformers ile Turkish embedding modeli seçerek başlayabilir.

---

## 📞 Hızlı Komutlar Özeti

```bash
# Virtual environment aktifleştir
venv\Scripts\Activate.ps1

# Demo çalıştır
python scripts/ingest_demo.py

# Tüm testleri çalıştır
python verify_csv_mapping.py

# Yeni veri ingestion
python scripts/ingest_data.py --input <dosya> --output <ad> --type {qa,legal}

# HuggingFace dataset
python scripts/ingest_turkish_datasets.py --hf

# Özel Python kodu çalıştır
python -c "import pandas; print(pandas.__version__)"
```

---

**Proje Başlatıldı**: Mart 13, 2026  
**Son Güncelleme**: Mart 13, 2026  
**Durum**: ✅ Operasyonel
