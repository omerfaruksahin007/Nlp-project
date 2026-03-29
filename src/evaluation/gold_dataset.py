"""
PROMPT 11: Gold Standard Test Dataset
Turkish Legal Q&A pairs with ground truth answers and citations
"""

GOLD_TEST_DATASET = [
    {
        "id": "test_001",
        "question": "Madde 141 hırsızlık suçu nedir ve cezası ne kadar?",
        "gold_answer": (
            "Türk Ceza Kanunu Madde 141'e göre, hırsızlık suçu başkasının mal varlığından "
            "olan bir taşınırı rızası olmaksızın kendisine veya başkasına ait kılmak isteği "
            "ile çalan kişi tarafından işlenebilecek bir suçtur. "
            "Hırsızlık suçunun cezası iki yıldan beş yıla kadar hapistir. "
            "Eğer suç bir evden veya işletmeden işlenirse veya gece ve birden çok kişi tarafından "
            "işlenirse ceza artırılabilir."
        ),
        "gold_citations": ["Madde 141", "TCK"],
        "relevant_doc_ids": ["doc_tck_141", "doc_hırsızlık_suçu"],
        "keywords": ["hırsızlık", "ceza", "mal", "taşınır"]
    },
    {
        "id": "test_002",
        "question": "Türk ceza hukukunda dayanılmaz özür nedir?",
        "gold_answer": (
            "Türk Ceza Kanunu Madde 24'te düzenlenen dayanılmaz özür, "
            "fail tarafından bilhassa işlenen suçun sonucunu görmemesinden veya "
            "fakat değişen koşullar yüzünden suçun işlenmesi anında özür unsurlarının "
            "meydana gelmemesinden bahsetmektedir. "
            "Dayanılmaz özür halinde fail için özel bir hüküm bulunmadığı takdirde "
            "ceza indiriminden istifade edilmez."
        ),
        "gold_citations": ["Madde 24", "TCK"],
        "relevant_doc_ids": ["doc_tck_24", "doc_özür"],
        "keywords": ["özür", "ceza", "indiri", "fail"]
    },
    {
        "id": "test_003",
        "question": "Kasten adam öldürme suçu kaç yıl hapisle cezalandırılır?",
        "gold_answer": (
            "Türk Ceza Kanunu Madde 81'e göre, kasten adam öldürme suçunun "
            "temel cezası yirmi yıldan otuz yıla kadar hapistir. "
            "Ancak ağırlaştırıcı veya hafifletici koşullar bulunabilir. "
            "Örneğin, cinsel saldırı sonucu ölüm varsa ceza daha ağır olur. "
            "Alkol etkisi altında işlenmişse ise hafifletici sebep sayılabilir."
        ),
        "gold_citations": ["Madde 81", "TCK"],
        "relevant_doc_ids": ["doc_tck_81", "doc_adam_öldürme"],
        "keywords": ["adam öldürme", "ceza", "hapsi", "kasten"]
    },
    {
        "id": "test_004",
        "question": "Dolandırıcılık suçunun unsurları nelerdir?",
        "gold_answer": (
            "Türk Ceza Kanunu Madde 157'de düzenlenen dolandırıcılık suçunun unsurları şunlardır: "
            "(1) Hile ve hilenin oluşturduğu aldatmaca, "
            "(2) Aldatmaca nedeniyle mağdurun kandırılması, "
            "(3) Kendisine veya başkasına ait kılmak isteğiyle çalma maddesi olmaksızın "
            "mal varlığından doğan bir menfaat elde etme, "
            "(4) Mağdurun bu menfaati vermede böylelikle zararlanması. "
            "Dolandırıcılık suçunda ceza iki yıldan altı yıla kadar hapistir."
        ),
        "gold_citations": ["Madde 157", "TCK"],
        "relevant_doc_ids": ["doc_tck_157", "doc_dolandırıcılık"],
        "keywords": ["dolandırıcılık", "unsur", "hile", "kandırma"]
    },
    {
        "id": "test_005",
        "question": "Tecavüz suçu TCK'da nasıl tanımlanmaktadır?",
        "gold_answer": (
            "Türk Ceza Kanunu Madde 102'de cinsel saldırı suçu düzenlenmiştir. "
            "Rızası olmayan bir kişiye karşı gerçekleştirilen cinsel davranış tecavüz oluşturur. "
            "Tecavüz suçunun cezası sekiz yıldan onbeş yıla kadar hapistir. "
            "Mağdurun çocuk olması, şiddet veya tehdit kullanılması durumlarında ceza artar. "
            "Bu suç hayatı boyunca kayıtlı kalması sorununu yaratabilir."
        ),
        "gold_citations": ["Madde 102", "TCK", "cinsel saldırı"],
        "relevant_doc_ids": ["doc_tck_102", "doc_tecavüz"],
        "keywords": ["tecavüz", "cinsel saldırı", "ceza", "rıza"]
    },
    {
        "id": "test_006",
        "question": "Evlilikte miras payları nasıl belirlenir?",
        "gold_answer": (
            "Türk Medeni Kanunu Madde 495-541 arasında miras hukuku düzenlenmiştir. "
            "Eşin miras payı duruma göre değişikmektedir. Eğer çocuk varsa eşin payı 1/4, "
            "eğer çocuk yoksa ve ana-baba varsa eşin payı 1/2, eğer çocuk ve ana-baba de yoksa "
            "eş tüm mirası alır. Miras paylaşımında erkek-kadın eşitliği sağlanmıştır. "
            "Yaşayan eş ölen eşin kişisel eşyalarından öncelik sahibidir."
        ),
        "gold_citations": ["Madde 495-541", "TMK", "miras"],
        "relevant_doc_ids": ["doc_tmk_miras", "doc_evlilik_miras"],
        "keywords": ["miras", "eş", "pay", "çocuk"]
    },
    {
        "id": "test_007",
        "question": "İş sözleşmesinin sona erdirilme nedenleri nelerdir?",
        "gold_answer": (
            "Türk İş Kanunu Madde 24-26 maddelerde iş sözleşmesinin sona erme nedenleri "
            "düzenlenmiştir. Sözleşme feshedilebilir nedenler: (1) Çalışanın kullanılamayacak "
            "derecede hasta veya sakatlanması, (2) Çalışanın uygunsuz davranışı, (3) Ekonomik "
            "nedenler gibi işyerinin kapatılması, (4) Teknolojik değişiklikler, "
            "(5) Disiplin cezası olarak fesih. Fesih öncesinde kınama, uyarı, geçici olarak "
            "işten uzaklaştırma gibi cezalar uygulanmalıdır."
        ),
        "gold_citations": ["Madde 24-26", "İş Kanunu"],
        "relevant_doc_ids": ["doc_işk_fesih", "doc_sözleşme"],
        "keywords": ["iş sözleşmesi", "fesih", "işten çıkarma"]
    },
    {
        "id": "test_008",
        "question": "Vergi Kanununda vergi mükellefi kimdir?",
        "gold_answer": (
            "Türk Vergi Sistemi'ne göre vergi mükellefi, gelir vergisine tabi olan "
            "gerçek ve tüzel kişiler ile kuruluşlardır. Gelir vergisi mükellefleri: "
            "ticaret, sanayi, esnaf faaliyetleri ve meslek sahipleri, gayrimenkul kiraları "
            "ve tarımsal faaliyetlerden gelir elde eden kişilerdir. Kurumlar vergisinin "
            "mükellefleri ise ortaklıklar, dernek, vakıf, sendika ve benzer teşekküller "
            "ile işletmeleridir. Vergi mükellefi olabilmek için kanuni veya fiili mukim olması "
            "veya Türkiye'de ekonomik menfaat ilişkisinin bulunması gerekir."
        ),
        "gold_citations": ["Vergi Kanunu", "gelir vergisi", "kurumlar vergisi"],
        "relevant_doc_ids": ["doc_vergi_mükellefi", "doc_gelir_vergisi"],
        "keywords": ["vergi", "mükellefi", "gelir", "kurumlar"]
    },
    {
        "id": "test_009",
        "question": "Ticari işletmenin satışında alıcının sorumluluğu nedir?",
        "gold_answer": (
            "Türk Medeni Kanunu Madde 479 ve Türk Ticaret Kanunu hükümlerine göre "
            "ticari işletme satışında alıcının sorumluluğu bulunmaktadır. "
            "Alıcı satıcının ödenmemiş borçlarından belli koşullarda sorumlu olabilir. "
            "Özellikle işletmeyle ilgili vergi borçları, sosyal sigorta primleri ve işçi alacakları "
            "bakımından alıcı sorumlu tutulabilir. Bu sorumluluk satın alma bedelinden kesinti yoluyla "
            "veya müşteri alacaklarının tahsili yoluyla karşılanabilir. Alıcı işletme devralındıktan "
            "sonra belirli bir süre içinde bildirimde bulunmazsa sorumluluğu türeyebilir."
        ),
        "gold_citations": ["Madde 479", "TMK", "TTK", "işletme satışı"],
        "relevant_doc_ids": ["doc_işletme_satışı", "doc_alıcı_sorumluluğu"],
        "keywords": ["işletme", "satış", "alıcı", "sorumluluk"]
    },
    {
        "id": "test_010",
        "question": "Aile hukuku kapsamında boşanma nedenleri nelerdir?",
        "gold_answer": (
            "Türk Medeni Kanunu Madde 161-162'de boşanma nedenleri düzenlenmiştir. "
            "Maddi boşanma nedenleri: (1) Evliliğin ortak yaşamını sürdürmenin mümkün olmayacak ölçüde "
            "güçleşmesi, (2) Taraflardan birinin önemli kusuru varsa diğeri boşanma isteyebilir. "
            "Kusur nedenleri: zina, şiddet, kötü muamele, aşırı içme, uyuşturucu bağımlılığı gibi. "
            "Ayrıca, taraflardan biri uzun süre (ruh hastalığı gibi) tedaviye ihtiyaç duyabilir. "
            "Boşanmada eşler ve velayetten bahsedilen çocukların durumları titizlikle ele alınmalıdır."
        ),
        "gold_citations": ["Madde 161-162", "TMK", "boşanma"],
        "relevant_doc_ids": ["doc_tmk_boşanma", "doc_boşanma_nedenleri"],
        "keywords": ["boşanma", "neden", "kusur", "evlilik"]
    }
]

def save_gold_dataset(filepath: str = "data/gold_qa_pairs.json"):
    """Save gold dataset to JSON file"""
    import json
    from pathlib import Path
    
    output_dir = Path(filepath).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(GOLD_TEST_DATASET, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Gold dataset saved to {filepath}")
    print(f"[OK] Total test cases: {len(GOLD_TEST_DATASET)}")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Save to project data directory
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "gold_qa_pairs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_gold_dataset(str(output_path))
