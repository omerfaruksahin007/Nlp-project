"""
Training Data Preparation for Turkish Legal RAG

Converts raw Turkish legal QA data into instruction-tuning format.

Format:
{
    "instruction": "Turkish legal question",
    "context": "Relevant legal texts",
    "output": "Grounded answer with citations"
}
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single instruction-tuning example"""
    instruction: str  # Turkish legal question
    context: str      # Relevant legal texts (chunks)
    output: str       # Expected answer with citations
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class TrainingDataGenerator:
    """
    Generate synthetic/curated instruction-tuning examples
    from Turkish legal data
    """
    
    # Example Turkish legal Q&A pairs
    LEGAL_QA_PAIRS = [
        {
            "question": "Hırsızlık suçu nedir?",
            "context": """
Türk Ceza Kanunu Madde 141: Başkasına ait bir malı, sahibini haksız yere yoksun etmek amacıyla 
alan kişi hırsızlık suçunu işlemiş olur. Hırsızlık cezası Madde 142'de belirtilmiştir.

Madde 142: Hırsızlık suçunun cezası altı aydan üç yıla kadar hapis cezasıdır. Eğer suç hafif 
sayılabilen hallerde gerçekleşmişse, ceza üç aydan bir yıla kadar hapis cezası olur.

Yargıtay İçtihadı: Hırsızlık suçunda asıl unsur, hukuki mülkiyet hakkını ihlal etmek 
amacının varlığıdır. Bu amaç çalıştığı sürece fiilin hırsızlık olarak değerlendirilmesi mümkündür.
            """,
            "answer": """
YANIT:
Türk Ceza Kanunu Madde 141'e göre, hırsızlık suçu başkasına ait bir malı, sahibini haksız yere 
yoksun etmek amacıyla almak filiyatıdır. Bu suçun işlenmesi için failin söz konusu mal üzerinde 
hukuki mülkiyet hakkını ihlal etme niyeti bulunmalıdır.

MANTIK:
Tanım doğrudan Türk Ceza Kanunu'nun 141. maddesinden alınmıştır. Suçun oluşabilmesi için:
1. Failin bilinci ve iradesi olmalı
2. Malın başkasına ait olması gerekli
3. Mülkiyet hakkını ihlal etme amacı bulunmalı

KAYNAKLAR:
1. Türk Ceza Kanunu, Madde 141
2. Türk Ceza Kanunu, Madde 142
3. Yargıtay İçtihadı

GÜVENİLİRLİK: high
            """
        },
        {
            "question": "Müebbet hapis cezası nedir?",
            "context": """
Türk Ceza Kanunu Madde 48: Müebbet hapis cezası seumur hayat boyunca verilen 
bir hapis cezasıdır. Müebbet hapis cezasına mahkûm bir kişi, çok nadir hallerde 
hakim kararıyla koşullu salıverme yoluna tabi tutulabilir.

Madde 49: Müebbet hapis cezasında koşullu salıverme, en az yirmi dört yıl 
yatış koşuluyla verilebilir.

Ceza İnfaz Kanunu: Müebbet hapis cezasının infazı, ıslah ve sosyal uyum 
ilkeleri göz önüne alınarak gerçekleştirilir.
            """,
            "answer": """
YANIT:
Türk Ceza Kanunu Madde 48'e göre, müebbet hapis cezası seumur hayat boyunca verilen 
bir hapis cezasıdır. Bu cezaya mahkûm olan kişiler, belirtilen koşullar altında 
koşullu salıvermeye tabi olabilirler.

MANTIK:
Müebbet hapis, cezanın en ağır biçimidir ve ancak kanunun öngördüğü en ağır suçlar 
için verilebilir. Koşullu salıverme olanağı, çok ağır olsa da, uyum ve ıslah açısından 
avantaj sağlayabilir.

KAYNAKLAR:
1. Türk Ceza Kanunu, Madde 48
2. Türk Ceza Kanunu, Madde 49
3. Ceza İnfaz Kanunu

GÜVENİLİRLİK: high
            """
        },
        {
            "question": "Medeni Kanun'da evlilik nedir?",
            "context": """
Türk Medeni Kanunu Madde 175: Evlilik, bir erkek ile bir kadın arasında kurulan 
hukuki ilişkidir. Evlilik, bu kanunda öngörülen şekil ve şartlar altında işlem 
görmesi gerekir.

Madde 176: Evlilik, ancak rızaları ile evlenmeye karar vermiş olan bir erkek ile 
bir kadın arasında kurulabilir. Evlenme rızası, serbest ve tam iradeli olmalı, 
herhangi bir zorlama altında olmamalıdır.

Madde 177: Evlilik, nüfus müdürü tarafından kaydedilir ve tescil edilir. 
Bu tescil olmaksızın evlilik geçerli değildir.
            """,
            "answer": """
YANIT:
Türk Medeni Kanunu Madde 175'e göre, evlilik bir erkek ile bir kadın arasında kurulan 
hukuki ilişkidir. Bu ilişki, kanunun öngördüğü şekil ve şartlar altında kurulmalı ve 
resmî olarak tescil edilmelidir.

MANTIK:
Evliliğin geçerli olması için:
1. Nişanlı çiftlerin serbest ve tam rızası gerekir (Madde 176)
2. Kanunda belirtilen yaş ve diğer şartlar sağlanmalı
3. Nüfus müdürü tarafından resmî olarak tescil edilmeli (Madde 177)

KAYNAKLAR:
1. Türk Medeni Kanunu, Madde 175
2. Türk Medeni Kanunu, Madde 176
3. Türk Medeni Kanunu, Madde 177

GÜVENİLİRLİK: high
            """
        },
        {
            "question": "İş Kanunu'nda çalışma saatleri ne kadar?",
            "context": """
Türk İş Kanunu Madde 63: İşçinin haftalık çalışma süresi kırk saat olarak belirlenmiş, 
haftanın beş veya altı günü içerisinde dağıtılabilir. Günlük çalışma süresi sekiz saat 
veya dokuz saate kadar uzatılabilir.

Madde 64: Fazla çalışma yapılabilmesi için işveren ve işçilerin, toplu iş sözleşmesinin 
bulunması halinde sendika onayı, yoksa işçilerin yazılı onayı gerekir.

Madde 65: Fazla çalışmanın günlük en çok üç saat, haftalık otuz saat olması kaydıyla 
yapılabilir.
            """,
            "answer": """
YANIT:
Türk İş Kanunu Madde 63'e göre, işçinin haftalık çalışma süresi kırk saat olarak 
belirlenmiştir. Bu süre haftanın beş veya altı günü içerisinde dağıtılabilir; 
günlük çalışma süresi ise sekiz saat veya dokuz saate kadar uzatılabilir.

MANTIK:
Türk iş hukuku'nda çalışma saatleri titizlikle düzenlenmiştir:
- Haftalık: 40 saat (temel hak)
- Günlük: 8-9 saat (iş yerinin durumuna göre)
- Fazla çalışma: İşçi onayı ve kanunî sınırlar içinde

KAYNAKLAR:
1. Türk İş Kanunu, Madde 63
2. Türk İş Kanunu, Madde 64
3. Türk İş Kanunu, Madde 65

GÜVENİLİRLİK: high
            """
        }
    ]
    
    def __init__(self):
        """Initialize generator"""
        self.examples: List[TrainingExample] = []
    
    def generate_examples(self) -> List[TrainingExample]:
        """
        Generate training examples from curated QA pairs
        
        Returns:
        --------
        List[TrainingExample]: Training examples in instruction format
        """
        self.examples = []
        
        for qa in self.LEGAL_QA_PAIRS:
            example = TrainingExample(
                instruction=qa["question"],
                context=qa["context"].strip(),
                output=qa["answer"].strip()
            )
            self.examples.append(example)
        
        logger.info(f"Generated {len(self.examples)} training examples")
        return self.examples
    
    def augment_examples(self, factor: int = 3) -> List[TrainingExample]:
        """
        Augment examples by creating variations
        (for small datasets, keep it simple)
        
        Parameters:
        -----------
        factor: int
            Multiplication factor (total = original * factor)
        
        Returns:
        --------
        List[TrainingExample]: Augmented examples
        """
        if not self.examples:
            self.generate_examples()
        
        augmented = self.examples.copy()
        
        # Simple variation: rephrase same question slightly
        variations = [
            ("Lütfen açıkla: {}", "Please explain: {}"),
            ("Tanımla: {}", "Define: {}"),
            ("Hukuki açıdan ne demektir: {}", "What does it mean legally: {}"),
        ]
        
        for i, example in enumerate(self.examples):
            for j in range(factor - 1):
                # Vary the instruction slightly
                variation_prefix, _ = variations[j % len(variations)]
                new_instruction = variation_prefix.format(example.instruction)
                
                augmented.append(TrainingExample(
                    instruction=new_instruction,
                    context=example.context,
                    output=example.output
                ))
        
        logger.info(f"Augmented to {len(augmented)} examples (factor={factor})")
        return augmented
    
    def save_examples(
        self,
        examples: List[TrainingExample],
        output_path: Path,
        split_ratio: tuple = (0.7, 0.15, 0.15)
    ) -> Dict[str, Path]:
        """
        Save examples to train/val/test splits
        
        Parameters:
        -----------
        examples: List[TrainingExample]
            Training examples to save
        output_path: Path
            Directory to save splits
        split_ratio: tuple
            (train, val, test) ratios
        
        Returns:
        --------
        Dict[str, Path]: Paths to train/val/test files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Split examples
        n = len(examples)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        
        train_examples = examples[:n_train]
        val_examples = examples[n_train:n_train + n_val]
        test_examples = examples[n_train + n_val:]
        
        # Save as JSONL
        splits = {
            'train': train_examples,
            'val': val_examples,
            'test': test_examples
        }
        
        saved_paths = {}
        for split_name, split_examples in splits.items():
            path = output_path / f"{split_name}.jsonl"
            
            with open(path, 'w', encoding='utf-8') as f:
                for example in split_examples:
                    f.write(example.to_json() + '\n')
            
            saved_paths[split_name] = path
            logger.info(f"Saved {len(split_examples)} {split_name} examples to {path}")
        
        return saved_paths
    
    @staticmethod
    def load_examples(path: Path) -> List[TrainingExample]:
        """
        Load examples from JSONL file
        
        Parameters:
        -----------
        path: Path
            Path to JSONL file
        
        Returns:
        --------
        List[TrainingExample]: Loaded examples
        """
        examples = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    example = TrainingExample(**data)
                    examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from {path}")
        return examples


class PromptFormatter:
    """
    Format instruction-tuning examples into LLM prompts
    Supports different model formats (Alpaca, ChatML, etc.)
    """
    
    @staticmethod
    def format_alpaca(example: TrainingExample) -> str:
        """
        Alpaca format:
        ### Instruction:\n...\n### Input:\n...\n### Response:\n...
        """
        return f"""### Instruction:
{example.instruction}

### Input:
{example.context}

### Response:
{example.output}"""
    
    @staticmethod
    def format_chatml(example: TrainingExample) -> str:
        """
        ChatML format:
        <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
        """
        system = """YANIT, MANTIK, KAYNAKLAR, GÜVENİLİRLİK bölümleriyle Turkish legal answer ver."""
        
        return f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Context: {example.context}

Question: {example.instruction}<|im_end|>
<|im_start|>assistant
{example.output}<|im_end|>"""
    
    @staticmethod
    def format_mistral(example: TrainingExample) -> str:
        """
        Mistral format:
        [INST] user message [/INST] assistant message
        """
        return f"""[INST] Şu bağlamda cevap ver:

Context: {example.context}

Question: {example.instruction} [/INST]

{example.output}"""


if __name__ == "__main__":
    # Test
    generator = TrainingDataGenerator()
    examples = generator.generate_examples()
    print(f"Generated {len(examples)} examples")
    
    # Show first example
    if examples:
        print("\n=== First Example ===")
        print(f"Instruction: {examples[0].instruction}")
        print(f"Context: {examples[0].context[:100]}...")
        print(f"Output: {examples[0].output[:100]}...")
