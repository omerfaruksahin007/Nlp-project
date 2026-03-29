#!/usr/bin/env python3
"""
PROMPT 9 (Part 1): Turkish Legal QA Prompt Templates

Provides prompt templates for grounded answer generation.
Focus on:
- Context-only answering (no hallucinations)
- Citation consistency
- Explicit "insufficient context" handling
- Turkish legal terminology

Design principles:
1. Be specific: Tell the model exactly what to do
2. Give examples: Show format and citation style
3. Set constraints: What to do if context is insufficient
4. Require reasoning: Explain why answer is grounded
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class AnswerConfidence(Enum):
    """Confidence level of generated answer"""
    HIGH = "high"          # Strong evidence in context
    MEDIUM = "medium"      # Some evidence but gaps
    LOW = "low"            # Weak evidence
    INSUFFICIENT = "insufficient"  # Cannot answer from context


@dataclass
class PromptComponents:
    """Structured prompt components"""
    system_instruction: str
    context_header: str
    question_instruction: str
    output_format: str
    examples: str


class TurkishLegalPromptTemplate:
    """
    Turkish-language prompt templates for legal QA.
    
    Design:
    - System instruction: Role and constraints
    - Context: Retrieved documents with clear boundaries
    - Question: User query with instructions
    - Output format: Structured answer with citations
    - Examples: Few-shot examples of correct answers
    """
    
    def __init__(self, language: str = "turkish"):
        """Initialize with language choice"""
        self.language = language
        self.components = self._build_components()
    
    def _build_components(self) -> PromptComponents:
        """Build prompt components in Turkish"""
        
        system_instruction = """SEN: Türkçe hukuk uzmanı asistanı
GÖREV: Verilen hukuk belgeleri ve maddeleri temel alarak sorulara yanıt vermek
KURAL: SADECE verilen bağlamda bulunan bilgileri kullan

👉 ÖNEMLİ KURALLAR:
1. Yanıtın her iddiası verilen belgelerle desteklenmelidir
2. Eğer verilen bağlamda yeterli bilgi yoksa, NET VE AÇIK olarak "Verilen kaynaklar yeterli değil" diye belirt
3. Asla bağlamın dışından bilgi ekleme (halüsinasyon yapma)
4. Kaynakları açık ve tutarlı şekilde göster (Madde numarası, kanun adı, vb.)
5. Cevaptan sonra kullandığın kaynakları listele
6. Eğer soruda tartışmalı bir konu varsa, tüm tarafları adil sunmaya çalış"""
        
        context_header = """════════════════════════════════════════
VERILEN HUKUK BİLGİLERİ (BAĞLAM)
════════════════════════════════════════

Aşağıdaki belge parçaları senin SADECE bilgi kaynağındır. Diğer hiçbir dış bilgi kullanma."""
        
        question_instruction = """════════════════════════════════════════
SORU
════════════════════════════════════════

Aşağıdaki soruya yukarıdaki bağlamdan SADECE yanıt ver:"""
        
        output_format = """════════════════════════════════════════
YANIT FORMATI
════════════════════════════════════════

Lütfen aşağıdaki formatta yanıt ver:

1. YANIT (1-3 paragraf, net ve özlü):
   [Sorunun doğrudan yanıtı. Eğer yeterli bilgi yoksa: "Verilen kaynaklar yeterli değil: [nedeni]"]

2. MANTIK (Neden bu yanıtı verdiysen) :
   [Cevabının hangi parçalardan geldiğini kısaca açıkla]

3. KAYNAKLAR:
   - Madde [No]: [Kanun Adı] - [Alıntı]
   - Madde [No]: [Kanun Adı] - [Alıntı]
   
4. GÜVENİLİRLİK: [high/medium/low/insufficient]"""
        
        examples = """════════════════════════════════════════
ÖRNEK
════════════════════════════════════════

Örnek Bağlam:
> Türk Ceza Kanunu Madde 1: Kanunun adı, 5237 sayılı Türk Ceza Kanunudur.
> Madde 2: Kanunun amacı, kişi hak ve özgürlüğünü korumaktır.
> Medeni Kanun Madde 1: Türkiye Cumhuriyeti'nde yaşayan kimse medeni kişiliktir.

Örnek Soru: "Türk Ceza Kanunun amacı nedir?"

Örnek Yanıt:

1. YANIT:
   Türk Ceza Kanunun amacı, kişi hak ve özgürlüğünü korumaktır. Kanun, bireylerin temel haklarını
   ve özgürlüklerini himaye etmek için çeşitli suçlar tanımlamakta ve cezalandırmaktadır.

2. MANTIK:
   Cevap doğrudan Türk Ceza Kanunu Madde 2'den alınmıştır. Bu madde kanunun amacını açık
   ve kesin şekilde belirtmektedir.

3. KAYNAKLAR:
   - Madde 2: Türk Ceza Kanunu - "Kanunun amacı, kişi hak ve özgürlüğünü korumaktır."

4. GÜVENİLİRLİK: high"""
        
        return PromptComponents(
            system_instruction=system_instruction,
            context_header=context_header,
            question_instruction=question_instruction,
            output_format=output_format,
            examples=examples
        )
    
    def build_full_prompt(
        self,
        question: str,
        context_chunks: List[Dict],
        include_examples: bool = True
    ) -> str:
        """
        Build complete prompt for LLM.
        
        Args:
            question: User's Turkish legal question
            context_chunks: Retrieved document chunks with metadata
            include_examples: Whether to include few-shot examples
        
        Returns:
            Full prompt string ready for LLM input
        """
        prompt_parts = [
            self.components.system_instruction,
            "",
            self.components.context_header,
            ""
        ]
        
        # Add context chunks with clear boundaries
        for idx, chunk in enumerate(context_chunks, 1):
            chunk_text = chunk.get('text', chunk.get('chunk_text', ''))
            source = chunk.get('source', 'Bilinmeyen Kaynak')
            relevance = chunk.get('relevance_score', 0)
            
            prompt_parts.append(f"[Belge {idx} - {source} (İlgi: {relevance:.2f})]")
            prompt_parts.append(chunk_text)
            prompt_parts.append("")
        
        # Add question instruction and format
        prompt_parts.extend([
            self.components.question_instruction,
            "",
            f"SORU: {question}",
            "",
            self.components.output_format,
        ])
        
        # Add examples if requested
        if include_examples:
            prompt_parts.extend([
                "",
                self.components.examples,
            ])
        
        prompt_parts.append("\n" + "="*50)
        prompt_parts.append("YANIT BU ALANLA BAŞLAMALIDIR:")
        prompt_parts.append("="*50 + "\n")
        
        return "\n".join(prompt_parts)
    
    def build_minimal_prompt(
        self,
        question: str,
        context_chunks: List[Dict]
    ) -> str:
        """
        Build minimal prompt without examples (faster, fewer tokens).
        
        Useful for quick iterations or token-limited scenarios.
        """
        return self.build_full_prompt(
            question=question,
            context_chunks=context_chunks,
            include_examples=False
        )
    
    @staticmethod
    def validate_answer_format(answer_text: str) -> Dict:
        """
        Validate if answer follows required format.
        
        Returns:
            {'valid': bool, 'has_answer': bool, 'has_reasoning': bool, 'has_sources': bool}
        """
        has_answer = "YANIT" in answer_text and (":" in answer_text or "ı" in answer_text)
        has_reasoning = "MANTIK" in answer_text or "MANTIĞ" in answer_text
        has_sources = "KAYNAK" in answer_text
        
        return {
            'valid': has_answer,  # At minimum, must have answer section
            'has_answer': has_answer,
            'has_reasoning': has_reasoning,
            'has_sources': has_sources,
            'has_confidence': "GÜVENİLİRLİK" in answer_text
        }
    
    @staticmethod
    def suggest_confidence(
        answer_text: str,
        num_sources: int,
        context_coverage: float  # 0-1: how much of context was used
    ) -> AnswerConfidence:
        """
        Suggest confidence level based on answer characteristics.
        
        Args:
            answer_text: Generated answer
            num_sources: Number of cited sources
            context_coverage: 0-1 fraction of context that was cited
        
        Returns:
            Recommended confidence level
        """
        # Insufficient if model said so
        if "yeterli değil" in answer_text.lower() or "yetersiz" in answer_text.lower():
            return AnswerConfidence.INSUFFICIENT
        
        # High confidence: multiple sources, good coverage
        if num_sources >= 2 and context_coverage >= 0.7:
            return AnswerConfidence.HIGH
        
        # Medium confidence: some sources, partial coverage
        if num_sources >= 1 and context_coverage >= 0.3:
            return AnswerConfidence.MEDIUM
        
        # Low confidence: weak evidence
        return AnswerConfidence.LOW


class PromptBuilder:
    """
    Helper class for building and managing prompts.
    
    Supports:
    - Template selection (standard, minimal, expert mode)
    - Dynamic prompt generation
    - Token counting (estimate LLM cost)
    - Prompt caching/reuse
    """
    
    def __init__(self, template: Optional[TurkishLegalPromptTemplate] = None):
        """Initialize with optional custom template"""
        self.template = template or TurkishLegalPromptTemplate()
        self._prompt_cache = {}
    
    def get_prompt(
        self,
        question: str,
        context_chunks: List[Dict],
        mode: str = "full"
    ) -> str:
        """
        Get prompt with optional caching.
        
        Args:
            question: User question
            context_chunks: Retrieved chunks
            mode: 'full', 'minimal', or 'expert'
        
        Returns:
            Prompt string
        """
        if mode == "full":
            return self.template.build_full_prompt(question, context_chunks, include_examples=True)
        elif mode == "minimal":
            return self.template.build_minimal_prompt(question, context_chunks)
        elif mode == "expert":
            # Expert mode: more detailed instructions, no examples (assumes expert user knows format)
            return self.template.build_minimal_prompt(question, context_chunks)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def estimate_tokens(self, prompt: str) -> int:
        """
        Rough token count estimate (4 chars ≈ 1 token for Turkish).
        
        For accurate count, use tiktoken with actual tokenizer.
        """
        return len(prompt) // 4
    
    def estimate_cost(self, prompt: str, model: str = "gpt-3.5-turbo") -> Dict:
        """
        Estimate API cost (simplified).
        
        Args:
            prompt: Prompt string
            model: Model name (e.g., 'gpt-3.5-turbo', 'gpt-4')
        
        Returns:
            {'input_tokens': int, 'estimated_output_tokens': int, 'cost_usd': float}
        """
        input_tokens = self.estimate_tokens(prompt)
        # Rough estimate: answer will be ~200-500 tokens
        output_tokens = 350
        
        # Rough pricing (update as needed)
        pricing = {
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'gpt-4': {'input': 0.03, 'output': 0.06},
        }
        
        rates = pricing.get(model, pricing['gpt-3.5-turbo'])
        cost = (input_tokens * rates['input'] / 1000) + (output_tokens * rates['output'] / 1000)
        
        return {
            'input_tokens': input_tokens,
            'estimated_output_tokens': output_tokens,
            'cost_usd': cost,
            'model': model
        }


if __name__ == '__main__':
    # Example usage
    template = TurkishLegalPromptTemplate()
    
    question = "Türk Ceza Kanununa göre hırsızlık suçu nedir?"
    context = [
        {
            'text': 'Madde 141: Başkasına ait bir mal, sahibini haksız yere yoksun etmek amacıyla alıp götüren kişi hırsızlık suçunu işlemiş olur.',
            'source': 'Türk Ceza Kanunu',
            'relevance_score': 0.95
        }
    ]
    
    prompt = template.build_minimal_prompt(question, context)
    print(prompt)
