#!/usr/bin/env python3
"""
PROMPT 9 (Part 2): Answer Generation with LLM

Implements provider-agnostic answer generation.
Supports multiple LLM backends:
- OpenAI (GPT-3.5, GPT-4)
- HuggingFace (open-source models)
- Ollama (local models)
- Anthropic (Claude)

Design principle: Abstract LLM calls behind a common interface
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import os

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for answer generation"""
    provider: str           # 'openai', 'huggingface', 'ollama', 'ollama_local'
    model_name: str         # Model identifier
    temperature: float      # 0-2, higher = more creative
    max_tokens: int         # Max output length
    top_p: float           # Nucleus sampling
    timeout: int           # Seconds to wait for response
    api_key: Optional[str] # API key if needed
    base_url: Optional[str] # Base URL for self-hosted


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Tuple[str, Dict]:
        """
        Generate answer from prompt.
        
        Returns:
            (answer_text, metadata)
            where metadata includes: tokens_used, confidence, model, etc.
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI (GPT-3.5-turbo, GPT-4) provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed: pip install openai")
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Note: For OpenAI v1.0+, create a client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.use_new_api = True
        except ImportError:
            # Fallback to old API style
            openai.api_key = self.api_key
            self.use_new_api = False
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Tuple[str, Dict]:
        """Generate using OpenAI API"""
        try:
            if self.use_new_api:
                response = self.client.chat.completions.create(
                    model=config.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    timeout=config.timeout
                )
                answer = response.choices[0].message.content
                metadata = {
                    'model': response.model,
                    'tokens_used': response.usage.total_tokens,
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'finish_reason': response.choices[0].finish_reason,
                    'provider': 'openai'
                }
            else:
                # Old API
                response = openai.ChatCompletion.create(
                    model=config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    timeout=config.timeout
                )
                answer = response['choices'][0]['message']['content']
                metadata = {
                    'model': response['model'],
                    'tokens_used': response['usage']['total_tokens'],
                    'input_tokens': response['usage']['prompt_tokens'],
                    'output_tokens': response['usage']['completion_tokens'],
                    'finish_reason': response['choices'][0]['finish_reason'],
                    'provider': 'openai'
                }
            
            logger.info(f"✅ OpenAI response: {metadata['tokens_used']} tokens")
            return answer, metadata
        
        except Exception as e:
            logger.error(f"❌ OpenAI error: {e}")
            raise


class HuggingFaceProvider(LLMProvider):
    """HuggingFace (local transformers) provider"""
    
    def __init__(self, model_name: str = None):
        """Initialize with model name"""
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("transformers not installed: pip install transformers")
        
        self.model_name = model_name or 'gpt2'  # Fallback
        logger.info(f"Loading HuggingFace model: {self.model_name}")
        self.pipeline = pipeline(
            'text-generation',
            model=self.model_name,
            device='cpu'  # Or 'cuda:0' for GPU
        )
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Tuple[str, Dict]:
        """Generate using HuggingFace transformers"""
        try:
            result = self.pipeline(
                prompt,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True
            )
            
            answer = result[0]['generated_text']
            # Remove prompt from output
            if answer.startswith(prompt):
                answer = answer[len(prompt):]
            
            metadata = {
                'model': self.model_name,
                'tokens_used': config.max_tokens,  # Approximate
                'provider': 'huggingface',
                'device': 'cpu'
            }
            
            logger.info(f"✅ HuggingFace response generated")
            return answer.strip(), metadata
        
        except Exception as e:
            logger.error(f"❌ HuggingFace error: {e}")
            raise


class OllamaProvider(LLMProvider):
    """Ollama (local LLM) provider"""
    
    def __init__(self, base_url: str = 'http://localhost:11434'):
        """Initialize Ollama client"""
        if not OLLAMA_AVAILABLE:
            raise ImportError("requests not installed: pip install requests")
        
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        
        # Test connection
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                logger.info("✅ Ollama connected")
        except:
            logger.warning(f"⚠️ Ollama not responding at {base_url}")
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Tuple[str, Dict]:
        """Generate using Ollama API"""
        try:
            payload = {
                'model': config.model_name,
                'prompt': prompt,
                'stream': False,
                'temperature': config.temperature,
                'top_p': config.top_p,
                'num_predict': config.max_tokens
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get('response', '')
            
            metadata = {
                'model': config.model_name,
                'tokens_used': result.get('eval_count', 0),
                'provider': 'ollama',
                'duration_ms': result.get('total_duration', 0) / 1_000_000
            }
            
            logger.info(f"✅ Ollama response: {metadata['tokens_used']} tokens")
            return answer.strip(), metadata
        
        except Exception as e:
            logger.error(f"❌ Ollama error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
            )
        except ImportError:
            raise ImportError("anthropic not installed: pip install anthropic")
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Tuple[str, Dict]:
        """Generate using Claude API"""
        try:
            message = self.client.messages.create(
                model=config.model_name,
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                top_p=config.top_p
            )
            
            answer = message.content[0].text
            metadata = {
                'model': message.model,
                'tokens_used': message.usage.output_tokens +message.usage.input_tokens,
                'input_tokens': message.usage.input_tokens,
                'output_tokens': message.usage.output_tokens,
                'stop_reason': message.stop_reason,
                'provider': 'anthropic'
            }
            
            logger.info(f"✅ Claude response: {metadata['tokens_used']} tokens")
            return answer, metadata
        
        except Exception as e:
            logger.error(f"❌ Claude error: {e}")
            raise


class AnswerGenerator:
    """High-level answer generation interface"""
    
    def __init__(self, config: GenerationConfig):
        """Initialize with configuration"""
        self.config = config
        self.provider = self._get_provider()
        self.logger = logging.getLogger(__name__)
    
    def _get_provider(self) -> LLMProvider:
        """Get appropriate provider based on config"""
        provider_name = self.config.provider.lower()
        
        if provider_name == 'openai':
            return OpenAIProvider(api_key=self.config.api_key)
        elif provider_name == 'huggingface':
            return HuggingFaceProvider(model_name=self.config.model_name)
        elif provider_name in ['ollama', 'ollama_local']:
            return OllamaProvider(base_url=self.config.base_url or 'http://localhost:11434')
        elif provider_name == 'anthropic':
            return AnthropicProvider(api_key=self.config.api_key)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def generate(self, prompt: str) -> Tuple[str, Dict]:
        """
        Generate answer from prompt.
        
        Returns:
            (answer, metadata)
        """
        answer, metadata = self.provider.generate(prompt, self.config)
        return answer, metadata
    
    def generate_with_fallback(
        self,
        prompt: str,
        fallback_answer: str = "Verilen kaynaklar yeterli değil."
    ) -> Tuple[str, Dict]:
        """
        Generate with fallback if provider fails.
        """
        try:
            answer, metadata = self.generate(prompt)
            return answer, {**metadata, 'fallback': False}
        except Exception as e:
            self.logger.error(f"Generation failed, using fallback: {e}")
            return fallback_answer, {'fallback': True, 'error': str(e)}


# Default configurations for common models
DEFAULT_CONFIGS = {
    'openai_gpt35': GenerationConfig(
        provider='openai',
        model_name='gpt-3.5-turbo',
        temperature=0.3,  # Low for factual answers
        max_tokens=1500,
        top_p=0.9,
        timeout=60,
        api_key=None,
        base_url=None
    ),
    'openai_gpt4': GenerationConfig(
        provider='openai',
        model_name='gpt-4',
        temperature=0.2,  # Very low for legal accuracy
        max_tokens=2000,
        top_p=0.95,
        timeout=120,
        api_key=None,
        base_url=None
    ),
    'ollama_mistral': GenerationConfig(
        provider='ollama',
        model_name='mistral',
        temperature=0.3,
        max_tokens=1500,
        top_p=0.9,
        timeout=60,
        api_key=None,
        base_url='http://localhost:11434'
    ),
    'huggingface_gpt2': GenerationConfig(
        provider='huggingface',
        model_name='gpt2',
        temperature=0.3,
        max_tokens=500,
        top_p=0.9,
        timeout=30,
        api_key=None,
        base_url=None
    ),
}


if __name__ == '__main__':
    # Example: Using OpenAI provider
    config = DEFAULT_CONFIGS['openai_gpt35']
    # Note: Set OPENAI_API_KEY environment variable first
    
    generator = AnswerGenerator(config)
    
    prompt = "Türkiye'nin başkenti neresidir?"
    answer, metadata = generator.generate_with_fallback(prompt)
    
    print(f"Answer: {answer}")
    print(f"Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
