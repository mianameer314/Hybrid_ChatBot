"""
Multi-LLM Provider Management
Supports OpenAI, Gemini, and HuggingFace models
"""
import openai
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator
import logging
import time
from abc import ABC, abstractmethod
import asyncio
import re

# HuggingFace imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Google Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini SDK not available")

from app.core.config import settings

logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class QuotaExceededException(Exception):
    """Raised when API quota is exceeded"""
    def __init__(self, provider: str, message: str = None):
        self.provider = provider
        self.message = message or f"{provider} API quota exceeded"
        super().__init__(self.message)

class ProviderUnavailableException(Exception):
    """Raised when a provider is temporarily unavailable"""
    def __init__(self, provider: str, message: str = None):
        self.provider = provider
        self.message = message or f"{provider} provider is temporarily unavailable"
        super().__init__(self.message)

def is_quota_error(error_message: str) -> bool:
    """Check if error message indicates quota exceeded"""
    quota_keywords = [
        "quota", "limit", "rate limit", "insufficient_quota", 
        "billing", "exceeded", "429", "too many requests"
    ]
    error_lower = error_message.lower()
    return any(keyword in error_lower for keyword in quota_keywords)

def is_auth_error(error_message: str) -> bool:
    """Check if error message indicates authentication issue"""
    auth_keywords = [
        "unauthorized", "401", "invalid api key", "authentication", "api key"
    ]
    error_lower = error_message.lower()
    return any(keyword in error_lower for keyword in auth_keywords)

class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self):
        """Initialize the provider"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        pass
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat completion"""
        pass
    
    @abstractmethod
    async def stream_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat completion"""
        pass
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count (basic implementation)"""
        return len(text.split()) * 1.3  # Rough estimation

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        model_name = model_name or settings.OPENAI_MODEL
        api_key = api_key or settings.OPENAI_API_KEY
        super().__init__(model_name, api_key)
        self.client = None
    
    async def initialize(self):
        """Initialize OpenAI client"""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.is_initialized = True
            logger.info(f"OpenAI provider initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0)
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat completion"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            logger.error(f"OpenAI chat completion error: {error_str}")
            
            # Check if it's a quota error
            if is_quota_error(error_str):
                raise QuotaExceededException("OpenAI", error_str)
            elif is_auth_error(error_str):
                raise ProviderUnavailableException("OpenAI", f"Authentication error: {error_str}")
            else:
                raise
    
    async def stream_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat completion"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Gemini SDK not available")
        
        model_name = model_name or settings.GEMINI_MODEL
        api_key = api_key or settings.GEMINI_API_KEY
        super().__init__(model_name, api_key)
        self.model = None
    
    async def initialize(self):
        """Initialize Gemini client"""
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.is_initialized = True
            logger.info(f"Gemini provider initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.8),
                    top_k=kwargs.get('top_k', 40),
                    max_output_tokens=kwargs.get('max_tokens', 1000)
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat completion"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Convert messages to Gemini format
            history = []
            current_message = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    current_message = msg["content"] + "\n\n"
                elif msg["role"] == "user":
                    current_message += f"User: {msg['content']}"
                elif msg["role"] == "assistant":
                    history.append({"role": "user", "parts": [current_message]})
                    history.append({"role": "model", "parts": [msg["content"]]})
                    current_message = ""
            
            # Add final user message if exists
            if current_message:
                chat = self.model.start_chat(history=history[:-1] if history else [])
                response = await asyncio.to_thread(chat.send_message, current_message)
            else:
                response = await asyncio.to_thread(
                    self.model.generate_content, 
                    "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                )
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini chat completion error: {e}")
            raise
    
    async def stream_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat completion"""
        # Note: Gemini streaming is more complex, this is a simplified implementation
        response = await self.chat_completion(messages, **kwargs)
        
        # Simulate streaming by yielding chunks
        words = response.split()
        for i in range(0, len(words), 3):  # Yield 3 words at a time
            chunk = " ".join(words[i:i+3])
            if i + 3 < len(words):
                chunk += " "
            yield chunk
            await asyncio.sleep(0.1)  # Small delay to simulate streaming

class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace transformers provider"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        model_name = model_name or settings.HUGGINGFACE_MODEL
        super().__init__(model_name, api_key)
        self.tokenizer = None
        self.model = None
        self.generator = None
    
    async def initialize(self):
        """Initialize HuggingFace model"""
        try:
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.is_initialized = True
            logger.info(f"HuggingFace provider initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace provider: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            result = await asyncio.to_thread(
                self.generator,
                prompt,
                max_length=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            # Remove the original prompt from the response
            return generated_text[len(prompt):].strip()
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat completion"""
        # Convert messages to a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        return await self.generate(prompt, **kwargs)
    
    async def stream_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat completion"""
        response = await self.chat_completion(messages, **kwargs)
        
        # Simulate streaming
        words = response.split()
        for i in range(0, len(words), 2):  # Yield 2 words at a time
            chunk = " ".join(words[i:i+2])
            if i + 2 < len(words):
                chunk += " "
            yield chunk
            await asyncio.sleep(0.05)

class LLMProviderManager:
    """Manager for all LLM providers"""
    
    def __init__(self):
        self.providers = {}
        self.default_provider = settings.DEFAULT_LLM
    
    def add_provider(self, name: str, provider: BaseLLMProvider):
        """Add a provider"""
        self.providers[name] = provider
    
    async def get_provider(self, provider_name: str = None) -> BaseLLMProvider:
        """Get a provider by name"""
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            # Initialize provider on first use
            if provider_name == "openai":
                provider = OpenAIProvider()
            elif provider_name == "gemini":
                provider = GeminiProvider()
            elif provider_name == "huggingface":
                provider = HuggingFaceProvider()
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
            
            await provider.initialize()
            self.providers[provider_name] = provider
        
        return self.providers[provider_name]
    
    async def generate_response_with_fallback(
        self, 
        messages: List[Dict[str, str]], 
        provider_name: str = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using specified provider with automatic fallback"""
        provider_name = provider_name or self.default_provider
        
        # Define fallback order
        fallback_order = []
        if provider_name == "openai":
            fallback_order = ["openai", "gemini", "huggingface"]
        elif provider_name == "gemini":
            fallback_order = ["gemini", "openai", "huggingface"]
        else:
            fallback_order = ["huggingface", "gemini", "openai"]
        
        last_error = None
        
        for current_provider in fallback_order:
            try:
                logger.info(f"Attempting to use {current_provider} provider")
                
                # Skip providers without API keys
                if current_provider == "openai" and not settings.OPENAI_API_KEY:
                    logger.warning("OpenAI API key not available, skipping")
                    continue
                elif current_provider == "gemini" and not settings.GEMINI_API_KEY:
                    logger.warning("Gemini API key not available, skipping")
                    continue
                
                provider = await self.get_provider(current_provider)
                
                if stream:
                    response = provider.stream_chat_completion(messages, **kwargs)
                else:
                    response = await provider.chat_completion(messages, **kwargs)
                
                logger.info(f"Successfully generated response using {current_provider}")
                return {
                    'response': response,
                    'model_used': current_provider,
                    'fallback_used': current_provider != provider_name
                }
                
            except QuotaExceededException as e:
                logger.warning(f"{current_provider} quota exceeded: {e.message}")
                last_error = e
                continue
                
            except ProviderUnavailableException as e:
                logger.warning(f"{current_provider} unavailable: {e.message}")
                last_error = e
                continue
                
            except Exception as e:
                logger.error(f"Error with {current_provider} provider: {str(e)}")
                last_error = e
                continue
        
        # If all providers failed, raise the last error
        if last_error:
            if isinstance(last_error, (QuotaExceededException, ProviderUnavailableException)):
                raise last_error
            else:
                raise Exception(f"All LLM providers failed. Last error: {str(last_error)}")
        else:
            raise Exception("No LLM providers available")
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        provider_name: str = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate response using specified provider (legacy method)"""
        result = await self.generate_response_with_fallback(
            messages, provider_name, stream, **kwargs
        )
        return result['response']

# Global provider manager
llm_manager = LLMProviderManager()
