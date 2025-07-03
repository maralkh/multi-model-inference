# File: utils/real_models.py
"""Real model implementations with Hugging Face transformers"""

import torch
import torch.nn as nn
import copy
import logging
from typing import List, Dict, Optional, Union
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    pipeline
)
from core.input_types import ModelSpec, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSize:
    """Model size configurations"""
    SMALL = "small"    # 1-3B params (for research/development)
    MEDIUM = "medium"  # 7-13B params (balanced)
    LARGE = "large"    # 30B+ params (production)

class RealModelConfig:
    """Configuration for real model loading"""
    def __init__(self, 
                 model_size: str = ModelSize.SMALL,
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.float16):
        self.model_size = model_size
        self.device = device
        self.torch_dtype = torch_dtype

# Model configurations by size and specialization
MODEL_CONFIGS = {
    ModelSize.SMALL: {
        "math_specialist": "gpt2",
        "creative_specialist": "gpt2",
        "reasoning_specialist": "gpt2", 
        "code_specialist": "gpt2",
        "general_model": "gpt2"
    },
    ModelSize.MEDIUM: {
        "math_specialist": "gpt2-medium",
        "creative_specialist": "gpt2-medium", 
        "reasoning_specialist": "gpt2-medium",
        "code_specialist": "gpt2-medium",
        "general_model": "gpt2-medium"
    },
    ModelSize.LARGE: {
        "math_specialist": "gpt2-large",
        "creative_specialist": "gpt2-large",
        "reasoning_specialist": "gpt2-large", 
        "code_specialist": "gpt2-large",
        "general_model": "gpt2-large"
    }
}

class RealModelWrapper:
    """Wrapper for real Hugging Face models to match our interface"""
    
    def __init__(self, model_name: str, config: RealModelConfig):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
    
        def _load_model(self):
            """Load the actual model and tokenizer"""
            try:
                logger.info(f"Loading model: {self.model_name}")
                
                # Configure model loading parameters
                model_kwargs = {
                    "torch_dtype": self.config.torch_dtype,
                    "device_map": self.config.device if self.config.device != "auto" else "auto",
                }
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Add pad token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # Create generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map=self.config.device if self.config.device != "auto" else None
                )
                
                logger.info(f"Successfully loaded {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {self.model_name}: {e}")
                raise
    
    def generate(self, prompt: str, max_length: int = 200, 
                temperature: float = 0.7, **kwargs) -> Dict:
        """Generate text using the real model"""
        try:
            # Generate using pipeline
            outputs = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
            
            generated_text = outputs[0]['generated_text']
            
            return {
                'generated_text': generated_text,
                'tokens_generated': len(self.tokenizer.encode(generated_text)),
                'model_name': self.model_name,
                'generation_successful': True
            }
            
        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            return {
                'generated_text': f"[Generation failed: {str(e)}]",
                'tokens_generated': 0,
                'model_name': self.model_name,
                'generation_successful': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "parameters": self.model.num_parameters() if hasattr(self.model, 'num_parameters') else "unknown",
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
            "status": "loaded"
        }

class MockProcessRewardModel:
    """Mock PRM for real models (can be replaced with actual PRM)"""
    def __init__(self, base_model):
        self.base_model = base_model
    
    def compute_rewards(self, states):
        # Mock implementation - replace with real PRM
        return torch.randn(len(states), 1) * 0.1 + 0.8

class MockOutcomeRewardModel:
    """Mock ORM for real models (can be replaced with actual ORM)"""
    def __init__(self, base_model):
        self.base_model = base_model
    
    def compute_reward(self, text):
        # Mock implementation - replace with real ORM
        return torch.randn(1).item() * 0.1 + 0.8

def create_real_models(config: RealModelConfig) -> List[ModelSpec]:
    """Create real model specifications"""
    
    model_configs = MODEL_CONFIGS[config.model_size]
    models = []
    
    logger.info(f"Creating {config.model_size} models...")
    
    # Math Specialist
    try:
        math_model = RealModelWrapper(model_configs["math_specialist"], config)
        models.append(ModelSpec(
            model_id="math_specialist",
            model=math_model,
            prm=MockProcessRewardModel(math_model),
            orm=MockOutcomeRewardModel(math_model),
            task_types=[TaskType.MATHEMATICAL, TaskType.SCIENTIFIC],
            specialized_domains=['mathematics', 'physics', 'engineering'],
            performance_metrics={
                'mathematical': 0.95,
                'scientific': 0.85,
                'reasoning': 0.80,
                'factual_qa': 0.70
            },
            description=f"Math specialist using {model_configs['math_specialist']}"
        ))
    except Exception as e:
        logger.warning(f"Failed to load math specialist: {e}")
    
    # Creative Specialist
    try:
        creative_model = RealModelWrapper(model_configs["creative_specialist"], config)
        models.append(ModelSpec(
            model_id="creative_specialist",
            model=creative_model,
            prm=None,  # Creative tasks typically don't need step-by-step rewards
            orm=MockOutcomeRewardModel(creative_model),
            task_types=[TaskType.CREATIVE_WRITING, TaskType.CONVERSATIONAL],
            specialized_domains=['literature', 'arts', 'entertainment'],
            performance_metrics={
                'creative_writing': 0.90,
                'conversational': 0.85,
                'factual_qa': 0.60,
                'reasoning': 0.65
            },
            description=f"Creative specialist using {model_configs['creative_specialist']}"
        ))
    except Exception as e:
        logger.warning(f"Failed to load creative specialist: {e}")
    
    # Reasoning Specialist
    try:
        reasoning_model = RealModelWrapper(model_configs["reasoning_specialist"], config)
        models.append(ModelSpec(
            model_id="reasoning_specialist",
            model=reasoning_model,
            prm=MockProcessRewardModel(reasoning_model),
            orm=MockOutcomeRewardModel(reasoning_model),
            task_types=[TaskType.REASONING, TaskType.FACTUAL_QA],
            specialized_domains=['philosophy', 'logic', 'analysis'],
            performance_metrics={
                'reasoning': 0.92,
                'factual_qa': 0.88,
                'scientific': 0.75,
                'mathematical': 0.70
            },
            description=f"Reasoning specialist using {model_configs['reasoning_specialist']}"
        ))
    except Exception as e:
        logger.warning(f"Failed to load reasoning specialist: {e}")
    
    # Code Specialist
    try:
        code_model = RealModelWrapper(model_configs["code_specialist"], config)
        models.append(ModelSpec(
            model_id="code_specialist",
            model=code_model,
            prm=MockProcessRewardModel(code_model),
            orm=MockOutcomeRewardModel(code_model),
            task_types=[TaskType.CODE_GENERATION, TaskType.REASONING],
            specialized_domains=['programming', 'software', 'algorithms'],
            performance_metrics={
                'code_generation': 0.93,
                'reasoning': 0.85,
                'mathematical': 0.80,
                'factual_qa': 0.70
            },
            description=f"Code specialist using {model_configs['code_specialist']}"
        ))
    except Exception as e:
        logger.warning(f"Failed to load code specialist: {e}")
    
    # General Model
    try:
        general_model = RealModelWrapper(model_configs["general_model"], config)
        models.append(ModelSpec(
            model_id="general_model",
            model=general_model,
            prm=MockProcessRewardModel(general_model),
            orm=MockOutcomeRewardModel(general_model),
            task_types=list(TaskType),
            specialized_domains=['general'],
            performance_metrics={
                task_type.value: 0.75 for task_type in TaskType
            },
            description=f"General model using {model_configs['general_model']}"
        ))
    except Exception as e:
        logger.warning(f"Failed to load general model: {e}")
    
    logger.info(f"Successfully created {len(models)} real models")
    return models

class RealTokenizer:
    """Real tokenizer wrapper to match our interface"""
    
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode(self, text: str, return_tensors=None):
        return self.tokenizer.encode(text, return_tensors=return_tensors)
    
    def decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

def create_real_tokenizer(model_size: str = ModelSize.SMALL) -> RealTokenizer:
    """Create a real tokenizer based on model size"""
    model_configs = MODEL_CONFIGS[model_size]
    # Use the general model's tokenizer as default
    return RealTokenizer(model_configs["general_model"])

# Utility functions for easy model creation

def create_small_models() -> List[ModelSpec]:
    """Create small models for research/development"""
    config = RealModelConfig(model_size=ModelSize.SMALL)
    return create_real_models(config)

def create_medium_models() -> List[ModelSpec]:
    """Create medium models for balanced performance"""
    config = RealModelConfig(model_size=ModelSize.MEDIUM)
    return create_real_models(config)

def create_large_models() -> List[ModelSpec]:
    """Create large models for production"""
    config = RealModelConfig(model_size=ModelSize.LARGE)
    return create_real_models(config)

def create_optimized_models(device: str = "cuda") -> List[ModelSpec]:
    """Create optimized models for production deployment"""
    config = RealModelConfig(
        model_size=ModelSize.LARGE,
        device=device,
        torch_dtype=torch.float16
    )
    return create_real_models(config)

# Example usage and testing
if __name__ == "__main__":
    # Test with small models first
    print("Testing small models...")
    try:
        small_models = create_small_models()
        print(f"Created {len(small_models)} small models successfully")
        
        # Test one model
        if small_models:
            test_model = small_models[0]
            print(f"Testing {test_model.model_id}...")
            result = test_model.model.generate("Hello, how are you?", max_length=50)
            print(f"Generation result: {result}")
            
    except Exception as e:
        print(f"Error testing small models: {e}")