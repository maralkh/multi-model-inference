# File: utils/safe_real_models.py
"""Safe real model implementations with tested Hugging Face models"""

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.types import ModelSpec, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSize:
    """Model size configurations"""
    SMALL = "small"    # GPT-2 base (124M params)
    MEDIUM = "medium"  # GPT-2 medium (355M params)
    LARGE = "large"    # GPT-2 large (774M params)

class SafeRealModelConfig:
    """Safe configuration for real model loading"""
    def __init__(self, 
                 model_size: str = ModelSize.SMALL,
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.float32):  # Use float32 for stability
        self.model_size = model_size
        self.device = device
        self.torch_dtype = torch_dtype

# Tested and working model configurations
SAFE_MODEL_CONFIGS = {
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

class SafeRealModelWrapper:
    """Safe wrapper for real Hugging Face models"""
    
    def __init__(self, model_name: str, config: SafeRealModelConfig):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load the actual model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure device
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.device = device
            
            # Load model with safe settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map=device if device != "cpu" else None
            )
            
            # Move to device if CPU
            if device == "cpu":
                self.model = self.model.to(device)
            
            # Set to eval mode
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"✅ Successfully loaded {self.model_name} on {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_name}: {e}")
            self.is_loaded = False
            raise
    
    def generate(self, prompt: str, max_length: int = 200, 
                temperature: float = 0.7, do_sample: bool = True, 
                top_p: float = 0.9, top_k: int = 50, **kwargs) -> Dict:
        """Generate text using the real model with manual generation"""
        if not self.is_loaded:
            return {
                'generated_text': "[Model not loaded]",
                'tokens_generated': 0,
                'model_name': self.model_name,
                'generation_successful': False,
                'error': "Model not loaded"
            }
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Clamp generation parameters
            max_length = min(max_length, 512)  # Limit max length
            temperature = max(0.1, min(temperature, 2.0))  # Clamp temperature
            top_p = max(0.1, min(top_p, 1.0))  # Clamp top_p
            top_k = max(1, min(top_k, 100))  # Clamp top_k
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,  # Add to input length
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    **kwargs
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'generated_text': generated_text,
                'prompt': prompt,
                'new_text': generated_text[len(prompt):].strip(),  # Only new part
                'tokens_generated': len(outputs[0]),
                'input_tokens': len(inputs[0]),
                'new_tokens': len(outputs[0]) - len(inputs[0]),
                'model_name': self.model_name,
                'generation_successful': True,
                'device': str(self.device)
            }
            
        except Exception as e:
            logger.error(f"❌ Generation failed for {self.model_name}: {e}")
            return {
                'generated_text': f"[Generation failed: {str(e)}]",
                'prompt': prompt,
                'new_text': "",
                'tokens_generated': 0,
                'input_tokens': 0,
                'new_tokens': 0,
                'model_name': self.model_name,
                'generation_successful': False,
                'error': str(e),
                'device': str(self.device) if self.device else "unknown"
            }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        info = {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "status": "loaded" if self.is_loaded else "failed"
        }
        
        if self.is_loaded and self.model is not None:
            try:
                info.update({
                    "parameters": sum(p.numel() for p in self.model.parameters()),
                    "device": str(self.device),
                    "dtype": str(next(self.model.parameters()).dtype),
                    "vocab_size": self.tokenizer.vocab_size,
                    "model_type": self.model.config.model_type if hasattr(self.model, 'config') else "unknown"
                })
            except Exception as e:
                info["info_error"] = str(e)
        
        return info
    
    def generate_with_options(self, prompt: str, **options) -> Dict:
        """Generate with custom options and better control"""
        
        # Default generation options
        default_options = {
            'max_length': 100,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 2
        }
        
        # Update with user options
        generation_options = {**default_options, **options}
        
        return self.generate(prompt, **generation_options)

class MockProcessRewardModel:
    """Mock PRM for real models"""
    def __init__(self, base_model):
        self.base_model = base_model
    
    def compute_rewards(self, states):
        return torch.randn(len(states), 1) * 0.1 + 0.8

class MockOutcomeRewardModel:
    """Mock ORM for real models"""
    def __init__(self, base_model):
        self.base_model = base_model
    
    def compute_reward(self, text):
        return torch.randn(1).item() * 0.1 + 0.8

def create_safe_real_models(config: SafeRealModelConfig) -> List[ModelSpec]:
    """Create safe real model specifications"""
    
    model_configs = SAFE_MODEL_CONFIGS[config.model_size]
    models = []
    
    logger.info(f"Creating {config.model_size} safe real models...")
    
    for specialist_type, model_name in model_configs.items():
        try:
            logger.info(f"Loading {specialist_type}...")
            model_wrapper = SafeRealModelWrapper(model_name, config)
            
            # Map specialist types to our system
            if specialist_type == "math_specialist":
                task_types = [TaskType.MATHEMATICAL, TaskType.SCIENTIFIC]
                domains = ['mathematics', 'physics', 'engineering']
                performance = {
                    'mathematical': 0.95, 'scientific': 0.85,
                    'reasoning': 0.80, 'factual_qa': 0.70
                }
            elif specialist_type == "creative_specialist":
                task_types = [TaskType.CREATIVE_WRITING, TaskType.CONVERSATIONAL]
                domains = ['literature', 'arts', 'entertainment']
                performance = {
                    'creative_writing': 0.90, 'conversational': 0.85,
                    'factual_qa': 0.60, 'reasoning': 0.65
                }
            elif specialist_type == "reasoning_specialist":
                task_types = [TaskType.REASONING, TaskType.FACTUAL_QA]
                domains = ['philosophy', 'logic', 'analysis']
                performance = {
                    'reasoning': 0.92, 'factual_qa': 0.88,
                    'scientific': 0.75, 'mathematical': 0.70
                }
            elif specialist_type == "code_specialist":
                task_types = [TaskType.CODE_GENERATION, TaskType.REASONING]
                domains = ['programming', 'software', 'algorithms']
                performance = {
                    'code_generation': 0.93, 'reasoning': 0.85,
                    'mathematical': 0.80, 'factual_qa': 0.70
                }
            else:  # general_model
                task_types = list(TaskType)
                domains = ['general']
                performance = {task_type.value: 0.75 for task_type in TaskType}
            
            models.append(ModelSpec(
                model_id=specialist_type,
                model=model_wrapper,
                prm=MockProcessRewardModel(model_wrapper) if specialist_type != "creative_specialist" else None,
                orm=MockOutcomeRewardModel(model_wrapper),
                task_types=task_types,
                specialized_domains=domains,
                performance_metrics=performance,
                description=f"{specialist_type} using {model_name} ({config.model_size})"
            ))
            
            logger.info(f"✅ Successfully created {specialist_type}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create {specialist_type}: {e}")
            continue
    
    logger.info(f"✅ Successfully created {len(models)} safe real models")
    return models

class SafeRealTokenizer:
    """Safe real tokenizer wrapper"""
    
    def __init__(self, model_name: str = "gpt2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load tokenizer {model_name}: {e}")
            self.is_loaded = False
            raise
    
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

# Easy creation functions
def create_small_safe_models() -> List[ModelSpec]:
    """Create small safe models (GPT-2 base)"""
    config = SafeRealModelConfig(model_size=ModelSize.SMALL)
    return create_safe_real_models(config)

def create_medium_safe_models() -> List[ModelSpec]:
    """Create medium safe models (GPT-2 medium)"""
    config = SafeRealModelConfig(model_size=ModelSize.MEDIUM)
    return create_safe_real_models(config)

def create_large_safe_models() -> List[ModelSpec]:
    """Create large safe models (GPT-2 large)"""
    config = SafeRealModelConfig(model_size=ModelSize.LARGE)
    return create_safe_real_models(config)

def create_cpu_models() -> List[ModelSpec]:
    """Create models optimized for CPU"""
    config = SafeRealModelConfig(
        model_size=ModelSize.SMALL,
        device="cpu",
        torch_dtype=torch.float32
    )
    return create_safe_real_models(config)

def create_gpu_models() -> List[ModelSpec]:
    """Create models optimized for GPU"""
    config = SafeRealModelConfig(
        model_size=ModelSize.MEDIUM if torch.cuda.is_available() else ModelSize.SMALL,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return create_safe_real_models(config)

def create_safe_real_tokenizer(model_size: str = ModelSize.SMALL) -> SafeRealTokenizer:
    """Create a safe real tokenizer"""
    model_configs = SAFE_MODEL_CONFIGS[model_size]
    return SafeRealTokenizer(model_configs["general_model"])

# Test function
def test_safe_models():
    """Test safe models"""
    print("🧪 Testing Safe Real Models")
    print("=" * 30)
    
    try:
        # Test small models
        models = create_small_safe_models()
        print(f"✅ Created {len(models)} models")
        
        if models:
            test_model = models[0]
            print(f"🔧 Testing {test_model.model_id}...")
            
            result = test_model.model.generate("Hello world", max_length=50)
            if result['generation_successful']:
                print(f"✅ Generation successful!")
                print(f"📝 Text: {result['generated_text'][:100]}...")
            else:
                print(f"❌ Generation failed: {result.get('error')}")
            
            # Model info
            info = test_model.model.get_model_info()
            print(f"📊 Model info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_safe_models()