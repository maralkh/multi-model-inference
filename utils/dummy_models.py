# File: utils/dummy_models.py
"""Dummy models and utilities for demonstration"""

import torch
import torch.nn as nn
import copy
from typing import List
from core.input_types import ModelSpec, TaskType

# Mock model classes for demonstration
class MockLlamaModel(nn.Module):
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(32000, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(32)
        ])
        self.lm_head = nn.Linear(hidden_size, 32000)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.lm_head(x)

class MockProcessRewardModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.reward_head = nn.Linear(4096, 1)
    
    def compute_rewards(self, states):
        # Mock reward computation
        return torch.randn(len(states), 1)

class MockOutcomeRewardModel:
    def __init__(self, base_model, config):
        self.base_model = base_model
        self.config = config
        self.reward_head = nn.Linear(4096, 1)
    
    def compute_reward(self, text):
        # Mock reward computation
        return torch.randn(1).item()

def create_llama_7b():
    """Create a mock Llama 7B model"""
    return MockLlamaModel(hidden_size=4096)

class DummyTokenizer:
    """Enhanced dummy tokenizer"""
    
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.vocab_size = 32000
    
    def encode(self, text: str, return_tensors=None):
        tokens = [hash(word) % self.vocab_size for word in text.split()[:100]]
        if return_tensors == 'pt':
            return torch.tensor(tokens).unsqueeze(0)
        return tokens
    
    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return f"[Generated text from {len(tokens)} tokens]"
    
    def __call__(self, text, **kwargs):
        tokens = self.encode(text)
        max_length = kwargs.get('max_length', 512)
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return {
            'input_ids': torch.tensor(tokens).unsqueeze(0),
            'attention_mask': torch.ones(1, len(tokens))
        }

def create_specialized_models() -> List[ModelSpec]:
    """Create specialized models for different task types"""
    
    # Create base models (in practice, these would be different architectures/weights)
    math_model = create_llama_7b()
    creative_model = create_llama_7b() 
    reasoning_model = create_llama_7b()
    code_model = create_llama_7b()
    general_model = create_llama_7b()
    
    # Create dummy reward models (simplified for demo)
    def create_dummy_prm(base_model):
        return MockProcessRewardModel(copy.deepcopy(base_model))
    
    def create_dummy_orm(base_model):
        class MockConfig:
            def __init__(self):
                self.hidden_size = 4096
        config = MockConfig()
        return MockOutcomeRewardModel(copy.deepcopy(base_model), config)
    
    models = [
        ModelSpec(
            model_id="math_specialist",
            model=math_model,
            prm=create_dummy_prm(math_model),
            orm=create_dummy_orm(math_model),
            task_types=[TaskType.MATHEMATICAL, TaskType.SCIENTIFIC],
            specialized_domains=['mathematics', 'physics', 'engineering'],
            performance_metrics={
                'mathematical': 0.95,
                'scientific': 0.85,
                'reasoning': 0.80,
                'factual_qa': 0.70
            },
            description="Specialized for mathematical and scientific problem solving"
        ),
        
        ModelSpec(
            model_id="creative_specialist",
            model=creative_model,
            prm=None,  # Creative tasks don't need step-by-step rewards
            orm=create_dummy_orm(creative_model),
            task_types=[TaskType.CREATIVE_WRITING, TaskType.CONVERSATIONAL],
            specialized_domains=['literature', 'arts', 'entertainment'],
            performance_metrics={
                'creative_writing': 0.90,
                'conversational': 0.85,
                'factual_qa': 0.60,
                'reasoning': 0.65
            },
            description="Specialized for creative writing and conversational tasks"
        ),
        
        ModelSpec(
            model_id="reasoning_specialist", 
            model=reasoning_model,
            prm=create_dummy_prm(reasoning_model),
            orm=create_dummy_orm(reasoning_model),
            task_types=[TaskType.REASONING, TaskType.FACTUAL_QA],
            specialized_domains=['philosophy', 'logic', 'analysis'],
            performance_metrics={
                'reasoning': 0.92,
                'factual_qa': 0.88,
                'scientific': 0.75,
                'mathematical': 0.70
            },
            description="Specialized for complex reasoning and analytical tasks"
        ),
        
        ModelSpec(
            model_id="code_specialist",
            model=code_model,
            prm=create_dummy_prm(code_model),
            orm=create_dummy_orm(code_model),
            task_types=[TaskType.CODE_GENERATION, TaskType.REASONING],
            specialized_domains=['programming', 'software', 'algorithms'],
            performance_metrics={
                'code_generation': 0.93,
                'reasoning': 0.85,
                'mathematical': 0.80,
                'factual_qa': 0.70
            },
            description="Specialized for code generation and programming tasks"
        ),
        
        ModelSpec(
            model_id="general_model",
            model=general_model,
            prm=create_dummy_prm(general_model),
            orm=create_dummy_orm(general_model),
            task_types=list(TaskType),  # Handles all task types
            specialized_domains=['general'],
            performance_metrics={
                task_type.value: 0.75 for task_type in TaskType
            },
            description="General-purpose model for all task types"
        )
    ]
    
    return models