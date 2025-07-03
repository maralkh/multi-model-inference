# File: tests/test_basic.py
"""Basic tests for the multi-model inference system"""

import pytest
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.input_classifier import InputClassifier
from core.input_types import TaskType
from core.multi_model_engine import MultiModelInferenceEngine
from utils.dummy_models import create_specialized_models, DummyTokenizer

class TestInputClassifier:
    
    def setup_method(self):
        self.classifier = InputClassifier()
    
    def test_mathematical_classification(self):
        """Test classification of mathematical inputs"""
        math_prompts = [
            "Solve the equation x² + 2x - 3 = 0",
            "Calculate the derivative of f(x) = x³",
            "Find the integral of sin(x) dx"
        ]
        
        for prompt in math_prompts:
            analysis = self.classifier.analyze_input(prompt)
            assert analysis.task_type == TaskType.MATHEMATICAL
            assert analysis.confidence > 0.1
    
    def test_creative_writing_classification(self):
        """Test classification of creative writing inputs"""
        creative_prompts = [
            "Write a short story about time travel",
            "Create a poem about the ocean",
            "Develop a character for a fantasy novel"
        ]
        
        for prompt in creative_prompts:
            analysis = self.classifier.analyze_input(prompt)
            assert analysis.task_type == TaskType.CREATIVE_WRITING
            assert analysis.confidence > 0.1
    
    def test_code_generation_classification(self):
        """Test classification of code generation inputs"""
        code_prompts = [
            "Write a Python function to sort a list",
            "Implement binary search in JavaScript",
            "Create a class for a binary tree"
        ]
        
        for prompt in code_prompts:
            analysis = self.classifier.analyze_input(prompt)
            assert analysis.task_type == TaskType.CODE_GENERATION
            assert analysis.confidence > 0.1
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        prompt = "Calculate the derivative of f(x) = x² + 3x + 1"
        analysis = self.classifier.analyze_input(prompt)
        
        assert 'length' in analysis.features
        assert 'word_count' in analysis.features
        assert 'mathematical_symbols' in analysis.features
        assert analysis.features['mathematical_symbols'] > 0

class TestMultiModelEngine:
    
    def setup_method(self):
        self.models = create_specialized_models()
        self.tokenizer = DummyTokenizer()
        self.engine = MultiModelInferenceEngine(
            models=self.models,
            tokenizer=self.tokenizer,
            default_model_id="general_model"
        )
    
    def test_model_initialization(self):
        """Test proper model initialization"""
        assert len(self.engine.models) == 5
        assert "math_specialist" in self.engine.models
        assert "creative_specialist" in self.engine.models
        assert "general_model" in self.engine.models
    
    def test_model_selection(self):
        """Test model selection logic"""
        # Test mathematical input
        math_prompt = "Solve x² - 4x + 3 = 0"
        model_id, analysis, confidence = self.engine.select_model(math_prompt)
        
        assert analysis.task_type == TaskType.MATHEMATICAL
        assert confidence > 0
        assert model_id in self.engine.models
    
    def test_generation(self):
        """Test text generation"""
        prompt = "Calculate the derivative of x³"
        result = self.engine.generate(prompt, max_length=100)
        
        assert 'selected_model' in result
        assert 'generated_text' in result
        assert 'generation_time' in result
        assert 'input_analysis' in result
        assert result['generation_successful']
    
    def test_forced_model_selection(self):
        """Test forcing specific model selection"""
        prompt = "Any prompt"
        result = self.engine.generate(
            prompt, 
            force_model="creative_specialist"
        )
        
        assert result['selected_model'] == "creative_specialist"
    
    def test_usage_statistics(self):
        """Test usage statistics tracking"""
        initial_stats = self.engine.get_model_stats()
        
        # Generate some requests
        prompts = [
            "Solve equation",
            "Write story", 
            "Code function"
        ]
        
        for prompt in prompts:
            self.engine.generate(prompt)
        
        final_stats = self.engine.get_model_stats()
        assert final_stats['total_generations'] >= initial_stats['total_generations']

class TestIntegration:
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        models = create_specialized_models()
        tokenizer = DummyTokenizer()
        engine = MultiModelInferenceEngine(models, tokenizer)
        
        test_cases = [
            {
                'prompt': "Solve the quadratic equation x² - 5x + 6 = 0",
                'expected_task': 'mathematical'
            },
            {
                'prompt': "Write a short story about a robot",
                'expected_task': 'creative_writing'
            },
            {
                'prompt': "Implement bubble sort in Python",
                'expected_task': 'code_generation'
            }
        ]
        
        for case in test_cases:
            result = engine.generate(case['prompt'])
            
            # Verify result structure
            assert 'selected_model' in result
            assert 'generated_text' in result
            assert 'input_analysis' in result
            assert 'generation_time' in result
            
            # Verify task type detection
            assert result['input_analysis']['task_type'] == case['expected_task']

def test_basic_functionality():
    """Test basic system functionality"""
    # Test model creation
    models = create_specialized_models()
    assert len(models) == 5
    
    # Test tokenizer
    tokenizer = DummyTokenizer()
    text = "Hello world"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    
    # Test engine creation
    engine = MultiModelInferenceEngine(models, tokenizer)
    assert engine is not None
    assert len(engine.models) == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])