# File: core/safe_generation_wrapper.py
"""Safe generation wrapper that ensures consistent output format"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class SafeGenerationResult:
    """Safe generation result with all required fields"""
    generated_text: str = ""
    prompt: str = ""
    new_text: str = ""
    selected_model: str = "unknown"
    generation_successful: bool = False
    generation_time: float = 0.0
    generation_method: str = "basic"
    quality_score: float = 0.0
    input_analysis: Dict[str, Any] = None
    model_confidence: float = 0.5
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.input_analysis is None:
            self.input_analysis = {
                'task_type': 'conversational',
                'confidence_score': 0.5,
                'manifold_type': 'euclidean',
                'complexity_score': 0.5
            }
        
        # Calculate new_text if not provided
        if not self.new_text and self.generated_text and self.prompt:
            if self.generated_text.startswith(self.prompt):
                self.new_text = self.generated_text[len(self.prompt):].strip()
            else:
                self.new_text = self.generated_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            'generated_text': self.generated_text,
            'prompt': self.prompt,
            'new_text': self.new_text,
            'selected_model': self.selected_model,
            'generation_successful': self.generation_successful,
            'generation_time': self.generation_time,
            'generation_method': self.generation_method,
            'quality_score': self.quality_score,
            'input_analysis': self.input_analysis,
            'model_confidence': self.model_confidence,
            'error': self.error
        }

class SafeGenerationWrapper:
    """Wrapper to make any generation engine safe and consistent"""
    
    def __init__(self, base_engine, safe_classifier=None):
        self.base_engine = base_engine
        self.safe_classifier = safe_classifier
        self.generation_cache = {}
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate with safe, consistent output format"""
        
        start_time = time.time()
        
        try:
            # Analyze input first
            if self.safe_classifier:
                try:
                    analysis = self.safe_classifier.analyze_input(prompt)
                    input_analysis = {
                        'task_type': analysis.task_type.value,
                        'confidence_score': analysis.confidence_score,
                        'manifold_type': analysis.manifold_type,
                        'complexity_score': analysis.complexity_score
                    }
                except Exception:
                    input_analysis = {
                        'task_type': 'conversational',
                        'confidence_score': 0.5,
                        'manifold_type': 'euclidean',
                        'complexity_score': 0.5
                    }
            else:
                input_analysis = {
                    'task_type': 'conversational',
                    'confidence_score': 0.5,
                    'manifold_type': 'euclidean',
                    'complexity_score': 0.5
                }
            
            # Try generation
            result = self.base_engine.generate(prompt, **kwargs)
            generation_time = time.time() - start_time
            
            # Convert to safe format
            safe_result = SafeGenerationResult(
                generated_text=result.get('generated_text', ''),
                prompt=prompt,
                selected_model=result.get('selected_model', 'unknown'),
                generation_successful=result.get('generation_successful', False),
                generation_time=generation_time,
                generation_method=result.get('generation_method', 
                                          result.get('selected_method', 'basic')),
                quality_score=result.get('quality_score', 
                                       result.get('final_score', 0.0)),
                input_analysis=result.get('input_analysis', input_analysis),
                model_confidence=result.get('model_confidence', 
                                          result.get('confidence', 0.5)),
                error=result.get('error')
            )
            
            return safe_result.to_dict()
            
        except Exception as e:
            # Safe fallback for any error
            generation_time = time.time() - start_time
            
            error_result = SafeGenerationResult(
                generated_text=f"[Generation failed: {str(e)}]",
                prompt=prompt,
                new_text="",
                selected_model="error",
                generation_successful=False,
                generation_time=generation_time,
                generation_method="error",
                quality_score=0.0,
                input_analysis={
                    'task_type': 'conversational',
                    'confidence_score': 0.5,
                    'manifold_type': 'euclidean',
                    'complexity_score': 0.5
                },
                model_confidence=0.0,
                error=str(e)
            )
            
            return error_result.to_dict()
    
    def select_model(self, text: str):
        """Safe model selection"""
        try:
            return self.base_engine.select_model(text)
        except Exception as e:
            # Return safe default
            if self.safe_classifier:
                analysis = self.safe_classifier.analyze_input(text)
                return "unknown", analysis, 0.5
            else:
                from core.fixed_input_analysis import FixedInputAnalysis
                from core.types import TaskType
                default_analysis = FixedInputAnalysis(
                    task_type=TaskType.CONVERSATIONAL,
                    confidence_score=0.5
                )
                return "unknown", default_analysis, 0.5
    
    def get_model_stats(self):
        """Safe stats access"""
        try:
            return self.base_engine.get_model_stats()
        except Exception as e:
            return {
                'total_generations': 0,
                'error': str(e),
                'safe_engine_info': {
                    'engine_type': 'error',
                    'real_models_enabled': False,
                    'total_models': 0,
                    'available_models': []
                }
            }
    
    def evaluate_performance(self, test_cases: List[str] = None):
        """Safe performance evaluation"""
        try:
            return self.base_engine.evaluate_performance(test_cases)
        except Exception as e:
            return {
                'total_tests': 0,
                'successful_generations': 0,
                'success_rate': 0.0,
                'avg_generation_time': 0.0,
                'error': str(e)
            }
    
    def is_functional(self) -> bool:
        """Check if engine is functional"""
        try:
            if hasattr(self.base_engine, 'is_functional'):
                return self.base_engine.is_functional()
            else:
                # Test with simple generation
                test_result = self.generate("test", max_length=10)
                return test_result.get('generation_successful', False)
        except:
            return False

def create_safe_generation_wrapper(engine, classifier=None):
    """Create safe generation wrapper"""
    return SafeGenerationWrapper(engine, classifier)

def test_safe_generation():
    """Test safe generation wrapper"""
    
    print("🔒 Testing Safe Generation Wrapper")
    print("=" * 40)
    
    # Test with minimal mock engine
    class MockEngine:
        def generate(self, prompt, **kwargs):
            return {
                'generated_text': f"Mock response to: {prompt}",
                'generation_successful': True,
                'selected_model': 'mock_model'
            }
        
        def get_model_stats(self):
            return {'total_generations': 1}
    
    mock_engine = MockEngine()
    safe_wrapper = create_safe_generation_wrapper(mock_engine)
    
    # Test generation
    result = safe_wrapper.generate("Hello world")
    
    print("✅ Safe generation test:")
    print(f"   Success: {result['generation_successful']}")
    print(f"   Text: {result['generated_text']}")
    print(f"   New Text: {result['new_text']}")
    print(f"   Method: {result['generation_method']}")
    print(f"   Analysis: {result['input_analysis']}")
    
    return True

if __name__ == "__main__":
    test_safe_generation()