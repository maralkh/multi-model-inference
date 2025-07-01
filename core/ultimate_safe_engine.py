# File: core/ultimate_safe_engine.py
"""Ultimate safe engine that handles all compatibility issues"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class UltimateSafeEngine:
    """Ultimate safe engine that wraps any engine and ensures compatibility"""
    
    def __init__(self, base_engine, safe_classifier=None):
        self.base_engine = base_engine
        self.safe_classifier = safe_classifier
        self.generation_count = 0
        self.error_count = 0
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Ultra-safe generation with comprehensive error handling"""
        
        start_time = time.time()
        self.generation_count += 1
        
        try:
            # Step 1: Classify input safely
            if self.safe_classifier:
                try:
                    analysis = self.safe_classifier.analyze_input(prompt)
                    input_analysis = {
                        'task_type': analysis.task_type.value,
                        'confidence_score': getattr(analysis, 'confidence_score', 0.5),
                        'manifold_type': getattr(analysis, 'manifold_type', 'euclidean'),
                        'complexity_score': getattr(analysis, 'complexity_score', 0.5)
                    }
                except Exception as e:
                    logger.warning(f"Classification failed: {e}")
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
            
            # Step 2: Try generation with multiple fallbacks
            result = None
            generation_error = None
            
            # Try main generation
            try:
                result = self.base_engine.generate(prompt, **kwargs)
            except Exception as e:
                generation_error = str(e)
                logger.warning(f"Primary generation failed: {e}")
                
                # Try simpler generation
                try:
                    simple_kwargs = {'max_length': kwargs.get('max_length', 100)}
                    result = self.base_engine.generate(prompt, **simple_kwargs)
                except Exception as e2:
                    generation_error = str(e2)
                    logger.warning(f"Simple generation also failed: {e2}")
                    
                    # Try basic engine if available
                    try:
                        if hasattr(self.base_engine, 'engine'):
                            result = self.base_engine.engine.generate(prompt, max_length=50)
                    except Exception as e3:
                        generation_error = str(e3)
                        logger.warning(f"Basic engine also failed: {e3}")
            
            generation_time = time.time() - start_time
            
            # Step 3: Process result safely
            if result and isinstance(result, dict):
                # Safe extraction of all possible fields
                generated_text = str(result.get('generated_text', ''))
                selected_model = str(result.get('selected_model', 'unknown'))
                generation_successful = bool(result.get('generation_successful', False))
                
                # Calculate new_text
                new_text = generated_text
                if generated_text and prompt:
                    if generated_text.startswith(prompt):
                        new_text = generated_text[len(prompt):].strip()
                    elif prompt in generated_text:
                        # Find prompt and extract what comes after
                        prompt_end = generated_text.find(prompt) + len(prompt)
                        new_text = generated_text[prompt_end:].strip()
                
                # Build safe result
                safe_result = {
                    'generated_text': generated_text,
                    'prompt': prompt,
                    'new_text': new_text,
                    'selected_model': selected_model,
                    'generation_successful': generation_successful,
                    'generation_time': generation_time,
                    'generation_method': str(result.get('generation_method', result.get('selected_method', 'basic'))),
                    'quality_score': float(result.get('quality_score', result.get('final_score', 0.0))),
                    'input_analysis': result.get('input_analysis', input_analysis),
                    'model_confidence': float(result.get('model_confidence', result.get('confidence', 0.5))),
                    'error': result.get('error')
                }
                
                return safe_result
            
            else:
                # Generation completely failed
                self.error_count += 1
                return self._create_error_result(
                    prompt, 
                    generation_error or "Generation returned invalid result", 
                    generation_time,
                    input_analysis
                )
                
        except Exception as e:
            # Ultimate catch-all
            self.error_count += 1
            generation_time = time.time() - start_time
            logger.error(f"Ultimate generation failure: {e}")
            
            return self._create_error_result(
                prompt,
                str(e),
                generation_time,
                {
                    'task_type': 'conversational',
                    'confidence_score': 0.5,
                    'manifold_type': 'euclidean',
                    'complexity_score': 0.5
                }
            )
    
    def _create_error_result(self, prompt: str, error: str, generation_time: float, input_analysis: Dict) -> Dict[str, Any]:
        """Create a safe error result"""
        
        return {
            'generated_text': f"[Generation failed: {error[:100]}]",
            'prompt': prompt,
            'new_text': "",
            'selected_model': "error",
            'generation_successful': False,
            'generation_time': generation_time,
            'generation_method': "error",
            'quality_score': 0.0,
            'input_analysis': input_analysis,
            'model_confidence': 0.0,
            'error': error
        }
    
    def select_model(self, text: str) -> Tuple[str, Any, float]:
        """Safe model selection"""
        
        try:
            return self.base_engine.select_model(text)
        except Exception as e:
            logger.warning(f"Model selection failed: {e}")
            
            # Create safe analysis
            if self.safe_classifier:
                try:
                    analysis = self.safe_classifier.analyze_input(text)
                    return "safe_fallback", analysis, 0.5
                except Exception as e2:
                    logger.warning(f"Safe classifier also failed: {e2}")
            
            # Ultimate fallback
            from core.fixed_input_analysis import FixedInputAnalysis
            from core.types import TaskType
            
            fallback_analysis = FixedInputAnalysis(
                task_type=TaskType.CONVERSATIONAL,
                confidence_score=0.3
            )
            
            return "unknown", fallback_analysis, 0.3
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Safe stats retrieval"""
        
        try:
            base_stats = self.base_engine.get_model_stats()
            
            # Add safe engine stats
            base_stats['ultimate_safe_stats'] = {
                'total_generations': self.generation_count,
                'error_count': self.error_count,
                'success_rate': (self.generation_count - self.error_count) / max(1, self.generation_count),
                'safety_wrapper': 'enabled'
            }
            
            return base_stats
            
        except Exception as e:
            logger.warning(f"Stats retrieval failed: {e}")
            
            return {
                'total_generations': self.generation_count,
                'error_count': self.error_count,
                'ultimate_safe_stats': {
                    'safety_wrapper': 'enabled',
                    'base_engine_failed': True,
                    'error': str(e)
                },
                'safe_engine_info': {
                    'engine_type': 'safe_fallback',
                    'real_models_enabled': False,
                    'total_models': 1,
                    'available_models': ['safe_fallback']
                }
            }
    
    def evaluate_performance(self, test_cases: List[str] = None) -> Dict[str, Any]:
        """Safe performance evaluation"""
        
        if test_cases is None:
            test_cases = [
                "Hello world",
                "Calculate 2+2", 
                "Write hello",
                "What is AI?",
                "Thanks"
            ]
        
        results = {
            'total_tests': len(test_cases),
            'successful_generations': 0,
            'generation_times': [],
            'detailed_results': []
        }
        
        for prompt in test_cases:
            try:
                start_time = time.time()
                result = self.generate(prompt, max_length=50)
                generation_time = time.time() - start_time
                
                success = result.get('generation_successful', False)
                if success:
                    results['successful_generations'] += 1
                
                results['generation_times'].append(generation_time)
                results['detailed_results'].append({
                    'prompt': prompt,
                    'success': success,
                    'time': generation_time,
                    'selected_model': result.get('selected_model', 'unknown')
                })
                
            except Exception as e:
                results['detailed_results'].append({
                    'prompt': prompt,
                    'success': False,
                    'error': str(e),
                    'time': 0.0
                })
        
        # Calculate summary
        if results['generation_times']:
            results['avg_generation_time'] = sum(results['generation_times']) / len(results['generation_times'])
            results['success_rate'] = results['successful_generations'] / results['total_tests']
        else:
            results['avg_generation_time'] = 0.0
            results['success_rate'] = 0.0
        
        return results
    
    def is_functional(self) -> bool:
        """Check if engine is functional"""
        
        try:
            test_result = self.generate("test", max_length=10)
            return test_result.get('generation_successful', False)
        except:
            return False

def create_ultimate_safe_engine(base_engine, safe_classifier=None):
    """Create ultimate safe engine wrapper"""
    return UltimateSafeEngine(base_engine, safe_classifier)

def test_ultimate_safe_engine():
    """Test ultimate safe engine"""
    
    print("🛡️ Testing Ultimate Safe Engine")
    print("=" * 40)
    
    # Test with various engine types
    class BrokenEngine:
        def generate(self, prompt, **kwargs):
            raise Exception("Intentional failure")
        
        def get_model_stats(self):
            raise Exception("Stats broken")
    
    broken_engine = BrokenEngine()
    safe_engine = create_ultimate_safe_engine(broken_engine)
    
    # Test generation
    result = safe_engine.generate("Hello world")
    
    print("✅ Ultimate safe engine test:")
    print(f"   Success: {result['generation_successful']}")
    print(f"   Text: {result['generated_text']}")
    print(f"   Error: {result.get('error', 'None')}")
    
    # Test stats
    stats = safe_engine.get_model_stats()
    print(f"   Stats available: {'ultimate_safe_stats' in stats}")
    
    return True

if __name__ == "__main__":
    test_ultimate_safe_engine()