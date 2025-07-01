# File: core/safe_engine_fallback.py
"""Safe fallback engine that works without complex dependencies"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from core.types import TaskType, InputAnalysis, ModelSpec
from core.input_classifier import InputClassifier
from core.multi_model_engine import MultiModelInferenceEngine
from core.updated_multi_model_engine import UpdatedMultiModelEngine

logger = logging.getLogger(__name__)

class SafeMultiModelEngine:
    """Safe multi-model engine with robust fallbacks"""
    
    def __init__(self, enable_real_models: bool = True):
        self.enable_real_models = enable_real_models
        self.models = {}
        self.tokenizer = None
        self.classifier = None
        self.engine = None
        
        # Initialize with fallbacks
        self._initialize_safe_engine()
    
    def _initialize_safe_engine(self):
        """Initialize engine with safe fallbacks"""
        
        logger.info("🔧 Initializing safe multi-model engine...")
        
        # Try to load real models first
        if self.enable_real_models:
            real_success = self._try_load_real_models()
            if real_success:
                logger.info("✅ Real models loaded successfully")
                return
        
        # Fallback to dummy models
        logger.info("🔄 Loading dummy models as fallback...")
        self._load_dummy_models()
    
    def _try_load_real_models(self) -> bool:
        """Try to load real models with comprehensive error handling"""
        
        try:
            from utils.safe_real_models import create_small_safe_models, create_safe_real_tokenizer, ModelSize
            
            # Load models
            safe_models = create_small_safe_models()
            self.tokenizer = create_safe_real_tokenizer(ModelSize.SMALL)
            
            if not safe_models:
                logger.warning("No real models created")
                return False
            
            # Try extended generation if available
            try:
                from utils.extended_generation import ExtendedGenerationWrapper, GenerationConfig
                
                # Wrap first model with extended generation
                if safe_models:
                    config = GenerationConfig(
                        max_length=100,
                        tree_depth=2,
                        branching_factor=2,
                        enable_backtrack=True
                    )
                    
                    # Only wrap the first model to test
                    extended_model = ExtendedGenerationWrapper(safe_models[0].model, config)
                    safe_models[0].model = extended_model
                    
                    logger.info("✅ Extended generation enabled")
                
            except Exception as e:
                logger.warning(f"Extended generation not available: {e}")
            
            # Convert to engine format
            for model_spec in safe_models:
                self.models[model_spec.model_id] = model_spec
            
            # Create classifier
            self.classifier = InputClassifier()
            
            # Create engine
            model_list = list(self.models.values())
            self.engine = UpdatedMultiModelEngine(model_list, self.tokenizer)
            
            logger.info(f"✅ Real models engine ready with {len(self.models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load real models: {e}")
            return False
    
    def _load_dummy_models(self):
        """Load dummy models as ultimate fallback"""
        
        try:
            from utils.dummy_models import create_specialized_models, DummyTokenizer
            
            dummy_models = create_specialized_models()
            self.tokenizer = DummyTokenizer()
            
            # Convert to dictionary
            for model_spec in dummy_models:
                self.models[model_spec.model_id] = model_spec
            
            # Create classifier
            self.classifier = InputClassifier()
            
            # Create engine
            self.engine = UpdatedMultiModelEngine(dummy_models, self.tokenizer)
            
            logger.info(f"✅ Dummy models engine ready with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load dummy models: {e}")
            raise RuntimeError("Cannot initialize any models - system is not functional")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using the loaded engine"""
        
        if not self.engine:
            return {
                'generated_text': '[Engine not initialized]',
                'generation_successful': False,
                'error': 'Engine not initialized'
            }
        
        try:
            return self.engine.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                'generated_text': f'[Generation failed: {str(e)}]',
                'generation_successful': False,
                'error': str(e)
            }
    
    def select_model(self, text: str) -> Tuple[str, InputAnalysis, float]:
        """Select best model for input"""
        
        if not self.engine:
            return "unknown", InputAnalysis(), 0.0
        
        try:
            return self.engine.select_model(text)
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            # Return first available model as fallback
            if self.models:
                first_model = list(self.models.keys())[0]
                return first_model, InputAnalysis(), 0.5
            return "unknown", InputAnalysis(), 0.0
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        
        if not self.engine:
            return {'error': 'Engine not initialized'}
        
        try:
            stats = self.engine.get_model_stats()
            
            # Add safe engine info
            stats['safe_engine_info'] = {
                'real_models_enabled': self.enable_real_models,
                'engine_type': 'real' if any('real' in str(type(model.model)) for model in self.models.values()) else 'dummy',
                'total_models': len(self.models),
                'available_models': list(self.models.keys())
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}
    
    def evaluate_performance(self, test_cases: List[str] = None) -> Dict[str, Any]:
        """Evaluate engine performance"""
        
        if test_cases is None:
            test_cases = [
                "Calculate 5 + 3",
                "Write hello world",
                "def test():",
                "What is water?",
                "Hello there"
            ]
        
        results = {
            'total_tests': len(test_cases),
            'successful_generations': 0,
            'generation_times': [],
            'detailed_results': []
        }
        
        for i, prompt in enumerate(test_cases):
            start_time = time.time()
            
            try:
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
                    'time': time.time() - start_time
                })
        
        # Calculate summary stats
        if results['generation_times']:
            results['avg_generation_time'] = sum(results['generation_times']) / len(results['generation_times'])
            results['success_rate'] = results['successful_generations'] / results['total_tests']
        else:
            results['avg_generation_time'] = 0
            results['success_rate'] = 0
        
        return results
    
    def is_functional(self) -> bool:
        """Check if engine is functional"""
        
        try:
            test_result = self.generate("test", max_length=10)
            return test_result.get('generation_successful', False)
        except:
            return False

def create_safe_engine(enable_real_models: bool = True) -> SafeMultiModelEngine:
    """Factory function to create safe engine"""
    
    return SafeMultiModelEngine(enable_real_models=enable_real_models)

def test_safe_engine():
    """Test the safe engine"""
    
    print("🔒 Testing Safe Multi-Model Engine")
    print("=" * 40)
    
    # Test with real models
    print("🤖 Testing with real models enabled...")
    engine_real = create_safe_engine(enable_real_models=True)
    
    if engine_real.is_functional():
        print("✅ Real models engine functional")
        
        # Quick test
        result = engine_real.generate("Hello world", max_length=30)
        if result['generation_successful']:
            print(f"✅ Generation: {result['generated_text'][:50]}...")
        
        # Performance test
        perf_results = engine_real.evaluate_performance()
        print(f"✅ Performance: {perf_results['success_rate']:.2f} success rate")
        
    else:
        print("⚠️ Real models engine not functional")
    
    # Test with dummy models fallback
    print("\n🎭 Testing with dummy models...")
    engine_dummy = create_safe_engine(enable_real_models=False)
    
    if engine_dummy.is_functional():
        print("✅ Dummy models engine functional")
        
        # Quick test
        result = engine_dummy.generate("Hello world", max_length=30)
        if result['generation_successful']:
            print(f"✅ Generation: {result['generated_text'][:50]}...")
    else:
        print("❌ Even dummy models engine not functional")
    
    # Choose best engine
    if engine_real.is_functional():
        print("\n🎉 Using real models engine")
        return engine_real
    elif engine_dummy.is_functional():
        print("\n🔄 Using dummy models engine as fallback")
        return engine_dummy
    else:
        print("\n❌ No functional engine available")
        return None

if __name__ == "__main__":
    test_safe_engine()