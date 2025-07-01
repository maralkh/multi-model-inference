# File: examples/real_models_example.py
"""Example usage of real models with multi-model inference system"""

import sys
import os
import time
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.safe_real_models import (
    create_small_safe_models, create_medium_safe_models, create_large_safe_models,
    create_safe_real_tokenizer, ModelSize, SafeRealModelConfig, create_safe_real_models
)
from utils.dummy_models import create_specialized_models  # Fallback
from core.multi_model_engine import MultiModelInferenceEngine

def test_single_real_model():
    """Test a single real model"""
    print("🔧 Testing Single Real Model")
    print("=" * 40)
    
    try:
        from utils.safe_real_models import SafeRealModelWrapper, SafeRealModelConfig
        
        # Test with GPT-2 (smallest and most reliable)
        config = SafeRealModelConfig(
            model_size=ModelSize.SMALL,
            device="cpu",  # Use CPU for reliability
            torch_dtype=torch.float32
        )
        model = SafeRealModelWrapper("gpt2", config)
        
        print(f"✅ Model loaded: {model.model_name}")
        print(f"📊 Model info: {model.get_model_info()}")
        
        # Test generation
        test_prompts = [
            "Hello, how are you?",
            "Write a short story about AI:",
            "def fibonacci(n):"
        ]
        
        for prompt in test_prompts:
            print(f"\n🧪 Testing: '{prompt}'")
            result = model.generate(prompt, max_length=100, temperature=0.7)
            
            if result['generation_successful']:
                generated = result['generated_text']
                # Only show the new part (remove input prompt)
                new_text = generated[len(prompt):].strip()
                print(f"✅ Generated: {new_text[:100]}...")
            else:
                print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Single model test failed: {e}")
        return False

def test_real_model_creation():
    """Test creating different sizes of real models"""
    print("\n🏭 Testing Real Model Creation")
    print("=" * 40)
    
    results = {}
    
    # Test small models
    print("📱 Testing small models...")
    try:
        small_models = create_small_safe_models()
        results['small'] = {
            'success': True,
            'count': len(small_models),
            'models': [m.model_id for m in small_models]
        }
        print(f"✅ Created {len(small_models)} small models: {[m.model_id for m in small_models]}")
    except Exception as e:
        results['small'] = {'success': False, 'error': str(e)}
        print(f"❌ Small models failed: {e}")
    
    # Test medium models (only if small worked and we have GPU)
    if results['small']['success'] and torch.cuda.is_available():
        print("\n💻 Testing medium models...")
        try:
            medium_models = create_medium_safe_models()
            results['medium'] = {
                'success': True,
                'count': len(medium_models),
                'models': [m.model_id for m in medium_models]
            }
            print(f"✅ Created {len(medium_models)} medium models: {[m.model_id for m in medium_models]}")
        except Exception as e:
            results['medium'] = {'success': False, 'error': str(e)}
            print(f"⚠️ Medium models failed: {e}")
    else:
        print("⏭️ Skipping medium models (no GPU or small models failed)")
        results['medium'] = {'success': False, 'skipped': True}
    
    return results

def test_real_vs_dummy_comparison():
    """Compare real models vs dummy models"""
    print("\n⚖️ Comparing Real vs Dummy Models")
    print("=" * 40)
    
    test_prompt = "Solve the equation x² + 2x - 3 = 0"
    
    # Test with dummy models
    print("🎭 Testing with dummy models...")
    try:
        dummy_models = create_specialized_models()
        dummy_tokenizer = None  # Use dummy tokenizer
        
        from utils.dummy_models import DummyTokenizer
        dummy_tokenizer = DummyTokenizer()
        
        dummy_engine = MultiModelInferenceEngine(dummy_models, dummy_tokenizer)
        dummy_result = dummy_engine.generate(test_prompt, max_length=100)
        
        print(f"✅ Dummy result: {dummy_result['selected_model']}")
        print(f"📝 Dummy text: {dummy_result['generated_text'][:80]}...")
        
    except Exception as e:
        print(f"❌ Dummy models failed: {e}")
        dummy_result = None
    
    # Test with real models
    print("\n🔧 Testing with real models...")
    try:
        real_models = create_small_safe_models()
        real_tokenizer = create_safe_real_tokenizer(ModelSize.SMALL)
        
        real_engine = MultiModelInferenceEngine(real_models, real_tokenizer)
        real_result = real_engine.generate(test_prompt, max_length=100)
        
        print(f"✅ Real result: {real_result['selected_model']}")
        print(f"📝 Real text: {real_result['generated_text'][:80]}...")
        
    except Exception as e:
        print(f"❌ Real models failed: {e}")
        real_result = None
    
    # Compare results
    if dummy_result and real_result:
        print(f"\n📊 Comparison:")
        print(f"   Dummy model selected: {dummy_result['selected_model']}")
        print(f"   Real model selected: {real_result['selected_model']}")
        print(f"   Same selection: {'✅' if dummy_result['selected_model'] == real_result['selected_model'] else '❌'}")
        print(f"   Dummy time: {dummy_result.get('generation_time', 0):.3f}s")
        print(f"   Real time: {real_result.get('generation_time', 0):.3f}s")

def test_model_performance():
    """Test model performance with different tasks"""
    print("\n🏃 Testing Model Performance")
    print("=" * 40)
    
    test_cases = [
        {
            'prompt': "Calculate the derivative of x³ + 2x² - 5x + 1",
            'expected_model': 'math_specialist',
            'task_type': 'mathematical'
        },
        {
            'prompt': "Write a creative story about a robot learning to paint",
            'expected_model': 'creative_specialist', 
            'task_type': 'creative'
        },
        {
            'prompt': "def quicksort(arr):",
            'expected_model': 'code_specialist',
            'task_type': 'coding'
        },
        {
            'prompt': "Analyze the pros and cons of renewable energy",
            'expected_model': 'reasoning_specialist',
            'task_type': 'reasoning'
        }
    ]
    
    try:
        # Use small models for testing
        models = create_small_safe_models()
        tokenizer = create_safe_real_tokenizer(ModelSize.SMALL)
        engine = MultiModelInferenceEngine(models, tokenizer)
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}] Testing {test_case['task_type']} task...")
            print(f"Prompt: {test_case['prompt'][:50]}...")
            
            start_time = time.time()
            result = engine.generate(
                test_case['prompt'], 
                max_length=150, 
                temperature=0.7
            )
            end_time = time.time()
            
            selected_model = result['selected_model']
            expected_model = test_case['expected_model']
            correct_selection = selected_model == expected_model
            
            print(f"Expected: {expected_model}")
            print(f"Selected: {selected_model}")
            print(f"Correct: {'✅' if correct_selection else '❌'}")
            print(f"Time: {end_time - start_time:.3f}s")
            
            if result.get('generation_successful', False):
                generated_preview = result['generated_text'][:100] + "..."
                print(f"Generated: {generated_preview}")
            else:
                print(f"❌ Generation failed")
            
            results.append({
                'task_type': test_case['task_type'],
                'correct_selection': correct_selection,
                'generation_time': end_time - start_time,
                'successful': result.get('generation_successful', False)
            })
        
        # Summary
        print(f"\n📈 Performance Summary:")
        correct_count = sum(1 for r in results if r['correct_selection'])
        success_count = sum(1 for r in results if r['successful'])
        avg_time = sum(r['generation_time'] for r in results) / len(results)
        
        print(f"   Selection Accuracy: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
        print(f"   Generation Success: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        print(f"   Average Time: {avg_time:.3f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return []

def main():
    """Main example function"""
    print("🚀 Safe Real Models Example")
    print("=" * 50)
    
    # Check system requirements
    print("🔍 System Check:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Device: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"   Recommended: Use CPU mode for stability")
    print()
    
    # Run tests step by step
    tests_results = {}
    
    # Test 1: Single model
    tests_results['single_model'] = test_single_real_model()
    
    # Test 2: Model creation
    if tests_results['single_model']:
        tests_results['model_creation'] = test_real_model_creation()
    else:
        print("⏭️ Skipping model creation test (single model failed)")
        tests_results['model_creation'] = {'skipped': True}
    
    # Test 3: Comparison
    if tests_results['single_model']:
        test_real_vs_dummy_comparison()
    else:
        print("⏭️ Skipping comparison test (real models not working)")
    
    # Test 4: Performance
    if tests_results['single_model']:
        performance_results = test_model_performance()
        tests_results['performance'] = performance_results
    else:
        print("⏭️ Skipping performance test (real models not working)")
        tests_results['performance'] = []
    
    # Final summary
    print(f"\n🎉 Example Complete!")
    print("=" * 30)
    print("Tests completed:")
    for test_name, result in tests_results.items():
        if isinstance(result, bool):
            status = "✅ PASSED" if result else "❌ FAILED"
        elif isinstance(result, dict) and result.get('skipped'):
            status = "⏭️ SKIPPED"
        elif isinstance(result, list):
            status = f"✅ COMPLETED ({len(result)} cases)"
        else:
            status = "✅ COMPLETED"
        print(f"   {test_name}: {status}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if not tests_results['single_model']:
        print("   • Install transformers: pip install transformers torch")
        print("   • Check internet connection for model downloads")
        print("   • Try CPU mode first: device='cpu'")
        print("   • Run: python test_safe_models.py")
    else:
        print("   • Real models are working! 🎉")
        print("   • Try medium/large models if you have GPU")
        print("   • Use safe_real_models for stable performance")
        print("   • GPT-2 models are reliable and fast")

if __name__ == "__main__":
    main()