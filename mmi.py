# File: main.py
"""Main demonstration script for the multi-model inference system"""

import torch
import numpy as np
import time
import json
from pathlib import Path

from core.multi_model_engine import MultiModelInferenceEngine
from core.input_types import TaskType
from utils.dummy_models import create_specialized_models, DummyTokenizer

def demonstrate_multi_model_inference():
    """Demonstrate multi-model inference with automatic selection"""
    
    print("🤖 Multi-Model Inference Demonstration")
    print("=" * 60)
    
    # Create models and tokenizer
    models = create_specialized_models()
    tokenizer = DummyTokenizer()
    
    # Create multi-model engine
    engine = MultiModelInferenceEngine(
        models=models,
        tokenizer=tokenizer,
        default_model_id="general_model"
    )
    
    print(f"✅ Created {len(models)} specialized models:")
    for model in models:
        print(f"   • {model.model_id}: {model.description}")
    
    # Test prompts of different types
    test_prompts = [
        {
            'text': "Solve the quadratic equation: x² - 5x + 6 = 0",
            'expected_model': 'math_specialist'
        },
        {
            'text': "Write a creative short story about a time-traveling detective",
            'expected_model': 'creative_specialist'
        },
        {
            'text': "Analyze the pros and cons of renewable energy adoption",
            'expected_model': 'reasoning_specialist'
        },
        {
            'text': "Write a Python function to implement binary search",
            'expected_model': 'code_specialist'
        },
        {
            'text': "What is the capital of France?",
            'expected_model': 'reasoning_specialist'
        },
        {
            'text': "Explain the process of photosynthesis in plants",
            'expected_model': 'reasoning_specialist'
        },
        {
            'text': "Debug this code: def slow_function(n): result = 0; for i in range(n): for j in range(n): result += i * j; return result",
            'expected_model': 'code_specialist'
        }
    ]
    
    results = []
    
    print(f"\n🧪 Testing {len(test_prompts)} different prompts")
    print("=" * 60)
    
    for i, prompt_data in enumerate(test_prompts, 1):
        prompt = prompt_data['text']
        expected = prompt_data['expected_model']
        
        print(f"\n[{i}/{len(test_prompts)}] Testing prompt:")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"Expected model: {expected}")
        
        # Generate response
        try:
            result = engine.generate(
                prompt=prompt,
                max_length=150,
                temperature=0.7
            )
            
            selected_model = result['selected_model']
            confidence = result['model_selection_confidence']
            analysis = result['input_analysis']
            
            print(f"Selected model: {selected_model}")
            print(f"Selection confidence: {confidence:.3f}")
            print(f"Task type detected: {analysis['task_type']}")
            print(f"Complexity score: {analysis['complexity_score']:.3f}")
            print(f"Generation time: {result['generation_time']:.3f}s")
            
            # Check if selection was correct
            correct_selection = selected_model == expected
            print(f"Selection accuracy: {'✅ Correct' if correct_selection else '❌ Incorrect'}")
            
            if not correct_selection:
                print(f"   Expected: {expected}, Got: {selected_model}")
            
            results.append({
                'prompt': prompt,
                'expected_model': expected,
                'selected_model': selected_model,
                'correct': correct_selection,
                'confidence': confidence,
                'analysis': analysis,
                'result': result
            })
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            results.append({
                'prompt': prompt,
                'expected_model': expected,
                'selected_model': None,
                'correct': False,
                'error': str(e)
            })
    
    return engine, results

def analyze_results(results):
    """Analyze the results of the demonstration"""
    
    print("\n📊 Results Analysis")
    print("=" * 40)
    
    successful_results = [r for r in results if r.get('selected_model')]
    
    if not successful_results:
        print("❌ No successful results to analyze")
        return {}
    
    # Calculate accuracy
    correct_selections = sum(1 for r in successful_results if r['correct'])
    accuracy = correct_selections / len(successful_results)
    
    print(f"Selection Accuracy: {accuracy:.2%} ({correct_selections}/{len(successful_results)})")
    
    # Analyze by task type
    task_type_performance = {}
    for result in successful_results:
        task_type = result['analysis']['task_type']
        if task_type not in task_type_performance:
            task_type_performance[task_type] = {'correct': 0, 'total': 0}
        
        task_type_performance[task_type]['total'] += 1
        if result['correct']:
            task_type_performance[task_type]['correct'] += 1
    
    print(f"\nPerformance by Task Type:")
    for task_type, stats in task_type_performance.items():
        task_accuracy = stats['correct'] / stats['total']
        print(f"  {task_type}: {task_accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    # Average confidence analysis
    avg_confidence = np.mean([r['confidence'] for r in successful_results])
    print(f"\nAverage Confidence: {avg_confidence:.3f}")
    
    # Model usage distribution
    model_usage = {}
    for result in successful_results:
        model = result['selected_model']
        model_usage[model] = model_usage.get(model, 0) + 1
    
    print(f"\nModel Usage Distribution:")
    for model, count in sorted(model_usage.items()):
        percentage = (count / len(successful_results)) * 100
        print(f"  {model}: {count} times ({percentage:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'task_type_performance': task_type_performance,
        'avg_confidence': avg_confidence,
        'model_usage': model_usage,
        'total_tests': len(successful_results)
    }

def save_results(engine, results, analysis):
    """Save results to files"""
    
    print("\n💾 Saving Results")
    print("=" * 25)
    
    # Get engine statistics
    stats = engine.get_model_stats()
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save comprehensive results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"multi_model_results_{timestamp}.json"
    
    output_data = {
        'timestamp': timestamp,
        'results': results,
        'analysis': analysis,
        'engine_stats': stats,
        'model_specs': {
            model_id: {
                'description': spec.description,
                'task_types': [tt.value for tt in spec.task_types],
                'domains': spec.specialized_domains,
                'metrics': spec.performance_metrics
            }
            for model_id, spec in engine.models.items()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"📁 Results saved to: {results_file}")
    
    # Create summary report
    summary_file = results_dir / f"summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("Multi-Model Inference System - Experiment Summary\n")
        f.write("=" * 55 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Tests: {len(results)}\n")
        f.write(f"Selection Accuracy: {analysis.get('accuracy', 0):.2%}\n")
        f.write(f"Average Confidence: {analysis.get('avg_confidence', 0):.3f}\n\n")
        
        f.write("Model Usage:\n")
        for model, count in analysis.get('model_usage', {}).items():
            percentage = (count / analysis.get('total_tests', 1)) * 100
            f.write(f"  {model}: {count} ({percentage:.1f}%)\n")
        
        f.write(f"\nTask Type Performance:\n")
        for task, perf in analysis.get('task_type_performance', {}).items():
            accuracy = perf['correct'] / perf['total']
            f.write(f"  {task}: {accuracy:.2%}\n")
    
    print(f"📄 Summary saved to: {summary_file}")
    
    return results_file, summary_file

def main():
    """Main demonstration function"""
    
    print("🚀 Multi-Model Reward-Guided Inference System")
    print("=" * 60)
    print("Features demonstrated:")
    print("• Automatic model selection based on input analysis")
    print("• Task-specific model routing")
    print("• Performance monitoring and statistics")
    print("• Comprehensive result analysis")
    print("=" * 60)
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run demonstration
        print("\n🎯 Running Multi-Model Inference Demonstration")
        engine, results = demonstrate_multi_model_inference()
        
        # Analyze results
        print("\n📊 Analyzing Results")
        analysis = analyze_results(results)
        
        # Save results
        print("\n💾 Saving Results")
        results_file, summary_file = save_results(engine, results, analysis)
        
        # Final summary
        print("\n🎉 Demonstration Complete!")
        print("=" * 40)
        print(f"✅ Tested {len(engine.models)} specialized models")
        print(f"✅ Evaluated {len(results)} different prompts")
        print(f"✅ Achieved {analysis.get('accuracy', 0):.2%} selection accuracy")
        print(f"✅ Saved comprehensive results")
        
        # Best performing aspects
        model_usage = analysis.get('model_usage', {})
        if model_usage:
            most_used = max(model_usage.keys(), key=lambda k: model_usage[k])
            print(f"\n🏆 Most utilized model: {most_used}")
            usage_count = model_usage[most_used]
            usage_percent = (usage_count / len(results)) * 100
            print(f"   Usage: {usage_count} times ({usage_percent:.1f}%)")
        
        # Recommendations
        print(f"\n💡 System Performance:")
        if analysis.get('accuracy', 0) > 0.8:
            print("   ✅ Excellent model selection accuracy")
        elif analysis.get('accuracy', 0) > 0.6:
            print("   ⚠️ Good model selection, room for improvement")
        else:
            print("   ❌ Model selection needs optimization")
        
        if analysis.get('avg_confidence', 0) > 0.7:
            print("   ✅ High confidence in selections")
        else:
            print("   ⚠️ Consider adjusting selection thresholds")
        
        print(f"\n📁 Files generated:")
        print(f"   • Results: {results_file}")
        print(f"   • Summary: {summary_file}")
        
        return engine, results, analysis
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def quick_test():
    """Quick test function for development"""
    
    print("🔧 Quick Test Mode")
    print("=" * 30)
    
    # Create simple engine
    models = create_specialized_models()
    tokenizer = DummyTokenizer()
    engine = MultiModelInferenceEngine(models, tokenizer)
    
    # Test a few prompts
    test_prompts = [
        "Calculate 2 + 2",
        "Write a haiku",
        "Sort an array in Python"
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting: {prompt}")
        result = engine.generate(prompt)
        print(f"Selected: {result['selected_model']}")
        print(f"Task: {result['input_analysis']['task_type']}")
    
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()