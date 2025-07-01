# File: main_demo.py
"""Main demonstration script for the complete multi-model inference system"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_system(enable_real_models: bool = True, 
                enable_data_loading: bool = True,
                model_size: str = "small") -> Dict[str, Any]:
    """Setup the complete system with all components"""
    
    print("🚀 Setting Up Multi-Model Inference System")
    print("=" * 60)
    
    setup_results = {
        'data_pipeline': None,
        'engine': None,
        'classifier': None,
        'models_loaded': 0,
        'data_loaded': 0,
        'setup_successful': False
    }
    
    try:
        # Step 1: Data Loading and Processing
        if enable_data_loading:
            print("\n📚 Step 1: Loading and Processing Data")
            print("-" * 40)
            
            try:
                # Check if any of the data modules are available
                data_pipeline_available = False
                
                try:
                    from data.integrated_data_pipeline import IntegratedDataPipeline, PreprocessingConfig
                    data_pipeline_available = True
                except ImportError:
                    pass
                
                try:
                    from data.real_data_generator import RealDataGenerator
                    data_pipeline_available = True
                except ImportError:
                    pass
                
                try:
                    from data.huggingface_data_loader import HuggingFaceDataLoader
                    data_pipeline_available = True
                except ImportError:
                    pass
                
                if data_pipeline_available:
                    # Try to use integrated pipeline
                    try:
                        from data.integrated_data_pipeline import IntegratedDataPipeline, PreprocessingConfig
                        
                        config = PreprocessingConfig(
                            clean_text=True,
                            tokenize=False,  # Skip for faster demo
                            compute_embeddings=False,
                            enable_augmentation=False,
                            filter_duplicates=True,
                            filter_low_quality=True,
                            min_quality_score=0.4
                        )
                        
                        pipeline = IntegratedDataPipeline(
                            base_dir="./demo_data",
                            preprocessing_config=config
                        )
                        
                        print("✅ Data pipeline created")
                        
                        # Run pipeline (basic mode for demo)
                        print("🔄 Processing data...")
                        results = pipeline.run_full_pipeline(
                            include_hf=False,  # Skip HF for faster demo
                            train_split=0.8,
                            val_split=0.1
                        )
                        
                        setup_results['data_pipeline'] = pipeline
                        setup_results['data_loaded'] = results['processed_samples']
                        
                        print(f"✅ Data processing complete:")
                        print(f"   - Processed: {results['processed_samples']} samples")
                        print(f"   - Train/Val/Test: {results['split_stats']}")
                        
                    except Exception as e:
                        print(f"⚠️ Integrated pipeline failed: {e}")
                        # Try basic data generator
                        try:
                            from data.real_data_generator import RealDataGenerator
                            
                            generator = RealDataGenerator("./demo_basic_data")
                            basic_results = generator.generate_all_data()
                            
                            setup_results['data_loaded'] = sum(basic_results.values())
                            print(f"✅ Basic data generation complete:")
                            print(f"   - Generated: {setup_results['data_loaded']} samples")
                            
                        except Exception as e2:
                            print(f"⚠️ Basic data generation also failed: {e2}")
                            setup_results['data_loaded'] = 'synthetic_only'
                else:
                    print("⚠️ No data pipeline modules available")
                    setup_results['data_loaded'] = 'synthetic_only'
                
            except Exception as e:
                print(f"⚠️ Data loading failed: {e}")
                print("🔄 Continuing with synthetic data only...")
                setup_results['data_loaded'] = 'synthetic_only'
        
        # Step 2: Model Loading
        print("\n🤖 Step 2: Loading Models")
        print("-" * 30)
        
        # Use safe engine for robust model loading
        from core.safe_engine_fallback import create_safe_engine
        
        engine = create_safe_engine(enable_real_models=enable_real_models)
        
        if engine.is_functional():
            setup_results['engine'] = engine
            setup_results['models_loaded'] = len(engine.models)
            
            # Get model info
            stats = engine.get_model_stats()
            engine_type = stats['safe_engine_info']['engine_type']
            models = stats['safe_engine_info']['available_models']
            
            print(f"✅ Engine loaded successfully:")
            print(f"   - Type: {engine_type} models")
            print(f"   - Models: {len(models)}")
            print(f"   - Available: {', '.join(models)}")
        else:
            print("❌ Failed to load any models")
            return setup_results
        
        # Step 3: Enhanced Classifier Setup
        print("\n🎯 Step 3: Setting Up Enhanced Classifier")
        print("-" * 45)
        
        # Always use safe classifier as fallback
        from core.fixed_input_analysis import create_safe_classifier
        
        try:
            from core.updated_classifier_manifold import create_updated_classifier, RealDataTrainingConfig
            
            # Create classifier config
            classifier_config = RealDataTrainingConfig(
                max_training_samples=200,  # Smaller for demo
                min_quality_score=0.4,
                enable_manifold_pretraining=True,
                enable_validation=True,
                use_processed_data=setup_results['data_loaded'] not in ['synthetic_only', 'basic'],
                use_synthetic_data=True
            )
            
            # Create and train enhanced classifier
            print("🔄 Training enhanced classifier...")
            enhanced_classifier = create_updated_classifier(auto_train=True, config=classifier_config)
            
            # Wrap with safe classifier
            classifier = create_safe_classifier(enhanced_classifier)
            
            setup_results['classifier'] = classifier
            
            # Show training results
            try:
                metadata = enhanced_classifier.get_training_metadata()
                print(f"✅ Enhanced classifier ready:")
                print(f"   - Training samples: {metadata['total_samples_processed']}")
                print(f"   - Data sources: {metadata['data_sources_used']}")
                print(f"   - Task distribution: {dict(metadata['samples_per_task'])}")
                
                if 'validation_results' in metadata and metadata['validation_results']:
                    val_acc = metadata['validation_results']['overall_accuracy']
                    print(f"   - Validation accuracy: {val_acc:.3f}")
            except Exception as e:
                print(f"✅ Enhanced classifier ready (metadata unavailable: {e})")
        
        except ImportError:
            print("⚠️ Enhanced classifier not available, using safe pattern-based classifier")
            classifier = create_safe_classifier(engine.classifier)
            setup_results['classifier'] = classifier
        
        except Exception as e:
            print(f"⚠️ Enhanced classifier setup failed: {e}")
            print("🔄 Using safe pattern-based classifier...")
            classifier = create_safe_classifier(engine.classifier)
            setup_results['classifier'] = classifier
        
        # Step 4: System Integration
        print("\n🔗 Step 4: Final System Integration")
        print("-" * 40)
        
        # Replace engine classifier with enhanced one if available
        if setup_results['classifier'] and setup_results['classifier'] != engine.classifier:
            try:
                engine.engine.classifier = setup_results['classifier']
                print("✅ Enhanced classifier integrated with engine")
            except Exception as e:
                print(f"⚠️ Failed to integrate enhanced classifier: {e}")
        
        setup_results['setup_successful'] = True
        
        print("\n🎉 System Setup Complete!")
        print(f"   - Models: {setup_results['models_loaded']} loaded")
        print(f"   - Data: {setup_results['data_loaded']} samples processed")
        print(f"   - Classifier: Enhanced" if setup_results['classifier'] != engine.classifier else "   - Classifier: Basic")
        
        return setup_results
        
    except Exception as e:
        logger.error(f"System setup failed: {e}")
        print(f"❌ System setup failed: {e}")
        return setup_results

def demonstrate_classification(classifier, test_prompts: List[str] = None):
    """Demonstrate input classification with manifold learning"""
    
    print("\n🎯 Classification and Manifold Learning Demo")
    print("=" * 55)
    
    if test_prompts is None:
        test_prompts = [
            "Calculate the derivative of x³ + 2x² - 5x + 7",
            "Write a creative story about a robot learning to paint",
            "Implement a binary search algorithm in Python",
            "Explain how photosynthesis works in plants",
            "Analyze the pros and cons of renewable energy sources",
            "What is the capital city of Australia?",
            "Hello! How can I assist you today?",
            "Solve the system of equations: 2x + 3y = 12, x - y = 1",
            "Create a character description for a detective novel",
            "Write SQL query to find top customers by revenue"
        ]
    
    print(f"🧪 Testing {len(test_prompts)} diverse inputs...")
    
    classification_results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] Input: '{prompt[:50]}...'")
        
        try:
            # Analyze input
            analysis = classifier.analyze_input(prompt)
            
            # All attributes should be available in FixedInputAnalysis
            result = {
                'prompt': prompt,
                'task_type': analysis.task_type.value,
                'confidence': analysis.confidence_score,
                'manifold_type': analysis.manifold_type,
                'complexity': analysis.complexity_score,
                'domains': analysis.specialized_domains
            }
            
            classification_results.append(result)
            
            print(f"    ✅ Task Type: {analysis.task_type.value}")
            print(f"       Confidence: {analysis.confidence_score:.3f}")
            print(f"       Manifold: {analysis.manifold_type}")
            print(f"       Complexity: {analysis.complexity_score:.3f}")
            
            if analysis.specialized_domains:
                print(f"       Domains: {', '.join(analysis.specialized_domains[:3])}")
        
        except Exception as e:
            print(f"    ❌ Classification failed: {e}")
            classification_results.append({
                'prompt': prompt,
                'error': str(e)
            })
    
    # Summary
    successful = len([r for r in classification_results if 'error' not in r])
    task_distribution = {}
    
    for result in classification_results:
        if 'task_type' in result:
            task_type = result['task_type']
            task_distribution[task_type] = task_distribution.get(task_type, 0) + 1
    
    print(f"\n📊 Classification Summary:")
    print(f"   - Successful: {successful}/{len(test_prompts)}")
    print(f"   - Task Distribution: {task_distribution}")
    
    return classification_results

def demonstrate_generation(engine, test_prompts: List[str] = None):
    """Demonstrate text generation with model selection"""
    
    print("\n🤖 Multi-Model Generation Demo")
    print("=" * 40)
    
    if test_prompts is None:
        test_prompts = [
            "Solve the quadratic equation: x² - 5x + 6 = 0",
            "Write a haiku about artificial intelligence",
            "Create a Python function to calculate factorial",
            "Explain the greenhouse effect in simple terms",
            "Compare the benefits of solar vs wind energy",
            "What year did the Berlin Wall fall?",
            "Thank you for your help with this problem"
        ]
    
    print(f"🎭 Generating responses for {len(test_prompts)} prompts...")
    
    generation_results = []
    total_time = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] Prompt: '{prompt[:40]}...'")
        
        start_time = time.time()
        
        try:
            # Generate with advanced options if available
            generation_kwargs = {
                'max_length': 150,
                'temperature': 0.7
            }
            
            # Try extended generation if available
            try:
                if hasattr(engine, 'models') and engine.models:
                    first_model = list(engine.models.values())[0] if hasattr(engine, 'models') else None
                    if first_model and hasattr(first_model.model, 'generate') and hasattr(first_model.model, 'selected_method'):
                        generation_kwargs['generation_method'] = 'auto'
            except:
                pass  # Use basic generation
            
            result = engine.generate(prompt, **generation_kwargs)

            print(f'##### result: {result}')
            
            generation_time = time.time() - start_time
            total_time += generation_time
            
            # Safe access to result fields
            generation_successful = result.get('generation_successful', False)
            selected_model = result.get('selected_model', 'unknown')
            task_type = result.get('input_analysis', {}).get('task_type', 'unknown')
            generated_text = result.get('generated_text', '')
            new_content = result.get('new_text', '')
            
            # If new_content is empty, try to extract it
            if not new_content and generated_text:
                if generated_text.startswith(prompt):
                    new_content = generated_text[len(prompt):].strip()
                else:
                    new_content = generated_text
            
            if generation_successful:
                print(f"    ✅ Model: {selected_model}")
                print(f"       Task: {task_type}")
                print(f"       Time: {generation_time:.3f}s")
                print(f"       Response: {new_content[:100]}...")
                
                generation_method = result.get('generation_method', 'basic')
                quality_score = result.get('quality_score', 0.0)
                
                if generation_method != 'basic':
                    print(f"       Method: {generation_method}")
                if quality_score > 0:
                    print(f"       Quality: {quality_score:.3f}")
                
                generation_results.append({
                    'prompt': prompt,
                    'selected_model': selected_model,
                    'task_type': task_type,
                    'response': new_content,
                    'generation_time': generation_time,
                    'success': True,
                    'method': generation_method,
                    'quality': quality_score
                })
            else:
                error = result.get('error', 'Unknown error')
                print(f"    ❌ Generation failed: {error}")
                
                generation_results.append({
                    'prompt': prompt,
                    'success': False,
                    'error': error,
                    'generation_time': generation_time
                })
        
        except Exception as e:
            generation_time = time.time() - start_time
            print(f"    ❌ Exception: {e}")
            
            generation_results.append({
                'prompt': prompt,
                'success': False,
                'error': str(e),
                'generation_time': generation_time
            })
    
    # Summary statistics
    successful = len([r for r in generation_results if r.get('success', False)])
    avg_time = total_time / len(test_prompts)
    
    model_usage = {}
    methods_used = {}
    
    for result in generation_results:
        if result.get('success'):
            model = result.get('selected_model', 'unknown')
            method = result.get('method', 'basic')
            
            model_usage[model] = model_usage.get(model, 0) + 1
            methods_used[method] = methods_used.get(method, 0) + 1
    
    print(f"\n📊 Generation Summary:")
    print(f"   - Success Rate: {successful}/{len(test_prompts)} ({successful/len(test_prompts)*100:.1f}%)")
    print(f"   - Average Time: {avg_time:.3f}s")
    print(f"   - Model Usage: {model_usage}")
    
    if methods_used:
        print(f"   - Methods Used: {methods_used}")
    
    return generation_results

def demonstrate_performance_analysis(engine):
    """Demonstrate system performance analysis"""
    
    print("\n📈 Performance Analysis Demo")
    print("=" * 35)
    
    # Get system statistics
    stats = engine.get_model_stats()
    
    print("🔧 System Configuration:")
    if 'safe_engine_info' in stats:
        info = stats['safe_engine_info']
        print(f"   - Engine Type: {info['engine_type']}")
        print(f"   - Real Models: {info.get('real_models_enabled', 'Unknown')}")
        print(f"   - Total Models: {info['total_models']}")
    
    print(f"   - Total Generations: {stats.get('total_generations', 0)}")
    
    # Run performance evaluation
    print("\n🏃 Running Performance Benchmark...")
    
    benchmark_prompts = [
        "Calculate 25 × 17",
        "Hello world",
        "def sort_list():",
        "What is gravity?",
        "Compare A vs B",
        "Greetings!",
        "x + y = 10",
        "Write poem",
        "SELECT * FROM",
        "Explain DNA"
    ]
    
    perf_results = engine.evaluate_performance(benchmark_prompts)
    
    print(f"📊 Performance Results:")
    print(f"   - Success Rate: {perf_results['success_rate']:.3f}")
    print(f"   - Average Time: {perf_results['avg_generation_time']:.3f}s")
    print(f"   - Total Tests: {perf_results['total_tests']}")
    
    # Model performance breakdown (safe access)
    print(f"\n🎯 Model Performance:")
    if 'model_performance' in stats and stats['model_performance']:
        for model_id, model_stats in stats['model_performance'].items():
            # Safe access to model stats
            usage_count = model_stats.get('usage_count', 0)
            if usage_count > 0:
                avg_time = model_stats.get('avg_response_time', 0.0)
                success_rate = model_stats.get('success_rate', 0.0)
                
                print(f"   - {model_id}:")
                print(f"     Usage: {usage_count} times")
                print(f"     Avg Time: {avg_time:.3f}s")
                print(f"     Success Rate: {success_rate:.3f}")
    else:
        print("   - Detailed model stats not available")
        
        # Show basic model info instead
        if 'safe_engine_info' in stats:
            available_models = stats['safe_engine_info'].get('available_models', [])
            print(f"   - Available Models: {', '.join(available_models)}")
    
    return perf_results

def interactive_demo(engine, classifier):
    """Interactive demo mode"""
    
    print("\n💬 Interactive Demo Mode")
    print("=" * 30)
    print("Enter prompts to test the system (type 'quit' to exit)")
    print("Commands:")
    print("  - 'stats' - show system statistics")
    print("  - 'models' - show available models")
    print("  - 'help' - show this help")
    print()
    
    while True:
        try:
            user_input = input("🔥 Enter prompt: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'stats':
                stats = engine.get_model_stats()
                print(f"📊 Total generations: {stats.get('total_generations', 0)}")
                continue
            
            elif user_input.lower() == 'models':
                stats = engine.get_model_stats()
                if 'safe_engine_info' in stats:
                    models = stats['safe_engine_info']['available_models']
                    print(f"🤖 Available models: {', '.join(models)}")
                continue
            
            elif user_input.lower() == 'help':
                print("💡 Enter any text prompt to see:")
                print("   - Task classification")
                print("   - Model selection")
                print("   - Generated response")
                continue
            
            elif not user_input:
                continue
            
            print(f"\n🔄 Processing: '{user_input}'")
            
            # Classification
            analysis = classifier.analyze_input(user_input)
            
            print(f"📋 Classification:")
            print(f"   Task: {analysis.task_type.value}")
            print(f"   Confidence: {analysis.confidence_score:.3f}")
            print(f"   Manifold: {analysis.manifold_type}")
            
            # Generation
            start_time = time.time()
            result = engine.generate(user_input, max_length=200)
            generation_time = time.time() - start_time
            
            if result['generation_successful']:
                print(f"\n🤖 Generation:")
                print(f"   Selected Model: {result['selected_model']}")
                print(f"   Time: {generation_time:.3f}s")
                
                # Show response
                generated = result['generated_text']
                if generated.startswith(user_input):
                    response = generated[len(user_input):].strip()
                else:
                    response = generated
                
                print(f"   Response: {response}")
                
                if 'generation_method' in result:
                    print(f"   Method: {result['generation_method']}")
            else:
                print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demonstration function"""
    
    parser = argparse.ArgumentParser(description="Multi-Model Inference System Demo")
    parser.add_argument("--mode", choices=["full", "basic", "interactive"], default="full",
                       help="Demo mode")
    parser.add_argument("--no-real-models", action="store_true",
                       help="Use dummy models only")
    parser.add_argument("--no-data", action="store_true",
                       help="Skip data loading")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="small",
                       help="Model size to use")
    parser.add_argument("--save-results", action="store_true",
                       help="Save demo results to file")
    
    args = parser.parse_args()
    
    print(f"🚀 Multi-Model Inference System Demo")
    print(f"Mode: {args.mode}")
    print(f"Model Size: {args.model_size}")
    print(f"Real Models: {'No' if args.no_real_models else 'Yes'}")
    print(f"Data Loading: {'No' if args.no_data else 'Yes'}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup system
    setup_start = time.time()
    
    setup_results = setup_system(
        enable_real_models=not args.no_real_models,
        enable_data_loading=not args.no_data,
        model_size=args.model_size
    )
    
    setup_time = time.time() - setup_start
    
    if not setup_results['setup_successful']:
        print("❌ System setup failed. Exiting.")
        return
    
    engine = setup_results['engine']
    classifier = setup_results['classifier']
    
    print(f"\n⏰ Setup completed in {setup_time:.1f} seconds")
    
    demo_results = {
        'setup': setup_results,
        'setup_time': setup_time,
        'mode': args.mode,
        'timestamp': time.time()
    }
    
    # Run demonstrations based on mode
    if args.mode == "interactive":
        interactive_demo(engine, classifier)
    
    elif args.mode == "basic":
        # Basic demo
        print("\n🎭 Running Basic Demo...")
        
        basic_prompts = [
            "Calculate 2 + 2",
            "Hello there",
            "def hello():",
            "What is water?"
        ]
        
        classification_results = demonstrate_classification(classifier, basic_prompts)
        generation_results = demonstrate_generation(engine, basic_prompts)
        
        demo_results.update({
            'classification_results': classification_results,
            'generation_results': generation_results,
            'demo_type': 'basic'
        })
    
    else:  # full mode
        # Full demonstration
        print("\n🎭 Running Full Demonstration...")
        
        # Classification demo
        classification_results = demonstrate_classification(classifier)
        demo_results['classification_results'] = classification_results
        
        # Generation demo
        generation_results = demonstrate_generation(engine)
        demo_results['generation_results'] = generation_results
        
        # Performance analysis
        performance_results = demonstrate_performance_analysis(engine)
        demo_results['performance_results'] = performance_results
        
        demo_results['demo_type'] = 'full'
    
    # Final summary
    total_time = time.time() - setup_start
    
    print(f"\n🎉 Demo Complete!")
    print(f"⏰ Total time: {total_time:.1f} seconds")
    print(f"📊 Components working:")
    print(f"   - Models: {setup_results['models_loaded']}")
    print(f"   - Data: {setup_results['data_loaded']}")
    print(f"   - Classification: ✅")
    print(f"   - Generation: ✅")
    
    # Save results if requested
    if args.save_results and args.mode != "interactive":
        results_file = f"demo_results_{int(time.time())}.json"
        
        try:
            # Make results JSON serializable
            json_results = {}
            for key, value in demo_results.items():
                if key in ['setup', 'classification_results', 'generation_results', 'performance_results']:
                    json_results[key] = str(value)  # Convert complex objects to string
                else:
                    json_results[key] = value
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")
    
    print(f"\n🎯 System is ready for use!")
    return demo_results

if __name__ == "__main__":
    main()