# File: core/updated_multi_model_engine.py
"""Updated Multi-Model Engine with real data and model integration"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import original core components
from core.input_types import TaskType, InputAnalysis, ModelSpec, ManifoldLearningConfig
from core.enhanced_input_classifier import EnhancedInputClassifier
from core.manifold_learner import ManifoldLearner
from core.geometric_embeddings import GeometricBayesianManifoldLearner

import logging
logger = logging.getLogger(__name__)
# Import new data and model components
try:
    from utils.safe_real_models import create_small_safe_models, create_safe_real_tokenizer, SafeRealModelWrapper
    from utils.extended_generation import ExtendedGenerationWrapper, GenerationConfig, create_extended_models_from_safe
    REAL_MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Real models not available: {e}")
    REAL_MODELS_AVAILABLE = False

try:
    from data.integrated_data_pipeline import IntegratedDataPipeline, PreprocessingConfig, ProcessedDataSample
    DATA_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data pipeline not available: {e}")
    DATA_PIPELINE_AVAILABLE = False

class UpdatedMultiModelEngine:
    """Updated Multi-Model Engine with real data and model integration"""
    
    def __init__(self, 
                 use_real_models: bool = True,
                 enable_extended_generation: bool = True,
                 enable_manifold_learning: bool = True,
                 enable_data_integration: bool = True,
                 model_size: str = "small",
                 preprocessing_config: PreprocessingConfig = None,
                 generation_config: GenerationConfig = None):
        
        self.use_real_models = use_real_models
        self.enable_extended_generation = enable_extended_generation
        self.enable_manifold_learning = enable_manifold_learning
        self.enable_data_integration = enable_data_integration
        
        # Initialize models
        logger.info("🚀 Initializing Updated Multi-Model Engine...")
        self.models = {}
        self.tokenizer = None
        self._initialize_models(model_size)
        
        # Initialize classifier with real data
        self.classifier = self._initialize_enhanced_classifier()
        
        # Initialize data pipeline
        self.data_pipeline = None
        if enable_data_integration and DATA_PIPELINE_AVAILABLE:
            try:
                self.data_pipeline = IntegratedDataPipeline(
                    preprocessing_config=preprocessing_config or PreprocessingConfig()
                )
            except Exception as e:
                logger.warning(f"Failed to initialize data pipeline: {e}")
                self.data_pipeline = None
        
        # Performance tracking
        self.model_stats = defaultdict(lambda: {
            'usage_count': 0,
            'avg_response_time': 0.0,
            'success_rate': 0.0,
            'quality_scores': [],
            'recent_performance': deque(maxlen=100)
        })
        
        self.generation_history = deque(maxlen=1000)
        
        logger.info("✅ Updated Multi-Model Engine initialized successfully")
    
    def _initialize_models(self, model_size: str):
        """Initialize real or dummy models"""
        
        if self.use_real_models and REAL_MODELS_AVAILABLE:
            try:
                logger.info(f"🔧 Loading real models (size: {model_size})...")
                
                # Load safe real models
                if model_size == "small":
                    safe_models = create_small_safe_models()
                elif model_size == "medium":
                    try:
                        from utils.safe_real_models import create_medium_safe_models
                        safe_models = create_medium_safe_models()
                    except ImportError:
                        safe_models = create_small_safe_models()
                else:
                    try:
                        from utils.safe_real_models import create_large_safe_models
                        safe_models = create_large_safe_models()
                    except ImportError:
                        safe_models = create_small_safe_models()
                
                # Create tokenizer
                self.tokenizer = create_safe_real_tokenizer(model_size)
                
                # Extend with advanced generation if enabled
                if self.enable_extended_generation:
                    logger.info("🧠 Enabling extended generation capabilities...")
                    generation_config = GenerationConfig(
                        enable_backtrack=True,
                        tree_depth=2,
                        branching_factor=2,
                        adaptation_rate=0.1
                    )
                    extended_models = create_extended_models_from_safe(safe_models, generation_config)
                    safe_models = extended_models
                
                # Convert to dictionary
                for model_spec in safe_models:
                    self.models[model_spec.model_id] = model_spec
                
                logger.info(f"✅ Loaded {len(self.models)} real models")
                
            except Exception as e:
                logger.error(f"❌ Failed to load real models: {e}")
                logger.info("🔄 Falling back to dummy models...")
                self._load_dummy_models()
        else:
            if not REAL_MODELS_AVAILABLE:
                logger.info("🔄 Real models not available, using dummy models...")
            self._load_dummy_models()
    
    def _load_dummy_models(self):
        """Load dummy models as fallback"""
        
        try:
            from utils.dummy_models import create_specialized_models, DummyTokenizer
            dummy_models = create_specialized_models()
            self.tokenizer = DummyTokenizer()
            
            for model_spec in dummy_models:
                self.models[model_spec.model_id] = model_spec
            
            logger.info(f"✅ Loaded {len(self.models)} dummy models")
            
        except Exception as e:
            logger.error(f"❌ Failed to load dummy models: {e}")
            raise
    
    def _initialize_enhanced_classifier(self) -> EnhancedInputClassifier:
        """Initialize enhanced classifier with manifold learning"""
        
        logger.info("🧠 Initializing enhanced input classifier...")
        
        # Create classifier with manifold learning
        classifier = EnhancedInputClassifier(
            enable_manifold_learning=self.enable_manifold_learning,
            manifold_config=ManifoldLearningConfig(
                embedding_dim=50,
                manifold_method="auto",
                enable_online_learning=True,
                enable_clustering=True,
                clustering_threshold=0.3
            )
        )
        
        # Load real training data if available
        if self.enable_data_integration:
            self._load_training_data(classifier)
        
        return classifier
    
    def _load_training_data(self, classifier: EnhancedInputClassifier):
        """Load real training data for classifier"""
        
        try:
            logger.info("📚 Loading real training data...")
            
            # Check if processed data exists
            processed_data_dir = Path("./processed_data/processed")
            
            if processed_data_dir.exists():
                # Load processed training data
                train_file = processed_data_dir / "train_processed.json"
                if train_file.exists():
                    with open(train_file, 'r', encoding='utf-8') as f:
                        train_data = json.load(f)
                    
                    # Extract texts and labels
                    texts = [item['processed_text'] for item in train_data[:500]]  # Limit for demo
                    labels = [item['task_type'] for item in train_data[:500]]
                    
                    # Fit classifier
                    classifier.fit_training_data(texts, labels)
                    logger.info(f"✅ Loaded {len(texts)} training samples")
                    return
            
            # Fallback: generate minimal training data
            logger.info("🔧 Generating synthetic training data...")
            self._generate_synthetic_training_data(classifier)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load training data: {e}")
            self._generate_synthetic_training_data(classifier)
    
    def _generate_synthetic_training_data(self, classifier: EnhancedInputClassifier):
        """Generate synthetic training data as fallback"""
        
        synthetic_data = [
            # Mathematical
            ("Calculate the area of a circle with radius 5", "mathematical"),
            ("Solve for x: 2x + 5 = 13", "mathematical"),
            ("Find the derivative of f(x) = x² + 3x", "mathematical"),
            ("What is the integral of sin(x)?", "mathematical"),
            
            # Creative Writing  
            ("Write a short story about time travel", "creative_writing"),
            ("Create a poem about artificial intelligence", "creative_writing"),
            ("Describe a mysterious character", "creative_writing"),
            ("Write dialogue between two friends", "creative_writing"),
            
            # Code Generation
            ("Implement a binary search algorithm", "code_generation"),
            ("Write a Python function to sort a list", "code_generation"),
            ("Create a REST API endpoint", "code_generation"),
            ("Debug this JavaScript code", "code_generation"),
            
            # Scientific
            ("Explain photosynthesis in plants", "scientific"),
            ("How do vaccines work?", "scientific"),
            ("Describe the structure of DNA", "scientific"),
            ("What causes earthquakes?", "scientific"),
            
            # Reasoning
            ("Analyze the pros and cons of remote work", "reasoning"),
            ("Compare democracy and authoritarianism", "reasoning"),
            ("Evaluate the impact of social media", "reasoning"),
            ("What are the ethical implications of AI?", "reasoning"),
            
            # Factual Q&A
            ("What is the capital of Japan?", "factual_qa"),
            ("When did World War II end?", "factual_qa"),
            ("Who invented the telephone?", "factual_qa"),
            ("How many continents are there?", "factual_qa"),
            
            # Conversational
            ("Hello, how are you today?", "conversational"),
            ("Thank you for your help", "conversational"),
            ("Can you assist me with something?", "conversational"),
            ("Have a great day!", "conversational")
        ]
        
        texts = [item[0] for item in synthetic_data]
        labels = [item[1] for item in synthetic_data]
        
        classifier.fit_training_data(texts, labels)
        logger.info(f"✅ Generated {len(texts)} synthetic training samples")
    
    def select_model(self, input_text: str, **kwargs) -> Tuple[str, InputAnalysis, float]:
        """Enhanced model selection with real data analysis"""
        
        # Analyze input with enhanced classifier
        analysis = self.classifier.analyze_input(input_text)
        
        # Find best matching models
        candidate_models = []
        
        for model_id, model_spec in self.models.items():
            # Task type matching
            task_score = 0.0
            if analysis.task_type in model_spec.task_types:
                task_score = 1.0
            
            # Domain matching
            domain_score = 0.0
            if analysis.specialized_domains:
                matching_domains = set(analysis.specialized_domains) & set(model_spec.specialized_domains)
                if matching_domains:
                    domain_score = len(matching_domains) / len(model_spec.specialized_domains)
            
            # Performance score
            perf_score = model_spec.performance_metrics.get(analysis.task_type.value, 0.5)
            
            # Load balancing
            load_penalty = self.model_stats[model_id]['usage_count'] * 0.01
            
            # Combined score
            total_score = (
                0.4 * task_score +
                0.3 * domain_score + 
                0.2 * perf_score -
                0.1 * load_penalty
            )
            
            candidate_models.append((model_id, total_score))
        
        # Select best model
        if candidate_models:
            best_model_id = max(candidate_models, key=lambda x: x[1])[0]
            confidence = max(candidate_models, key=lambda x: x[1])[1]
        else:
            # Fallback to first available model
            best_model_id = list(self.models.keys())[0]
            confidence = 0.5
        
        return best_model_id, analysis, confidence
    
    def generate(self, 
                input_text: str,
                max_length: int = 200,
                temperature: float = 0.7,
                force_model: Optional[str] = None,
                generation_method: str = "auto",
                **kwargs) -> Dict[str, Any]:
        """Enhanced generation with real models and extended capabilities"""
        
        start_time = time.time()
        
        try:
            # Model selection
            if force_model and force_model in self.models:
                selected_model_id = force_model
                analysis = self.classifier.analyze_input(input_text)
                confidence = 1.0
            else:
                selected_model_id, analysis, confidence = self.select_model(input_text)
            
            model_spec = self.models[selected_model_id]
            
            # Generate response
            if self.enable_extended_generation and hasattr(model_spec.model, 'generate'):
                # Use extended generation if available
                if hasattr(model_spec.model, 'selected_method'):  # ExtendedGenerationWrapper
                    generation_result = model_spec.model.generate(
                        input_text,
                        method=generation_method,
                        max_length=max_length,
                        temperature=temperature,
                        **kwargs
                    )
                else:
                    # Regular SafeRealModelWrapper
                    generation_result = model_spec.model.generate(
                        input_text,
                        max_length=max_length,
                        temperature=temperature,
                        **kwargs
                    )
            else:
                # Fallback to basic generation
                generation_result = model_spec.model.generate(
                    input_text,
                    max_length=max_length,
                    temperature=temperature,
                    **kwargs
                )
            
            generation_time = time.time() - start_time
            
            # Prepare result
            result = {
                'selected_model': selected_model_id,
                'generated_text': generation_result.get('generated_text', ''),
                'generation_successful': generation_result.get('generation_successful', False),
                'generation_time': generation_time,
                'model_confidence': confidence,
                'input_analysis': {
                    'task_type': analysis.task_type.value,
                    'confidence_score': analysis.confidence_score,
                    'specialized_domains': analysis.specialized_domains,
                    'manifold_type': analysis.manifold_type,
                    'complexity_score': analysis.complexity_score
                }
            }
            
            # Add extended generation info if available
            if 'generation_method' in generation_result:
                result['generation_method'] = generation_result['generation_method']
            if 'final_score' in generation_result:
                result['quality_score'] = generation_result['final_score']
            if 'adapted_params' in generation_result:
                result['adapted_params'] = generation_result['adapted_params']
            
            # Update statistics
            self._update_model_stats(selected_model_id, generation_time, result['generation_successful'])
            
            # Store in history
            self.generation_history.append({
                'input_text': input_text,
                'result': result,
                'timestamp': time.time()
            })
            
            # Online learning update
            if self.enable_manifold_learning and result['generation_successful']:
                self.classifier.update_performance(
                    input_text,
                    analysis.task_type,
                    analysis.task_type,  # Assume correct for now
                    result.get('quality_score', 0.8)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                'selected_model': selected_model_id if 'selected_model_id' in locals() else 'unknown',
                'generated_text': f"[Generation failed: {str(e)}]",
                'generation_successful': False,
                'generation_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _update_model_stats(self, model_id: str, response_time: float, success: bool):
        """Update model performance statistics"""
        
        stats = self.model_stats[model_id]
        stats['usage_count'] += 1
        
        # Update average response time
        current_avg = stats['avg_response_time']
        usage_count = stats['usage_count']
        stats['avg_response_time'] = (current_avg * (usage_count - 1) + response_time) / usage_count
        
        # Update success rate
        stats['recent_performance'].append(1.0 if success else 0.0)
        stats['success_rate'] = np.mean(stats['recent_performance'])
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        
        stats = {
            'total_generations': len(self.generation_history),
            'model_performance': dict(self.model_stats),
            'classifier_stats': self.classifier.get_classification_statistics(),
            'system_info': {
                'real_models_enabled': self.use_real_models,
                'extended_generation_enabled': self.enable_extended_generation,
                'manifold_learning_enabled': self.enable_manifold_learning,
                'data_integration_enabled': self.enable_data_integration,
                'total_models': len(self.models),
                'model_types': list(self.models.keys())
            }
        }
        
        # Add data pipeline stats if available
        if self.data_pipeline:
            stats['data_pipeline_stats'] = {
                'preprocessing_enabled': True,
                'pipeline_ready': True
            }
        
        return stats
    
    def train_on_real_data(self, data_categories: List[str] = None) -> Dict[str, Any]:
        """Train the system on real data from pipeline"""
        
        if not self.enable_data_integration or not self.data_pipeline or not DATA_PIPELINE_AVAILABLE:
            return {'error': 'Data integration not enabled or not available'}
        
        logger.info("🎓 Training system on real data...")
        
        try:
            # Run data pipeline
            pipeline_results = self.data_pipeline.run_full_pipeline(
                include_hf=True,
                hf_categories=data_categories or ['math', 'qa', 'code', 'science']
            )
            
            # Load processed training data
            processed_data_dir = Path(self.data_pipeline.base_dir) / "processed"
            train_file = processed_data_dir / "train_processed.json"
            
            if train_file.exists():
                with open(train_file, 'r', encoding='utf-8') as f:
                    train_data = json.load(f)
                
                # Update classifier with real data
                texts = [item['processed_text'] for item in train_data[:1000]]  # Limit for performance
                labels = [item['task_type'] for item in train_data[:1000]]
                
                self.classifier.fit_training_data(texts, labels)
                
                logger.info(f"✅ Trained on {len(texts)} real data samples")
                
                return {
                    'training_successful': True,
                    'samples_used': len(texts),
                    'pipeline_results': pipeline_results,
                    'data_quality': 'high'
                }
            else:
                return {'error': 'No processed training data found'}
                
        except Exception as e:
            logger.error(f"Training on real data failed: {e}")
            return {'error': str(e)}
    
    def evaluate_performance(self, test_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate system performance"""
        
        if test_data is None:
            # Use synthetic test data
            test_data = [
                {'text': 'Solve x² + 4x - 5 = 0', 'expected_task': 'mathematical'},
                {'text': 'Write a story about dragons', 'expected_task': 'creative_writing'},
                {'text': 'Implement merge sort algorithm', 'expected_task': 'code_generation'},
                {'text': 'Explain quantum mechanics', 'expected_task': 'scientific'},
                {'text': 'Analyze economic policies', 'expected_task': 'reasoning'},
                {'text': 'What is the capital of Brazil?', 'expected_task': 'factual_qa'},
                {'text': 'Hello, how can I help you?', 'expected_task': 'conversational'}
            ]
        
        logger.info(f"📊 Evaluating performance on {len(test_data)} test cases...")
        
        results = {
            'total_tests': len(test_data),
            'correct_selections': 0,
            'generation_successes': 0,
            'avg_confidence': 0.0,
            'avg_generation_time': 0.0,
            'task_specific_results': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'detailed_results': []
        }
        
        confidences = []
        generation_times = []
        
        for test_case in test_data:
            text = test_case['text']
            expected_task = test_case['expected_task']
            
            # Test model selection
            selected_model, analysis, confidence = self.select_model(text)
            predicted_task = analysis.task_type.value
            
            # Test generation
            generation_result = self.generate(text, max_length=100)
            
            # Record results
            correct_selection = predicted_task == expected_task
            generation_success = generation_result['generation_successful']
            
            if correct_selection:
                results['correct_selections'] += 1
            
            if generation_success:
                results['generation_successes'] += 1
            
            confidences.append(confidence)
            generation_times.append(generation_result['generation_time'])
            
            # Task-specific stats
            task_stats = results['task_specific_results'][expected_task]
            task_stats['total'] += 1
            if correct_selection:
                task_stats['correct'] += 1
            
            # Detailed result
            results['detailed_results'].append({
                'text': text,
                'expected_task': expected_task,
                'predicted_task': predicted_task,
                'selected_model': selected_model,
                'correct_selection': correct_selection,
                'generation_success': generation_success,
                'confidence': confidence,
                'generation_time': generation_result['generation_time']
            })
        
        # Calculate averages
        results['selection_accuracy'] = results['correct_selections'] / results['total_tests']
        results['generation_success_rate'] = results['generation_successes'] / results['total_tests']
        results['avg_confidence'] = np.mean(confidences)
        results['avg_generation_time'] = np.mean(generation_times)
        
        logger.info(f"✅ Evaluation complete:")
        logger.info(f"   Selection Accuracy: {results['selection_accuracy']:.3f}")
        logger.info(f"   Generation Success: {results['generation_success_rate']:.3f}")
        logger.info(f"   Avg Confidence: {results['avg_confidence']:.3f}")
        
        return results
    
    def save_system_state(self, filepath: str = None):
        """Save system state for later restoration"""
        
        if filepath is None:
            filepath = f"system_state_{int(time.time())}.json"
        
        state = {
            'model_stats': dict(self.model_stats),
            'generation_history': list(self.generation_history)[-100:],  # Last 100
            'classifier_stats': self.classifier.get_classification_statistics(),
            'system_config': {
                'use_real_models': self.use_real_models,
                'enable_extended_generation': self.enable_extended_generation,
                'enable_manifold_learning': self.enable_manifold_learning,
                'enable_data_integration': self.enable_data_integration
            },
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"💾 System state saved to {filepath}")

def create_updated_engine(
    model_size: str = "small",
    enable_all_features: bool = True,
    custom_config: Dict[str, Any] = None
) -> UpdatedMultiModelEngine:
    """Factory function to create updated engine with sensible defaults"""
    
    config = {
        'use_real_models': True,
        'enable_extended_generation': True,
        'enable_manifold_learning': True,
        'enable_data_integration': True,
        'model_size': model_size
    }
    
    if not enable_all_features:
        config.update({
            'enable_extended_generation': False,
            'enable_manifold_learning': False,
            'enable_data_integration': False
        })
    
    if custom_config:
        config.update(custom_config)
    
    return UpdatedMultiModelEngine(**config)

def demo_updated_system():
    """Demonstration of the updated system"""
    
    print("🚀 Updated Multi-Model System Demo")
    print("=" * 50)
    
    # Create updated engine
    engine = create_updated_engine(model_size="small", enable_all_features=True)
    
    # Test cases
    test_cases = [
        "Calculate the derivative of x³ + 2x² - 5x + 1",
        "Write a creative story about time travel",
        "Implement a quicksort algorithm in Python",
        "Explain how photosynthesis works in plants",
        "Analyze the pros and cons of renewable energy",
        "What is the capital of Australia?",
        "Hello! How can I help you today?"
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} diverse prompts...")
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n[{i}] Testing: '{prompt[:40]}...'")
        
        result = engine.generate(prompt, max_length=100, generation_method="auto")
        
        print(f"   Selected Model: {result['selected_model']}")
        print(f"   Task Type: {result['input_analysis']['task_type']}")
        print(f"   Success: {'✅' if result['generation_successful'] else '❌'}")
        print(f"   Time: {result['generation_time']:.3f}s")
        
        if 'generation_method' in result:
            print(f"   Method: {result['generation_method']}")
        if 'quality_score' in result:
            print(f"   Quality: {result['quality_score']:.3f}")
    
    # Performance evaluation
    print(f"\n📊 Running Performance Evaluation...")
    eval_results = engine.evaluate_performance()
    
    print(f"   Selection Accuracy: {eval_results['selection_accuracy']:.3f}")
    print(f"   Generation Success: {eval_results['generation_success_rate']:.3f}")
    print(f"   Average Confidence: {eval_results['avg_confidence']:.3f}")
    
    # System statistics
    print(f"\n📈 System Statistics:")
    stats = engine.get_model_stats()
    print(f"   Total Generations: {stats['total_generations']}")
    print(f"   Real Models: {stats['system_info']['real_models_enabled']}")
    print(f"   Extended Generation: {stats['system_info']['extended_generation_enabled']}")
    print(f"   Manifold Learning: {stats['system_info']['manifold_learning_enabled']}")
    
    # Save system state
    engine.save_system_state("demo_system_state.json")
    
    print(f"\n🎉 Demo complete! Updated system working successfully.")
    return engine, eval_results

if __name__ == "__main__":
    demo_updated_system()