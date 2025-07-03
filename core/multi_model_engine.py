# File: core/fixed_multi_model_engine.py
"""Fixed multi-model inference engine with proper error handling"""

import time
import copy
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

# Use fixed types that have all required attributes
try:
    from .input_types import ModelSpec, TaskType, InputAnalysis, create_safe_input_analysis
except ImportError:
    # Fallback imports
    try:
        from core.input_types import ModelSpec, TaskType, InputAnalysis, create_safe_input_analysis
    except ImportError:
        # Ultimate fallback - create minimal types
        from enum import Enum
        from dataclasses import dataclass, field
        from typing import List, Dict, Any
        
        class TaskType(Enum):
            MATHEMATICAL = "mathematical"
            CREATIVE_WRITING = "creative_writing"
            FACTUAL_QA = "factual_qa"
            REASONING = "reasoning"
            CODE_GENERATION = "code_generation"
            SCIENTIFIC = "scientific"
            CONVERSATIONAL = "conversational"
        
        @dataclass
        class InputAnalysis:
            task_type: TaskType
            confidence: float
            features: Dict[str, Any] = field(default_factory=dict)
            keywords: List[str] = field(default_factory=list)
            complexity_score: float = 0.0
            domain_indicators: List[str] = field(default_factory=list)
        
        def create_safe_input_analysis(task_type, confidence, **kwargs):
            return InputAnalysis(
                task_type=task_type,
                confidence=confidence,
                **kwargs
            )

# Safe attribute access helper
def safe_getattr(obj, attr_name: str, default_value=None):
    """Safely get attribute from object with fallback"""
    try:
        return getattr(obj, attr_name, default_value)
    except (AttributeError, TypeError):
        return default_value

class SafeInputClassifier:
    """Safe input classifier that always returns properly formatted InputAnalysis"""
    
    def __init__(self):
        self.task_patterns = {
            TaskType.MATHEMATICAL: {
                'keywords': ['solve', 'calculate', 'equation', 'formula', 'math'],
                'indicators': ['+', '-', '*', '/', '=', 'x²', '∫', '∑']
            },
            TaskType.CREATIVE_WRITING: {
                'keywords': ['story', 'write', 'creative', 'character', 'narrative'],
                'indicators': ['once upon', 'imagine', 'create']
            },
            TaskType.CODE_GENERATION: {
                'keywords': ['code', 'function', 'algorithm', 'program', 'implement'],
                'indicators': ['def ', 'class ', '{', '}', 'import']
            },
            TaskType.REASONING: {
                'keywords': ['analyze', 'compare', 'evaluate', 'reasoning', 'logic'],
                'indicators': ['pros and cons', 'because', 'therefore']
            },
            TaskType.FACTUAL_QA: {
                'keywords': ['what', 'who', 'when', 'where', 'why', 'how'],
                'indicators': ['?', 'explain', 'define']
            },
            TaskType.SCIENTIFIC: {
                'keywords': ['research', 'experiment', 'hypothesis', 'theory', 'science'],
                'indicators': ['study', 'data', 'evidence']
            },
            TaskType.CONVERSATIONAL: {
                'keywords': ['hello', 'hi', 'thanks', 'please', 'chat'],
                'indicators': ['how are you', 'good morning', 'thank you']
            }
        }
    
    def analyze_input(self, text: str) -> InputAnalysis:
        """Analyze input and return properly formatted InputAnalysis"""
        
        if not text or not isinstance(text, str):
            return create_safe_input_analysis(
                task_type=TaskType.CONVERSATIONAL,
                confidence=0.5,
                text=text or "",
                keywords=[],
                domain_indicators=[],
                complexity_score=0.0
            )
        
        text_lower = text.lower()
        
        # Calculate scores for each task type
        task_scores = {}
        
        for task_type, patterns in self.task_patterns.items():
            score = 0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in patterns['keywords'] 
                                if keyword in text_lower)
            score += keyword_matches * 2
            
            # Indicator matching  
            indicator_matches = sum(1 for indicator in patterns['indicators']
                                  if indicator in text_lower)
            score += indicator_matches * 3
            
            task_scores[task_type] = score
        
        # Determine best task type
        best_task = max(task_scores.keys(), key=lambda k: task_scores[k])
        max_score = task_scores[best_task]
        total_score = sum(task_scores.values())
        confidence = max_score / max(total_score, 1) if total_score > 0 else 0.5
        
        # Ensure minimum confidence
        confidence = max(confidence, 0.1)
        
        # Extract keywords and features
        keywords = self._extract_keywords(text)
        domain_indicators = self._extract_domains(text)
        complexity_score = self._calculate_complexity(text)
        
        # Create comprehensive features dictionary
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'task_scores': {t.value: s for t, s in task_scores.items()},
            'total_score': total_score,
            'analysis_method': 'safe_classifier'
        }
        
        return create_safe_input_analysis(
            task_type=best_task,
            confidence=confidence,
            text=text,
            features=features,
            keywords=keywords,
            domain_indicators=domain_indicators,
            complexity_score=complexity_score,
            processing_time=0.01,  # Mock processing time
            method_used='safe_pattern_matching'
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords safely"""
        if not text:
            return []
        
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common words
        stopwords = {'the', 'and', 'or', 'but', 'this', 'that', 'with', 'have'}
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Return top 5
        return list(set(keywords))[:5]
    
    def _extract_domains(self, text: str) -> List[str]:
        """Extract domain indicators safely"""
        if not text:
            return []
        
        text_lower = text.lower()
        domains = []
        
        domain_map = {
            'mathematics': ['math', 'equation', 'calculate', 'solve'],
            'science': ['research', 'experiment', 'theory', 'hypothesis'],
            'technology': ['code', 'algorithm', 'software', 'programming'],
            'creative': ['story', 'write', 'creative', 'narrative']
        }
        
        for domain, keywords in domain_map.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)
        
        return domains
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate complexity score safely"""
        if not text:
            return 0.0
        
        # Simple complexity calculation
        complexity = 0.0
        complexity += min(len(text) / 500, 0.5)  # Length factor
        complexity += min(len(text.split()) / 100, 0.3)  # Word count factor
        complexity += min(text.count('?') + text.count('!'), 0.2)  # Punctuation factor
        
        return min(complexity, 1.0)

# Mock reward guided config for demonstration
class RewardGuidedConfig:
    def __init__(self, search_strategy="beam_search", num_beams=4, num_candidates=8,
                 use_prm=True, use_orm=True, prm_weight=0.5, orm_weight=0.5,
                 reward_alpha=0.3, diversity_penalty=0.1, early_stopping=True,
                 tree_depth=3, branching_factor=2):
        self.search_strategy = search_strategy
        self.num_beams = num_beams
        self.num_candidates = num_candidates
        self.use_prm = use_prm
        self.use_orm = use_orm
        self.prm_weight = prm_weight
        self.orm_weight = orm_weight
        self.reward_alpha = reward_alpha
        self.diversity_penalty = diversity_penalty
        self.early_stopping = early_stopping
        self.tree_depth = tree_depth
        self.branching_factor = branching_factor

# Mock reward guided engine for demonstration
def create_reward_guided_engine(model, prm, orm, tokenizer):
    class MockRewardGuidedEngine:
        def __init__(self, model, prm, orm, tokenizer):
            self.model = model
            self.prm = prm
            self.orm = orm
            self.tokenizer = tokenizer
        
        def generate(self, prompt, config=None, max_length=200, temperature=0.7):
            # Mock generation with some realistic timing
            time.sleep(0.05)  # Simulate generation time
            
            return {
                'generated_text': f"[Generated response to: {prompt[:50]}...]",
                'tokens_generated': min(max_length, len(prompt.split()) + 20),
                'prm_scores': [0.8, 0.7, 0.9] if config and config.use_prm else [],
                'orm_score': 0.85 if config and config.use_orm else 0.0,
                'search_strategy': config.search_strategy if config else 'greedy',
                'generation_successful': True,
                'model_id': getattr(model, 'model_id', 'unknown_model')
            }
    
    return MockRewardGuidedEngine(model, prm, orm, tokenizer)

class FixedMultiModelInferenceEngine:
    """Fixed multi-model inference engine with comprehensive error handling"""
    
    def __init__(self, models: List[ModelSpec], tokenizer, default_model_id: str = None):
        self.models = {model.model_id: model for model in models}
        self.tokenizer = tokenizer
        self.default_model_id = default_model_id or models[0].model_id
        self.classifier = SafeInputClassifier()
        self.inference_engines = {}
        
        # Create inference engines for each model
        for model_spec in models:
            try:
                self.inference_engines[model_spec.model_id] = create_reward_guided_engine(
                    model=model_spec.model,
                    prm=model_spec.prm,
                    orm=model_spec.orm,
                    tokenizer=tokenizer
                )
            except Exception as e:
                print(f"Warning: Failed to create engine for {model_spec.model_id}: {e}")
                continue
        
        # Track usage statistics
        self.usage_stats = {model_id: 0 for model_id in self.models.keys()}
        self.performance_history = []
        self.error_count = 0
    
    def select_model(self, input_text: str) -> Tuple[str, InputAnalysis, float]:
        """Select the best model for the given input with safe error handling"""
        
        try:
            # Analyze input safely
            analysis = self.classifier.analyze_input(input_text)
            
            # Calculate model scores safely
            model_scores = {}
            
            for model_id, model_spec in self.models.items():
                score = 0.0
                
                try:
                    # Task type compatibility
                    if analysis.task_type in safe_getattr(model_spec, 'task_types', []):
                        score += 0.4
                    
                    # Domain specialization
                    domain_indicators = safe_getattr(analysis, 'domain_indicators', [])
                    specialized_domains = safe_getattr(model_spec, 'specialized_domains', [])
                    
                    if domain_indicators and specialized_domains:
                        domain_matches = len(set(domain_indicators) & set(specialized_domains))
                        score += domain_matches * 0.2
                    
                    # Performance metrics
                    performance_metrics = safe_getattr(model_spec, 'performance_metrics', {})
                    task_performance = performance_metrics.get(analysis.task_type.value, 0.5)
                    score += task_performance * 0.3
                    
                    # Complexity handling
                    complexity_score = safe_getattr(analysis, 'complexity_score', 0.0)
                    if complexity_score > 0.7:
                        reasoning_score = performance_metrics.get('reasoning', 0.5)
                        score += reasoning_score * 0.1
                    
                except Exception as e:
                    print(f"Warning: Error calculating score for {model_id}: {e}")
                    score = 0.1  # Minimal fallback score
                
                model_scores[model_id] = score
            
            # Select best model
            if model_scores:
                best_model_id = max(model_scores.keys(), key=lambda k: model_scores[k])
                confidence = model_scores[best_model_id] / max(sum(model_scores.values()), 1)
            else:
                best_model_id = self.default_model_id
                confidence = 0.3
            
            # Fallback to default if confidence is too low
            if confidence < 0.2:
                best_model_id = self.default_model_id
                confidence = 0.3
            
            return best_model_id, analysis, confidence
            
        except Exception as e:
            print(f"Error in model selection: {e}")
            self.error_count += 1
            
            # Create fallback analysis
            fallback_analysis = create_safe_input_analysis(
                task_type=TaskType.CONVERSATIONAL,
                confidence=0.3,
                text=input_text,
                keywords=[],
                domain_indicators=[],
                complexity_score=0.0
            )
            
            return self.default_model_id, fallback_analysis, 0.3
    
    def generate(self, prompt: str, config: RewardGuidedConfig = None, 
                max_length: int = 200, temperature: float = 0.7, 
                force_model: str = None) -> Dict[str, Any]:
        """Generate text using the most appropriate model with comprehensive error handling"""
        
        start_time = time.time()
        
        try:
            # Model selection
            if force_model and force_model in self.models:
                selected_model_id = force_model
                analysis = self.classifier.analyze_input(prompt)
                selection_confidence = 1.0
            else:
                selected_model_id, analysis, selection_confidence = self.select_model(prompt)
            
            # Update usage statistics
            self.usage_stats[selected_model_id] += 1
            
            # Adaptive configuration based on task type
            if config is None:
                config = self._get_adaptive_config(analysis)
            
            # Generate using selected model
            try:
                engine = self.inference_engines.get(selected_model_id)
                if engine is None:
                    raise ValueError(f"No engine available for model {selected_model_id}")
                
                result = engine.generate(
                    prompt=prompt,
                    config=config,
                    max_length=max_length,
                    temperature=temperature
                )
                
                generation_time = time.time() - start_time
                
                # Add metadata safely
                result.update({
                    'selected_model': selected_model_id,
                    'model_selection_confidence': selection_confidence,
                    'input_analysis': self._analysis_to_dict(analysis),
                    'generation_time': generation_time,
                    'config_used': config.__dict__ if config else {},
                    'success': True,
                    'error': None
                })
                
                # Record performance
                self._record_performance(selected_model_id, analysis, generation_time, True)
                
                return result
                
            except Exception as generation_error:
                print(f"Generation error with {selected_model_id}: {generation_error}")
                
                # Try fallback to default model
                if selected_model_id != self.default_model_id:
                    try:
                        fallback_engine = self.inference_engines.get(self.default_model_id)
                        if fallback_engine:
                            result = fallback_engine.generate(
                                prompt=prompt,
                                config=config,
                                max_length=max_length,
                                temperature=temperature
                            )
                            
                            result.update({
                                'selected_model': self.default_model_id,
                                'fallback_used': True,
                                'original_selection': selected_model_id,
                                'error': str(generation_error),
                                'success': True
                            })
                            
                            return result
                    except Exception as fallback_error:
                        print(f"Fallback also failed: {fallback_error}")
                
                # Ultimate fallback - return error response
                return {
                    'selected_model': selected_model_id,
                    'generated_text': f"[Error: Unable to generate response]",
                    'generation_successful': False,
                    'error': str(generation_error),
                    'success': False,
                    'generation_time': time.time() - start_time
                }
        
        except Exception as e:
            print(f"Critical error in generate method: {e}")
            self.error_count += 1
            
            return {
                'selected_model': self.default_model_id,
                'generated_text': f"[Critical Error: {str(e)}]",
                'generation_successful': False,
                'error': str(e),
                'success': False,
                'generation_time': time.time() - start_time
            }
    
    def _analysis_to_dict(self, analysis: InputAnalysis) -> Dict[str, Any]:
        """Safely convert analysis to dictionary"""
        try:
            if hasattr(analysis, 'to_dict'):
                return analysis.to_dict()
            else:
                return {
                    'task_type': safe_getattr(analysis, 'task_type', TaskType.CONVERSATIONAL).value,
                    'confidence': safe_getattr(analysis, 'confidence', 0.5),
                    'complexity_score': safe_getattr(analysis, 'complexity_score', 0.0),
                    'keywords': safe_getattr(analysis, 'keywords', []),
                    'domains': safe_getattr(analysis, 'domain_indicators', [])
                }
        except Exception as e:
            print(f"Error converting analysis to dict: {e}")
            return {'error': str(e)}
    
    def _record_performance(self, model_id: str, analysis: InputAnalysis, 
                           generation_time: float, success: bool):
        """Record performance metrics safely"""
        try:
            self.performance_history.append({
                'timestamp': time.time(),
                'model_id': model_id,
                'task_type': safe_getattr(analysis, 'task_type', TaskType.CONVERSATIONAL).value,
                'complexity': safe_getattr(analysis, 'complexity_score', 0.0),
                'generation_time': generation_time,
                'success': success
            })
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            print(f"Error recording performance: {e}")
    
    def _get_adaptive_config(self, analysis: InputAnalysis) -> RewardGuidedConfig:
        """Get adaptive configuration based on input analysis"""
        
        try:
            task_type = safe_getattr(analysis, 'task_type', TaskType.CONVERSATIONAL)
            
            task_configs = {
                TaskType.MATHEMATICAL: RewardGuidedConfig(
                    search_strategy="guided_sampling",
                    use_prm=True, use_orm=False, prm_weight=0.8
                ),
                TaskType.CREATIVE_WRITING: RewardGuidedConfig(
                    search_strategy="best_of_n", num_candidates=8,
                    use_prm=False, use_orm=True, orm_weight=0.9
                ),
                TaskType.FACTUAL_QA: RewardGuidedConfig(
                    search_strategy="beam_search", num_beams=5,
                    use_prm=True, use_orm=True, prm_weight=0.4, orm_weight=0.6
                ),
                TaskType.REASONING: RewardGuidedConfig(
                    search_strategy="tree_search", use_prm=True, use_orm=True,
                    tree_depth=4, branching_factor=3
                ),
                TaskType.CODE_GENERATION: RewardGuidedConfig(
                    search_strategy="beam_search", num_beams=4,
                    use_prm=True, use_orm=True, prm_weight=0.6, orm_weight=0.4
                )
            }
            
            base_config = task_configs.get(task_type, task_configs[TaskType.FACTUAL_QA])
            
            # Adjust based on complexity
            complexity = safe_getattr(analysis, 'complexity_score', 0.0)
            if complexity > 0.8:
                if hasattr(base_config, 'num_beams'):
                    base_config.num_beams = min(base_config.num_beams + 2, 8)
                if hasattr(base_config, 'tree_depth'):
                    base_config.tree_depth = min(base_config.tree_depth + 1, 6)
            
            return base_config
            
        except Exception as e:
            print(f"Error creating adaptive config: {e}")
            return RewardGuidedConfig()  # Return default config
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get usage and performance statistics with error handling"""
        
        try:
            total_usage = sum(self.usage_stats.values())
            
            stats = {
                'usage_distribution': {
                    model_id: (count / max(total_usage, 1)) * 100 
                    for model_id, count in self.usage_stats.items()
                },
                'total_generations': total_usage,
                'error_count': self.error_count,
                'model_performance': {},
                'available_models': list(self.models.keys()),
                'default_model': self.default_model_id
            }
            
            # Calculate performance metrics per model
            for model_id in self.models.keys():
                model_history = [h for h in self.performance_history if h['model_id'] == model_id]
                
                if model_history:
                    avg_time = np.mean([h['generation_time'] for h in model_history])
                    success_rate = sum(1 for h in model_history if h['success']) / len(model_history)
                    avg_complexity = np.mean([h['complexity'] for h in model_history])
                    
                    stats['model_performance'][model_id] = {
                        'average_generation_time': float(avg_time),
                        'success_rate': float(success_rate),
                        'average_complexity_handled': float(avg_complexity),
                        'total_requests': len(model_history)
                    }
            
            return stats
            
        except Exception as e:
            print(f"Error getting model stats: {e}")
            return {
                'error': str(e),
                'total_generations': 0,
                'available_models': list(self.models.keys()) if self.models else []
            }

# Convenience alias for backward compatibility
MultiModelInferenceEngine = FixedMultiModelInferenceEngine

def main():
    """Test the fixed multi-model engine"""
    print("🧪 Testing Fixed Multi-Model Engine")
    print("=" * 40)
    
    try:
        # Import dummy models for testing
        from utils.dummy_models import create_specialized_models, DummyTokenizer
        
        # Create models and engine
        models = create_specialized_models()
        tokenizer = DummyTokenizer()
        engine = FixedMultiModelInferenceEngine(models, tokenizer)
        
        # Test prompts
        test_prompts = [
            "Solve x² + 2x - 3 = 0",
            "Write a short story about robots",
            "Implement quicksort in Python",
            "What is the capital of Japan?",
            ""  # Empty prompt to test error handling
        ]
        
        print(f"\n🔬 Testing with {len(test_prompts)} prompts...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[{i}] Testing: '{prompt[:30]}{'...' if len(prompt) > 30 else ''}'")
            
            try:
                result = engine.generate(prompt, max_length=100)
                
                print(f"   Model: {result.get('selected_model', 'unknown')}")
                print(f"   Success: {result.get('success', False)}")
                print(f"   Time: {result.get('generation_time', 0):.3f}s")
                
                if result.get('error'):
                    print(f"   Error: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
        
        # Show statistics
        print(f"\n📊 Engine Statistics:")
        stats = engine.get_model_stats()
        print(f"   Total generations: {stats.get('total_generations', 0)}")
        print(f"   Errors: {stats.get('error_count', 0)}")
        print(f"   Available models: {len(stats.get('available_models', []))}")
        
        print(f"\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()