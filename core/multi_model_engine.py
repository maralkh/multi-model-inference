# File: core/multi_model_engine.py
"""Multi-model inference engine with automatic model selection"""

import time
import copy
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

from .types import ModelSpec, TaskType
from .input_classifier import InputClassifier, InputAnalysis

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
            time.sleep(0.1)  # Simulate generation time
            
            return {
                'generated_text': f"[Generated response to: {prompt[:50]}...]",
                'tokens_generated': max_length,
                'prm_scores': [0.8, 0.7, 0.9] if config and config.use_prm else [],
                'orm_score': 0.85 if config and config.use_orm else 0.0,
                'search_strategy': config.search_strategy if config else 'greedy',
                'generation_successful': True
            }
    
    return MockRewardGuidedEngine(model, prm, orm, tokenizer)

class MultiModelInferenceEngine:
    """Multi-model inference engine with automatic model selection"""
    
    def __init__(self, models: List[ModelSpec], tokenizer, default_model_id: str = None):
        self.models = {model.model_id: model for model in models}
        self.tokenizer = tokenizer
        self.default_model_id = default_model_id or models[0].model_id
        self.classifier = InputClassifier()
        self.inference_engines = {}
        
        # Create inference engines for each model
        for model_spec in models:
            self.inference_engines[model_spec.model_id] = create_reward_guided_engine(
                model=model_spec.model,
                prm=model_spec.prm,
                orm=model_spec.orm,
                tokenizer=tokenizer
            )
        
        # Track usage statistics
        self.usage_stats = {model_id: 0 for model_id in self.models.keys()}
        self.performance_history = []
    
    def select_model(self, input_text: str) -> Tuple[str, InputAnalysis, float]:
        """Select the best model for the given input"""
        
        # Analyze input
        analysis = self.classifier.analyze_input(input_text)
        
        # Calculate model scores
        model_scores = {}
        
        for model_id, model_spec in self.models.items():
            score = 0.0
            
            # Task type compatibility
            if analysis.task_type in model_spec.task_types:
                score += 0.4
            
            # Domain specialization
            domain_matches = len(set(analysis.domain_indicators) & 
                               set(model_spec.specialized_domains))
            score += domain_matches * 0.2
            
            # Performance metrics
            task_performance = model_spec.performance_metrics.get(
                analysis.task_type.value, 0.5)
            score += task_performance * 0.3
            
            # Complexity handling
            if analysis.complexity_score > 0.7:
                # Prefer models with better reasoning capabilities
                reasoning_score = model_spec.performance_metrics.get('reasoning', 0.5)
                score += reasoning_score * 0.1
            
            model_scores[model_id] = score
        
        # Select best model
        best_model_id = max(model_scores.keys(), key=lambda k: model_scores[k])
        confidence = model_scores[best_model_id] / max(sum(model_scores.values()), 1)
        
        # Fallback to default if confidence is too low
        if confidence < 0.3:
            best_model_id = self.default_model_id
            confidence = 0.3
        
        return best_model_id, analysis, confidence
    
    def generate(self, prompt: str, config: RewardGuidedConfig = None, 
                max_length: int = 200, temperature: float = 0.7, 
                force_model: str = None) -> Dict[str, Any]:
        """Generate text using the most appropriate model"""
        
        start_time = time.time()
        
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
            engine = self.inference_engines[selected_model_id]
            result = engine.generate(
                prompt=prompt,
                config=config,
                max_length=max_length,
                temperature=temperature
            )
            
            generation_time = time.time() - start_time
            
            # Add metadata
            result.update({
                'selected_model': selected_model_id,
                'model_selection_confidence': selection_confidence,
                'input_analysis': {
                    'task_type': analysis.task_type.value,
                    'confidence': analysis.confidence,
                    'complexity_score': analysis.complexity_score,
                    'keywords': analysis.keywords,
                    'domains': analysis.domain_indicators
                },
                'generation_time': generation_time,
                'config_used': config.__dict__
            })
            
            # Record performance
            self.performance_history.append({
                'timestamp': time.time(),
                'model_id': selected_model_id,
                'task_type': analysis.task_type.value,
                'complexity': analysis.complexity_score,
                'generation_time': generation_time,
                'success': True
            })
            
            return result
            
        except Exception as e:
            # Fallback to default model
            if selected_model_id != self.default_model_id:
                try:
                    fallback_engine = self.inference_engines[self.default_model_id]
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
                        'error': str(e)
                    })
                    
                    return result
                    
                except Exception as fallback_error:
                    raise Exception(f"Both primary ({e}) and fallback ({fallback_error}) failed")
            else:
                raise e
    
    def _get_adaptive_config(self, analysis: InputAnalysis) -> RewardGuidedConfig:
        """Get adaptive configuration based on input analysis"""
        
        task_configs = {
            TaskType.MATHEMATICAL: RewardGuidedConfig(
                search_strategy="guided_sampling",
                use_prm=True,
                use_orm=False,
                prm_weight=0.8,
                reward_alpha=0.3,
                early_stopping=True
            ),
            
            TaskType.CREATIVE_WRITING: RewardGuidedConfig(
                search_strategy="best_of_n",
                num_candidates=8,
                use_prm=False,
                use_orm=True,
                orm_weight=0.9,
                diversity_penalty=0.2
            ),
            
            TaskType.FACTUAL_QA: RewardGuidedConfig(
                search_strategy="beam_search",
                num_beams=5,
                use_prm=True,
                use_orm=True,
                prm_weight=0.4,
                orm_weight=0.6,
                early_stopping=True
            ),
            
            TaskType.REASONING: RewardGuidedConfig(
                search_strategy="tree_search",
                use_prm=True,
                use_orm=True,
                prm_weight=0.5,
                orm_weight=0.5,
                tree_depth=4,
                branching_factor=3
            ),
            
            TaskType.CODE_GENERATION: RewardGuidedConfig(
                search_strategy="beam_search",
                num_beams=4,
                use_prm=True,
                use_orm=True,
                prm_weight=0.6,
                orm_weight=0.4,
                early_stopping=True
            )
        }
        
        base_config = task_configs.get(analysis.task_type, task_configs[TaskType.FACTUAL_QA])
        
        # Adjust based on complexity
        if analysis.complexity_score > 0.8:
            # More thorough search for complex problems
            if hasattr(base_config, 'num_beams'):
                base_config.num_beams = min(base_config.num_beams + 2, 8)
            if hasattr(base_config, 'tree_depth'):
                base_config.tree_depth = min(base_config.tree_depth + 1, 6)
        
        return base_config
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get usage and performance statistics"""
        
        total_usage = sum(self.usage_stats.values())
        
        stats = {
            'usage_distribution': {
                model_id: (count / max(total_usage, 1)) * 100 
                for model_id, count in self.usage_stats.items()
            },
            'total_generations': total_usage,
            'model_performance': {}
        }
        
        # Calculate performance metrics per model
        for model_id in self.models.keys():
            model_history = [h for h in self.performance_history if h['model_id'] == model_id]
            
            if model_history:
                avg_time = np.mean([h['generation_time'] for h in model_history])
                success_rate = sum(1 for h in model_history if h['success']) / len(model_history)
                avg_complexity = np.mean([h['complexity'] for h in model_history])
                
                stats['model_performance'][model_id] = {
                    'average_generation_time': avg_time,
                    'success_rate': success_rate,
                    'average_complexity_handled': avg_complexity,
                    'total_requests': len(model_history)
                }
        
        return stats

class LoadBalancedMultiModelEngine(MultiModelInferenceEngine):
    """Enhanced engine with load balancing capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_load = {model_id: 0 for model_id in self.models.keys()}
        self.max_concurrent = 3
        self.load_history = []
    
    def select_model(self, input_text: str) -> Tuple[str, InputAnalysis, float]:
        """Select model with load balancing consideration"""
        
        # Get base selection
        primary_model, analysis, confidence = super().select_model(input_text)
        
        # Check load balancing
        if self.model_load[primary_model] >= self.max_concurrent:
            print(f"🔄 Model {primary_model} at capacity ({self.model_load[primary_model]}/{self.max_concurrent})")
            
            # Find alternative models that can handle this task type
            alternatives = []
            for model_id, model_spec in self.models.items():
                if (self.model_load[model_id] < self.max_concurrent and 
                    analysis.task_type in model_spec.task_types):
                    alternatives.append(model_id)
            
            if alternatives:
                # Select least loaded alternative
                alternative = min(alternatives, key=lambda m: self.model_load[m])
                print(f"🔄 Load balancing: {primary_model} -> {alternative}")
                return alternative, analysis, confidence * 0.9
            else:
                print(f"⚠️ No alternatives available, using primary model anyway")
        
        return primary_model, analysis, confidence
    
    def generate(self, prompt: str, **kwargs):
        """Generate with load tracking"""
        
        # Get model selection
        selected_model, analysis, confidence = self.select_model(prompt)
        
        # Track load
        self.model_load[selected_model] += 1
        load_start_time = time.time()
        
        try:
            # Call parent generate method with forced model
            result = super().generate(prompt, force_model=selected_model, **kwargs)
            
            # Record load metrics
            load_duration = time.time() - load_start_time
            self.load_history.append({
                'model_id': selected_model,
                'load_level': self.model_load[selected_model],
                'duration': load_duration,
                'timestamp': time.time()
            })
            
            return result
            
        finally:
            # Always decrement load
            self.model_load[selected_model] -= 1
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        
        return {
            'current_load': dict(self.model_load),
            'max_concurrent': self.max_concurrent,
            'load_history_size': len(self.load_history),
            'average_load_duration': np.mean([h['duration'] for h in self.load_history]) if self.load_history else 0,
            'peak_loads': {
                model_id: max([h['load_level'] for h in self.load_history if h['model_id'] == model_id], default=0)
                for model_id in self.models.keys()
            }
        }