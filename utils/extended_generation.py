# File: utils/extended_generation.py
"""Extended generation methods integrating sampling strategies"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import time
import copy
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.safe_real_models import SafeRealModelWrapper, SafeRealModelConfig

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for extended generation methods"""
    # Basic parameters
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Advanced sampling
    use_enhanced_sampling: bool = True
    sampling_method: str = "adaptive"  # adaptive, tree, bayesian, genetic
    
    # Tree-based generation
    tree_depth: int = 3
    branching_factor: int = 2
    beam_size: int = 4
    
    # Adaptive parameters
    adaptation_rate: float = 0.1
    quality_threshold: float = 0.7
    diversity_penalty: float = 0.1
    
    # Backtracking
    enable_backtrack: bool = True
    backtrack_threshold: float = 0.5
    max_backtracks: int = 3
    
    # Quality scoring
    coherence_weight: float = 0.4
    fluency_weight: float = 0.3
    relevance_weight: float = 0.3

@dataclass
class GenerationNode:
    """Node for tree-based generation"""
    text: str
    score: float
    depth: int
    parent: Optional['GenerationNode'] = None
    children: List['GenerationNode'] = None
    tokens: List[int] = None
    probabilities: List[float] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class QualityScorer:
    """Scores generation quality using multiple metrics"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.repetition_penalty = 1.2
        self.length_penalty = 1.0
        
    def score_text(self, text: str, prompt: str) -> Dict[str, float]:
        """Score text quality across multiple dimensions"""
        
        scores = {}
        
        # Coherence: check for repetition and flow
        scores['coherence'] = self._score_coherence(text)
        
        # Fluency: basic grammar and structure
        scores['fluency'] = self._score_fluency(text)
        
        # Relevance: how well it follows the prompt
        scores['relevance'] = self._score_relevance(text, prompt)
        
        # Length penalty
        scores['length'] = self._score_length(text)
        
        # Overall score
        scores['overall'] = (
            0.4 * scores['coherence'] +
            0.3 * scores['fluency'] +
            0.2 * scores['relevance'] +
            0.1 * scores['length']
        )
        
        return scores
    
    def _score_coherence(self, text: str) -> float:
        """Score text coherence"""
        words = text.lower().split()
        if len(words) < 3:
            return 0.5
        
        # Check for excessive repetition
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        max_count = max(word_counts.values())
        repetition_ratio = max_count / len(words)
        
        # Penalize high repetition
        coherence = max(0.0, 1.0 - repetition_ratio * 2)
        
        return min(1.0, coherence)
    
    def _score_fluency(self, text: str) -> float:
        """Score text fluency"""
        # Simple heuristics for fluency
        sentences = text.split('.')
        
        if not sentences:
            return 0.0
        
        # Check sentence lengths
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        if not sentence_lengths:
            return 0.0
        
        avg_length = np.mean(sentence_lengths)
        
        # Ideal sentence length is around 10-20 words
        if 5 <= avg_length <= 25:
            length_score = 1.0
        else:
            length_score = max(0.0, 1.0 - abs(avg_length - 15) / 15)
        
        return length_score
    
    def _score_relevance(self, text: str, prompt: str) -> float:
        """Score relevance to prompt"""
        prompt_words = set(prompt.lower().split())
        text_words = set(text.lower().split())
        
        if not prompt_words:
            return 0.5
        
        # Calculate word overlap
        overlap = len(prompt_words.intersection(text_words))
        relevance = overlap / len(prompt_words)
        
        return min(1.0, relevance * 2)  # Scale up
    
    def _score_length(self, text: str) -> float:
        """Score based on appropriate length"""
        words = text.split()
        word_count = len(words)
        
        # Ideal range is 20-100 words
        if 20 <= word_count <= 100:
            return 1.0
        elif word_count < 20:
            return word_count / 20
        else:
            return max(0.3, 100 / word_count)

class TreeBasedGenerator:
    """Tree-based generation with beam search and pruning"""
    
    def __init__(self, model: SafeRealModelWrapper, config: GenerationConfig):
        self.model = model
        self.config = config
        self.scorer = QualityScorer(model.tokenizer)
        
    def generate_tree(self, prompt: str) -> Dict[str, Any]:
        """Generate using tree-based exploration"""
        
        if not self.model.is_loaded:
            return self._error_result("Model not loaded")
        
        try:
            # Initialize root node
            root = GenerationNode(
                text=prompt,
                score=1.0,
                depth=0,
                tokens=self.model.tokenizer.encode(prompt)
            )
            
            # Build generation tree
            best_nodes = self._build_tree(root)
            
            # Select best path
            best_node = max(best_nodes, key=lambda x: x.score)
            
            return {
                'generated_text': best_node.text,
                'prompt': prompt,
                'new_text': best_node.text[len(prompt):].strip(),
                'generation_method': 'tree_based',
                'final_score': best_node.score,
                'tree_depth': best_node.depth,
                'nodes_explored': self._count_nodes(root),
                'generation_successful': True
            }
            
        except Exception as e:
            logger.error(f"Tree generation failed: {e}")
            return self._error_result(str(e))
    
    def _build_tree(self, root: GenerationNode) -> List[GenerationNode]:
        """Build generation tree using beam search"""
        
        active_nodes = [root]
        completed_nodes = []
        
        for depth in range(self.config.tree_depth):
            next_nodes = []
            
            for node in active_nodes:
                if self._should_stop_generation(node):
                    completed_nodes.append(node)
                    continue
                
                # Generate children
                children = self._generate_children(node)
                node.children = children
                
                # Add to next level
                next_nodes.extend(children)
            
            # Beam search: keep only top nodes
            if next_nodes:
                next_nodes.sort(key=lambda x: x.score, reverse=True)
                active_nodes = next_nodes[:self.config.beam_size]
            else:
                break
        
        # Add remaining active nodes to completed
        completed_nodes.extend(active_nodes)
        
        return completed_nodes
    
    def _generate_children(self, parent: GenerationNode) -> List[GenerationNode]:
        """Generate child nodes from parent"""
        
        children = []
        
        # Get current text
        current_text = parent.text
        
        # Generate multiple continuations
        for i in range(self.config.branching_factor):
            try:
                # Slightly different parameters for diversity
                temp = self.config.temperature * (1.0 + 0.1 * i)
                
                result = self.model.generate(
                    current_text,
                    max_length=len(current_text.split()) + 20,
                    temperature=temp,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True
                )
                
                if result['generation_successful']:
                    child_text = result['generated_text']
                    
                    # Score the child
                    scores = self.scorer.score_text(child_text, parent.text)
                    
                    child = GenerationNode(
                        text=child_text,
                        score=scores['overall'],
                        depth=parent.depth + 1,
                        parent=parent,
                        tokens=self.model.tokenizer.encode(child_text)
                    )
                    
                    children.append(child)
                    
            except Exception as e:
                logger.warning(f"Failed to generate child {i}: {e}")
                continue
        
        return children
    
    def _should_stop_generation(self, node: GenerationNode) -> bool:
        """Check if generation should stop at this node"""
        
        # Stop if text is getting too long
        if len(node.text.split()) > self.config.max_length:
            return True
        
        # Stop if score is too low
        if node.score < self.config.quality_threshold:
            return True
        
        # Stop if we hit depth limit
        if node.depth >= self.config.tree_depth:
            return True
        
        return False
    
    def _count_nodes(self, root: GenerationNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in root.children:
            count += self._count_nodes(child)
        return count
    
    def _error_result(self, error: str) -> Dict[str, Any]:
        """Return error result"""
        return {
            'generated_text': f"[Tree generation failed: {error}]",
            'prompt': "",
            'new_text': "",
            'generation_method': 'tree_based',
            'generation_successful': False,
            'error': error
        }

class AdaptiveGenerator:
    """Adaptive generation that learns and improves"""
    
    def __init__(self, model: SafeRealModelWrapper, config: GenerationConfig):
        self.model = model
        self.config = config
        self.scorer = QualityScorer(model.tokenizer)
        self.parameter_history = deque(maxlen=100)
        self.score_history = deque(maxlen=100)
        
    def generate_adaptive(self, prompt: str) -> Dict[str, Any]:
        """Generate using adaptive parameters"""
        
        if not self.model.is_loaded:
            return self._error_result("Model not loaded")
        
        try:
            # Adapt parameters based on history
            adapted_params = self._adapt_parameters()
            
            # Generate with adapted parameters
            result = self.model.generate(
                prompt,
                max_length=self.config.max_length,
                **adapted_params
            )
            
            if result['generation_successful']:
                # Score the result
                scores = self.scorer.score_text(result['generated_text'], prompt)
                
                # Update history
                self._update_history(adapted_params, scores['overall'])
                
                # Add adaptive info to result
                result.update({
                    'generation_method': 'adaptive',
                    'adapted_params': adapted_params,
                    'quality_scores': scores,
                    'adaptation_count': len(self.parameter_history)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive generation failed: {e}")
            return self._error_result(str(e))
    
    def _adapt_parameters(self) -> Dict[str, Any]:
        """Adapt generation parameters based on history"""
        
        base_params = {
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'do_sample': True
        }
        
        if len(self.score_history) < 5:
            return base_params
        
        # Analyze recent performance
        recent_scores = list(self.score_history)[-10:]
        avg_score = np.mean(recent_scores)
        score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        # Adapt based on performance
        adapted_params = base_params.copy()
        
        if avg_score < 0.6:  # Low quality
            # Be more conservative
            adapted_params['temperature'] = max(0.3, self.config.temperature * 0.8)
            adapted_params['top_p'] = min(0.95, self.config.top_p + 0.1)
        elif avg_score > 0.8:  # High quality
            # Be more creative
            adapted_params['temperature'] = min(1.5, self.config.temperature * 1.2)
            adapted_params['top_p'] = max(0.7, self.config.top_p - 0.1)
        
        if score_trend < 0:  # Declining performance
            # Adjust parameters more aggressively
            adapted_params['temperature'] = self.config.temperature * random.uniform(0.7, 1.3)
            adapted_params['top_k'] = max(20, min(100, int(self.config.top_k * random.uniform(0.8, 1.2))))
        
        return adapted_params
    
    def _update_history(self, params: Dict[str, Any], score: float):
        """Update parameter and score history"""
        self.parameter_history.append(params.copy())
        self.score_history.append(score)
    
    def _error_result(self, error: str) -> Dict[str, Any]:
        """Return error result"""
        return {
            'generated_text': f"[Adaptive generation failed: {error}]",
            'prompt': "",
            'new_text': "",
            'generation_method': 'adaptive',
            'generation_successful': False,
            'error': error
        }

class BacktrackingGenerator:
    """Generator with backtracking capability"""
    
    def __init__(self, model: SafeRealModelWrapper, config: GenerationConfig):
        self.model = model
        self.config = config
        self.scorer = QualityScorer(model.tokenizer)
        
    def generate_with_backtrack(self, prompt: str) -> Dict[str, Any]:
        """Generate with backtracking on low quality"""
        
        if not self.model.is_loaded:
            return self._error_result("Model not loaded")
        
        try:
            best_result = None
            best_score = 0.0
            backtrack_count = 0
            generation_history = []
            
            for attempt in range(self.config.max_backtracks + 1):
                # Generate
                result = self.model.generate(
                    prompt,
                    max_length=self.config.max_length,
                    temperature=self.config.temperature * (1.0 + 0.1 * attempt),
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True
                )
                
                if not result['generation_successful']:
                    continue
                
                # Score the result
                scores = self.scorer.score_text(result['generated_text'], prompt)
                current_score = scores['overall']
                
                generation_history.append({
                    'attempt': attempt,
                    'score': current_score,
                    'text_length': len(result['generated_text'])
                })
                
                # Check if this is the best so far
                if current_score > best_score:
                    best_result = result
                    best_score = current_score
                
                # Check if we should stop (good enough)
                if current_score >= self.config.backtrack_threshold:
                    break
                
                # Decide if we should backtrack
                if attempt < self.config.max_backtracks and current_score < self.config.backtrack_threshold:
                    backtrack_count += 1
                    logger.debug(f"Backtracking attempt {attempt + 1}, score: {current_score:.3f}")
                    continue
                
                break
            
            if best_result:
                best_result.update({
                    'generation_method': 'backtracking',
                    'final_score': best_score,
                    'backtrack_count': backtrack_count,
                    'generation_history': generation_history
                })
                return best_result
            else:
                return self._error_result("All backtracking attempts failed")
                
        except Exception as e:
            logger.error(f"Backtracking generation failed: {e}")
            return self._error_result(str(e))
    
    def _error_result(self, error: str) -> Dict[str, Any]:
        """Return error result"""
        return {
            'generated_text': f"[Backtracking generation failed: {error}]",
            'prompt': "",
            'new_text': "",
            'generation_method': 'backtracking',
            'generation_successful': False,
            'error': error
        }

class ExtendedGenerationWrapper:
    """Wrapper that extends SafeRealModelWrapper with advanced generation methods"""
    
    def __init__(self, base_model: SafeRealModelWrapper, config: GenerationConfig = None):
        self.base_model = base_model
        self.config = config or GenerationConfig()
        
        # Initialize generators
        self.tree_generator = TreeBasedGenerator(base_model, self.config)
        self.adaptive_generator = AdaptiveGenerator(base_model, self.config)
        self.backtrack_generator = BacktrackingGenerator(base_model, self.config)
        
        # Method selection statistics
        self.method_stats = defaultdict(lambda: {'count': 0, 'avg_score': 0.0})
        
    def generate(self, prompt: str, method: str = "auto", **kwargs) -> Dict[str, Any]:
        """Generate using specified or automatically selected method"""
        
        # Update config with kwargs
        temp_config = copy.deepcopy(self.config)
        for key, value in kwargs.items():
            if hasattr(temp_config, key):
                setattr(temp_config, key, value)
        
        # Select method
        if method == "auto":
            method = self._select_best_method(prompt)
        
        # Generate using selected method
        start_time = time.time()
        
        if method == "tree":
            result = self.tree_generator.generate_tree(prompt)
        elif method == "adaptive":
            result = self.adaptive_generator.generate_adaptive(prompt)
        elif method == "backtrack":
            result = self.backtrack_generator.generate_with_backtrack(prompt)
        elif method == "basic":
            result = self.base_model.generate(prompt, **kwargs)
            result['generation_method'] = 'basic'
        else:
            # Fallback to basic
            result = self.base_model.generate(prompt, **kwargs)
            result['generation_method'] = 'basic'
        
        # Add timing and method info
        result['generation_time'] = time.time() - start_time
        result['selected_method'] = method
        
        # Update statistics
        if 'final_score' in result or 'quality_scores' in result:
            score = result.get('final_score', result.get('quality_scores', {}).get('overall', 0.0))
            self._update_method_stats(method, score)
        
        return result
    
    def _select_best_method(self, prompt: str) -> str:
        """Automatically select the best generation method"""
        
        # Simple heuristics for method selection
        prompt_length = len(prompt.split())
        
        # For mathematical or technical prompts, use adaptive
        if any(word in prompt.lower() for word in ['calculate', 'solve', 'equation', 'formula', 'code', 'function']):
            return "adaptive"
        
        # For creative prompts, use tree-based for exploration
        if any(word in prompt.lower() for word in ['story', 'creative', 'write', 'imagine', 'describe']):
            return "tree"
        
        # For long prompts, use backtracking for quality
        if prompt_length > 20:
            return "backtrack"
        
        # Default to adaptive for balanced performance
        return "adaptive"
    
    def _update_method_stats(self, method: str, score: float):
        """Update method performance statistics"""
        stats = self.method_stats[method]
        stats['count'] += 1
        stats['avg_score'] = (stats['avg_score'] * (stats['count'] - 1) + score) / stats['count']
    
    def get_method_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for each method"""
        return dict(self.method_stats)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get extended model information"""
        base_info = self.base_model.get_model_info()
        base_info.update({
            'extended_generation': True,
            'available_methods': ['basic', 'tree', 'adaptive', 'backtrack', 'auto'],
            'method_stats': self.get_method_stats(),
            'config': asdict(self.config)
        })
        return base_info

def create_extended_models_from_safe(safe_models: List, config: GenerationConfig = None) -> List:
    """Convert safe models to extended generation models"""
    
    extended_models = []
    generation_config = config or GenerationConfig()
    
    for model_spec in safe_models:
        # Create extended wrapper
        extended_wrapper = ExtendedGenerationWrapper(model_spec.model, generation_config)
        
        # Create new model spec with extended wrapper
        extended_spec = copy.deepcopy(model_spec)
        extended_spec.model = extended_wrapper
        extended_spec.description += " (Extended Generation)"
        
        extended_models.append(extended_spec)
    
    return extended_models

# Example usage and testing
if __name__ == "__main__":
    from utils.safe_real_models import create_small_safe_models, SafeRealModelConfig, ModelSize
    
    print("🧪 Testing Extended Generation Methods")
    
    # Create base models
    config = SafeRealModelConfig(model_size=ModelSize.SMALL, device="cpu")
    safe_models = create_small_safe_models()
    
    if safe_models:
        # Test extended generation
        model_spec = safe_models[0]
        
        # Create extended wrapper
        gen_config = GenerationConfig(
            max_length=100,
            tree_depth=2,
            branching_factor=2,
            enable_backtrack=True
        )
        extended_model = ExtendedGenerationWrapper(model_spec.model, gen_config)
        
        # Test different methods
        test_prompt = "Write a short story about AI:"
        
        methods = ["basic", "tree", "adaptive", "backtrack", "auto"]
        
        for method in methods:
            print(f"\n🔧 Testing {method} method...")
            result = extended_model.generate(test_prompt, method=method, max_length=80)
            
            if result['generation_successful']:
                print(f"✅ Method: {result['selected_method']}")
                print(f"📝 Text: {result['new_text'][:100]}...")
                print(f"⏱️ Time: {result['generation_time']:.3f}s")
                if 'final_score' in result:
                    print(f"📊 Score: {result['final_score']:.3f}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Show statistics
        print(f"\n📈 Method Statistics:")
        stats = extended_model.get_method_stats()
        for method, stat in stats.items():
            print(f"   {method}: {stat['count']} uses, avg score: {stat['avg_score']:.3f}")
    
    else:
        print("❌ No safe models available for testing")