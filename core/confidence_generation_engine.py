# File: core/confidence_generation_engine.py
"""
Generation Engine with Real-time Confidence Scoring
Integrates PRM/ORM for dynamic confidence estimation during generation
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    from .reward_models import ProcessRewardModel, OutcomeRewardModel, HybridRewardModel, RewardScore
except ImportError:
    # Fallback for standalone usage
    from reward_models import ProcessRewardModel, OutcomeRewardModel, HybridRewardModel, RewardScore

class GenerationStrategy(Enum):
    """Generation strategies with different confidence approaches"""
    GREEDY = "greedy"                    # Simple greedy decoding
    BEAM_SEARCH = "beam_search"          # Beam search with confidence
    GUIDED_SAMPLING = "guided_sampling"  # PRM-guided token sampling
    BEST_OF_N = "best_of_n"             # Generate N, pick best by ORM
    TREE_SEARCH = "tree_search"          # Tree search with PRM guidance
    ADAPTIVE = "adaptive"                # Adaptive strategy selection

@dataclass
class GenerationConfig:
    """Configuration for confident generation"""
    
    # Basic generation parameters
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Strategy and search parameters
    strategy: GenerationStrategy = GenerationStrategy.BEAM_SEARCH
    num_beams: int = 4
    num_candidates: int = 8
    
    # Reward model configuration
    use_prm: bool = True
    use_orm: bool = True
    prm_weight: float = 0.4
    orm_weight: float = 0.6
    
    # Confidence thresholds
    min_confidence: float = 0.3
    target_confidence: float = 0.7
    confidence_penalty: float = 0.1
    
    # Search control
    early_stopping: bool = True
    diversity_penalty: float = 0.1
    reward_alpha: float = 0.3
    
    # Tree search specific
    tree_depth: int = 3
    branching_factor: int = 2
    
    # Adaptive parameters
    confidence_window: int = 10
    adaptation_threshold: float = 0.1

@dataclass
class GenerationStep:
    """Information about a single generation step"""
    
    token_id: int
    token_text: str
    logit: float
    probability: float
    cumulative_probability: float
    prm_score: float
    confidence: float
    reasoning_quality: float
    step_index: int
    alternatives: List[Tuple[int, float]]  # (token_id, score) alternatives

@dataclass
class GenerationResult:
    """Complete generation result with confidence analysis"""
    
    # Generated content
    generated_text: str
    tokens_generated: int
    
    # Confidence metrics
    overall_confidence: float
    step_confidences: List[float]
    prm_scores: List[float]
    orm_score: float
    
    # Generation metadata
    strategy_used: str
    generation_time: float
    num_candidates_explored: int
    
    # Detailed analysis
    generation_steps: List[GenerationStep]
    reward_breakdown: Dict[str, float]
    quality_metrics: Dict[str, float]
    
    # Success indicators
    generation_successful: bool
    met_confidence_threshold: bool
    early_stopped: bool

class ConfidenceTracker:
    """Tracks confidence during generation process"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.confidence_history = []
        self.prm_history = []
        self.quality_history = []
    
    def add_step(self, confidence: float, prm_score: float, quality: float):
        """Add confidence information for a generation step"""
        
        self.confidence_history.append(confidence)
        self.prm_history.append(prm_score)
        self.quality_history.append(quality)
        
        # Keep only recent history
        if len(self.confidence_history) > self.window_size:
            self.confidence_history.pop(0)
            self.prm_history.pop(0)
            self.quality_history.pop(0)
    
    def get_trend(self) -> Dict[str, float]:
        """Get confidence trend analysis"""
        
        if len(self.confidence_history) < 2:
            return {'trend': 0.0, 'stability': 1.0, 'recent_average': 0.5}
        
        # Calculate trend (slope of recent confidence)
        recent_conf = self.confidence_history[-min(5, len(self.confidence_history)):]
        if len(recent_conf) > 1:
            x = np.arange(len(recent_conf))
            trend = np.polyfit(x, recent_conf, 1)[0]  # Linear slope
        else:
            trend = 0.0
        
        # Calculate stability (inverse of variance)
        stability = 1.0 / (1.0 + np.var(self.confidence_history))
        
        # Recent average
        recent_average = np.mean(recent_conf)
        
        return {
            'trend': trend,
            'stability': stability,
            'recent_average': recent_average,
            'min_confidence': min(self.confidence_history),
            'max_confidence': max(self.confidence_history)
        }
    
    def should_continue(self, config: GenerationConfig) -> bool:
        """Determine if generation should continue based on confidence"""
        
        if not self.confidence_history:
            return True
        
        trend_info = self.get_trend()
        
        # Stop if confidence is consistently low
        if (trend_info['recent_average'] < config.min_confidence and 
            trend_info['trend'] < 0):
            return False
        
        # Stop if confidence is high and stable
        if (trend_info['recent_average'] > config.target_confidence and
            trend_info['stability'] > 0.8):
            return config.early_stopping == False  # Continue only if early stopping disabled
        
        return True

class ConfidentGenerationEngine:
    """
    Generation engine with integrated confidence scoring
    Uses PRM/ORM models for real-time quality assessment
    """
    
    def __init__(self, base_model, prm: ProcessRewardModel, orm: OutcomeRewardModel, 
                 tokenizer, hybrid_model: Optional[HybridRewardModel] = None):
        
        self.base_model = base_model
        self.prm = prm
        self.orm = orm
        self.tokenizer = tokenizer
        self.hybrid_model = hybrid_model or HybridRewardModel(prm, orm)
        
        # Generation state
        self.generation_history = []
        self.confidence_cache = {}
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> GenerationResult:
        """Generate text with confidence scoring"""
        
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        try:
            # Route to appropriate generation strategy
            if config.strategy == GenerationStrategy.GREEDY:
                result = self._greedy_generation(prompt, config)
            elif config.strategy == GenerationStrategy.BEAM_SEARCH:
                result = self._beam_search_generation(prompt, config)
            elif config.strategy == GenerationStrategy.GUIDED_SAMPLING:
                result = self._guided_sampling_generation(prompt, config)
            elif config.strategy == GenerationStrategy.BEST_OF_N:
                result = self._best_of_n_generation(prompt, config)
            elif config.strategy == GenerationStrategy.TREE_SEARCH:
                result = self._tree_search_generation(prompt, config)
            elif config.strategy == GenerationStrategy.ADAPTIVE:
                result = self._adaptive_generation(prompt, config)
            else:
                # Fallback to beam search
                result = self._beam_search_generation(prompt, config)
            
            # Add timing information
            result.generation_time = time.time() - start_time
            
            # Compute final ORM score
            final_orm_score = self.orm.compute_outcome_reward(prompt, result.generated_text, self.tokenizer)
            result.orm_score = final_orm_score.score
            
            # Update quality metrics
            result.quality_metrics.update({
                'final_orm_confidence': final_orm_score.confidence,
                'text_length_score': min(len(result.generated_text.split()) / 100, 1.0),
                'coherence_score': self._estimate_coherence(result.generated_text)
            })
            
            return result
            
        except Exception as e:
            # Return error result
            return GenerationResult(
                generated_text=f"[Generation Error: {str(e)}]",
                tokens_generated=0,
                overall_confidence=0.1,
                step_confidences=[],
                prm_scores=[],
                orm_score=0.1,
                strategy_used=config.strategy.value,
                generation_time=time.time() - start_time,
                num_candidates_explored=0,
                generation_steps=[],
                reward_breakdown={'error': 1.0},
                quality_metrics={'error': str(e)},
                generation_successful=False,
                met_confidence_threshold=False,
                early_stopped=True
            )
    
    def _beam_search_generation(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Beam search with confidence-guided selection"""
        
        # Tokenize prompt
        input_tokens = self.tokenizer(prompt, return_tensors='pt')
        
        # Initialize beam search state
        beams = [(input_tokens['input_ids'], 0.0, [], [])]  # (tokens, score, steps, confidences)
        confidence_tracker = ConfidenceTracker()
        
        generated_steps = []
        candidates_explored = 0
        
        for step in range(config.max_length):
            new_beams = []
            
            for beam_tokens, beam_score, beam_steps, beam_confidences in beams:
                # Get next token probabilities
                with torch.no_grad():
                    outputs = self.base_model(beam_tokens)
                    logits = outputs.logits[0, -1, :]  # Last position logits
                
                # Apply temperature
                scaled_logits = logits / config.temperature
                probs = F.softmax(scaled_logits, dim=-1)
                
                # Get top-k candidates
                top_probs, top_indices = torch.topk(probs, config.top_k)
                
                for i in range(min(config.top_k, len(top_indices))):
                    token_id = top_indices[i].item()
                    token_prob = top_probs[i].item()
                    token_text = self.tokenizer.decode([token_id])
                    
                    # Create new sequence
                    new_tokens = torch.cat([beam_tokens, torch.tensor([[token_id]])], dim=1)
                    new_text = self.tokenizer.decode(new_tokens[0])
                    
                    # Calculate PRM score for this step
                    prm_score = self._calculate_step_prm_score(new_text, config)
                    
                    # Calculate confidence
                    step_confidence = self._calculate_step_confidence(
                        token_prob, prm_score, beam_confidences
                    )
                    
                    # Create generation step
                    gen_step = GenerationStep(
                        token_id=token_id,
                        token_text=token_text,
                        logit=scaled_logits[token_id].item(),
                        probability=token_prob,
                        cumulative_probability=beam_score + np.log(token_prob),
                        prm_score=prm_score,
                        confidence=step_confidence,
                        reasoning_quality=prm_score,
                        step_index=step,
                        alternatives=[(top_indices[j].item(), top_probs[j].item()) 
                                    for j in range(min(3, len(top_indices)))]
                    )
                    
                    # Calculate beam score with confidence
                    new_score = (beam_score + np.log(token_prob) + 
                               config.reward_alpha * prm_score)
                    
                    # Add to new beams
                    new_beams.append((
                        new_tokens,
                        new_score,
                        beam_steps + [gen_step],
                        beam_confidences + [step_confidence]
                    ))
                    
                    candidates_explored += 1
            
            # Keep top beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:config.num_beams]
            
            # Update confidence tracker with best beam
            if beams:
                best_confidences = beams[0][3]
                if best_confidences:
                    confidence_tracker.add_step(
                        best_confidences[-1],
                        beams[0][2][-1].prm_score if beams[0][2] else 0.5,
                        beams[0][2][-1].reasoning_quality if beams[0][2] else 0.5
                    )
            
            # Early stopping check
            if config.early_stopping and not confidence_tracker.should_continue(config):
                break
            
            # Check for EOS token
            if beams and beams[0][0][0, -1].item() == self.tokenizer.eos_token_id:
                break
        
        # Select best beam
        if beams:
            best_tokens, best_score, best_steps, best_confidences = beams[0]
            generated_text = self.tokenizer.decode(best_tokens[0], skip_special_tokens=True)
            
            # Remove prompt from generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
        else:
            generated_text = ""
            best_steps = []
            best_confidences = []
        
        # Calculate overall confidence
        overall_confidence = np.mean(best_confidences) if best_confidences else 0.3
        
        # Create result
        return GenerationResult(
            generated_text=generated_text,
            tokens_generated=len(best_steps),
            overall_confidence=overall_confidence,
            step_confidences=best_confidences,
            prm_scores=[step.prm_score for step in best_steps],
            orm_score=0.0,  # Will be computed later
            strategy_used="beam_search_with_confidence",
            generation_time=0.0,  # Will be set by caller
            num_candidates_explored=candidates_explored,
            generation_steps=best_steps,
            reward_breakdown={
                'average_prm': np.mean([step.prm_score for step in best_steps]) if best_steps else 0.5,
                'confidence_trend': confidence_tracker.get_trend()['trend'],
                'stability': confidence_tracker.get_trend()['stability']
            },
            quality_metrics={
                'token_diversity': len(set(step.token_id for step in best_steps)) / max(len(best_steps), 1),
                'avg_probability': np.mean([step.probability for step in best_steps]) if best_steps else 0.5
            },
            generation_successful=len(generated_text.strip()) > 0,
            met_confidence_threshold=overall_confidence >= config.target_confidence,
            early_stopped=len(best_steps) < config.max_length
        )
    
    def _guided_sampling_generation(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """PRM-guided sampling generation"""
        
        input_tokens = self.tokenizer(prompt, return_tensors='pt')
        current_tokens = input_tokens['input_ids']
        
        generated_steps = []
        confidences = []
        confidence_tracker = ConfidenceTracker()
        candidates_explored = 0
        
        for step in range(config.max_length):
            # Get next token distribution
            with torch.no_grad():
                outputs = self.base_model(current_tokens)
                logits = outputs.logits[0, -1, :]
            
            # Apply temperature and get probabilities
            scaled_logits = logits / config.temperature
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Sample multiple candidates
            top_probs, top_indices = torch.topk(probs, min(config.top_k, len(probs)))
            
            # Evaluate each candidate with PRM
            candidate_scores = []
            for i in range(len(top_indices)):
                token_id = top_indices[i].item()
                token_prob = top_probs[i].item()
                
                # Create candidate sequence
                candidate_tokens = torch.cat([current_tokens, torch.tensor([[token_id]])], dim=1)
                candidate_text = self.tokenizer.decode(candidate_tokens[0])
                
                # Get PRM score
                prm_score = self._calculate_step_prm_score(candidate_text, config)
                
                # Combined score (probability + PRM)
                combined_score = token_prob + config.reward_alpha * prm_score
                candidate_scores.append((token_id, token_prob, prm_score, combined_score))
                candidates_explored += 1
            
            # Select best candidate
            best_candidate = max(candidate_scores, key=lambda x: x[3])
            token_id, token_prob, prm_score, combined_score = best_candidate
            
            # Calculate step confidence
            step_confidence = self._calculate_step_confidence(token_prob, prm_score, confidences)
            
            # Create generation step
            token_text = self.tokenizer.decode([token_id])
            gen_step = GenerationStep(
                token_id=token_id,
                token_text=token_text,
                logit=scaled_logits[token_id].item(),
                probability=token_prob,
                cumulative_probability=sum(s.probability for s in generated_steps) + token_prob,
                prm_score=prm_score,
                confidence=step_confidence,
                reasoning_quality=prm_score,
                step_index=step,
                alternatives=[(c[0], c[3]) for c in candidate_scores[:3]]
            )
            
            generated_steps.append(gen_step)
            confidences.append(step_confidence)
            
            # Update tokens
            current_tokens = torch.cat([current_tokens, torch.tensor([[token_id]])], dim=1)
            
            # Update confidence tracker
            confidence_tracker.add_step(step_confidence, prm_score, prm_score)
            
            # Early stopping checks
            if config.early_stopping and not confidence_tracker.should_continue(config):
                break
            
            if token_id == getattr(self.tokenizer, 'eos_token_id', -1):
                break
        
        # Generate final text
        generated_text = self.tokenizer.decode(current_tokens[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidences) if confidences else 0.3
        
        return GenerationResult(
            generated_text=generated_text,
            tokens_generated=len(generated_steps),
            overall_confidence=overall_confidence,
            step_confidences=confidences,
            prm_scores=[step.prm_score for step in generated_steps],
            orm_score=0.0,
            strategy_used="prm_guided_sampling",
            generation_time=0.0,
            num_candidates_explored=candidates_explored,
            generation_steps=generated_steps,
            reward_breakdown={
                'average_prm': np.mean([step.prm_score for step in generated_steps]),
                'prm_weighted_score': np.mean([step.prm_score * step.probability for step in generated_steps])
            },
            quality_metrics={
                'prm_consistency': np.std([step.prm_score for step in generated_steps]),
                'confidence_growth': confidences[-1] - confidences[0] if len(confidences) > 1 else 0.0
            },
            generation_successful=len(generated_text.strip()) > 0,
            met_confidence_threshold=overall_confidence >= config.target_confidence,
            early_stopped=len(generated_steps) < config.max_length
        )
    
    def _best_of_n_generation(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate N candidates and select best by ORM"""
        
        candidates = []
        total_candidates_explored = 0
        
        # Generate multiple candidates
        for i in range(config.num_candidates):
            # Use simpler generation for candidates
            simple_config = GenerationConfig(
                max_length=config.max_length,
                temperature=config.temperature + i * 0.1,  # Vary temperature
                strategy=GenerationStrategy.GREEDY,
                use_prm=False,  # Skip PRM for speed
                early_stopping=False
            )
            
            candidate_result = self._greedy_generation(prompt, simple_config)
            candidates.append(candidate_result)
            total_candidates_explored += candidate_result.num_candidates_explored
        
        # Evaluate all candidates with ORM
        best_candidate = None
        best_orm_score = -1.0
        
        for candidate in candidates:
            if candidate.generation_successful:
                orm_result = self.orm.compute_outcome_reward(prompt, candidate.generated_text, self.tokenizer)
                candidate.orm_score = orm_result.score
                
                if orm_result.score > best_orm_score:
                    best_orm_score = orm_result.score
                    best_candidate = candidate
        
        if best_candidate is None:
            # Fallback to first candidate
            best_candidate = candidates[0] if candidates else self._greedy_generation(prompt, config)
        
        # Update metadata
        best_candidate.strategy_used = "best_of_n_orm_selection"
        best_candidate.num_candidates_explored = total_candidates_explored
        best_candidate.reward_breakdown.update({
            'candidates_generated': len(candidates),
            'best_orm_score': best_orm_score,
            'orm_selection_used': True
        })
        
        return best_candidate
    
    def _greedy_generation(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Simple greedy generation with basic confidence"""
        
        input_tokens = self.tokenizer(prompt, return_tensors='pt')
        current_tokens = input_tokens['input_ids']
        
        generated_steps = []
        confidences = []
        
        for step in range(config.max_length):
            with torch.no_grad():
                outputs = self.base_model(current_tokens)
                logits = outputs.logits[0, -1, :]
            
            # Greedy selection
            next_token_id = torch.argmax(logits).item()
            next_token_prob = F.softmax(logits / config.temperature, dim=-1)[next_token_id].item()
            
            # Basic confidence (just probability)
            step_confidence = min(next_token_prob * 1.5, 1.0)  # Boost for confidence
            
            # Create step
            token_text = self.tokenizer.decode([next_token_id])
            gen_step = GenerationStep(
                token_id=next_token_id,
                token_text=token_text,
                logit=logits[next_token_id].item(),
                probability=next_token_prob,
                cumulative_probability=sum(s.probability for s in generated_steps) + next_token_prob,
                prm_score=0.5,  # Neutral for greedy
                confidence=step_confidence,
                reasoning_quality=0.5,
                step_index=step,
                alternatives=[]
            )
            
            generated_steps.append(gen_step)
            confidences.append(step_confidence)
            
            # Update tokens
            current_tokens = torch.cat([current_tokens, torch.tensor([[next_token_id]])], dim=1)
            
            # Stop on EOS
            if next_token_id == getattr(self.tokenizer, 'eos_token_id', -1):
                break
        
        # Generate text
        generated_text = self.tokenizer.decode(current_tokens[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return GenerationResult(
            generated_text=generated_text,
            tokens_generated=len(generated_steps),
            overall_confidence=np.mean(confidences) if confidences else 0.5,
            step_confidences=confidences,
            prm_scores=[0.5] * len(generated_steps),
            orm_score=0.0,
            strategy_used="greedy_generation",
            generation_time=0.0,
            num_candidates_explored=len(generated_steps),
            generation_steps=generated_steps,
            reward_breakdown={'greedy_selection': 1.0},
            quality_metrics={'avg_probability': np.mean(confidences)},
            generation_successful=len(generated_text.strip()) > 0,
            met_confidence_threshold=np.mean(confidences) >= config.target_confidence if confidences else False,
            early_stopped=False
        )
    
    def _tree_search_generation(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Tree search with PRM guidance (simplified)"""
        # For now, fallback to beam search
        return self._beam_search_generation(prompt, config)
    
    def _adaptive_generation(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Adaptive strategy selection based on prompt analysis"""
        
        # Analyze prompt to select best strategy
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['solve', 'calculate', 'equation']):
            # Mathematical problems benefit from guided sampling
            config.strategy = GenerationStrategy.GUIDED_SAMPLING
        elif any(word in prompt_lower for word in ['story', 'creative', 'write']):
            # Creative tasks benefit from best-of-n
            config.strategy = GenerationStrategy.BEST_OF_N
        elif any(word in prompt_lower for word in ['analyze', 'compare', 'evaluate']):
            # Reasoning tasks benefit from beam search
            config.strategy = GenerationStrategy.BEAM_SEARCH
        else:
            # Default to beam search
            config.strategy = GenerationStrategy.BEAM_SEARCH
        
        # Recursively call with selected strategy
        result = self.generate(prompt, config)
        result.strategy_used = f"adaptive_{result.strategy_used}"
        
        return result
    
    def _calculate_step_prm_score(self, text: str, config: GenerationConfig) -> float:
        """Calculate PRM score for a generation step"""
        
        if not config.use_prm:
            return 0.5
        
        try:
            # Use cached score if available
            cache_key = hash(text[-100:])  # Use last 100 chars as key
            if cache_key in self.confidence_cache:
                return self.confidence_cache[cache_key]
            
            # Tokenize and get PRM score
            tokens = self.tokenizer(text, return_tensors='pt', max_length=100, truncation=True)
            
            with torch.no_grad():
                prm_output = self.prm.forward(tokens['input_ids'], tokens.get('attention_mask'))
                score = prm_output['step_quality'].item()
            
            # Cache result
            self.confidence_cache[cache_key] = score
            
            return score
            
        except Exception:
            return 0.5  # Neutral score on error
    
    def _calculate_step_confidence(self, token_prob: float, prm_score: float, 
                                 previous_confidences: List[float]) -> float:
        """Calculate confidence for a generation step"""
        
        # Base confidence from token probability
        prob_confidence = min(token_prob * 2, 1.0)  # Boost probability
        
        # PRM-based confidence
        prm_confidence = prm_score
        
        # Historical trend
        if previous_confidences:
            trend_confidence = np.mean(previous_confidences[-3:])  # Last 3 steps
        else:
            trend_confidence = 0.5
        
        # Weighted combination
        combined_confidence = (
            0.4 * prob_confidence +
            0.4 * prm_confidence +
            0.2 * trend_confidence
        )
        
        return max(0.1, min(0.95, combined_confidence))
    
    def _estimate_coherence(self, text: str) -> float:
        """Estimate text coherence (simple heuristic)"""
        
        if not text:
            return 0.0
        
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.8  # Single sentence is coherent
        
        # Simple coherence based on sentence length variation
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5
        
        # Penalize extreme length variations
        length_std = np.std(lengths) / max(np.mean(lengths), 1)
        coherence = max(0.3, 1.0 - length_std * 0.5)
        
        return min(coherence, 1.0)

def main():
    """Test confidence generation engine"""
    
    print("🧪 Testing Confidence Generation Engine")
    print("=" * 40)
    
    # Mock models for testing
    class MockModel:
        def __call__(self, input_ids):
            # Return mock logits
            vocab_size = 1000
            seq_len = input_ids.shape[1]
            logits = torch.randn(1, seq_len, vocab_size)
            
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            
            return MockOutput(logits)
    
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 2
        
        def __call__(self, text, **kwargs):
            words = text.split()[:50]
            token_ids = [hash(word) % 1000 for word in words]
            result = {'input_ids': torch.tensor([token_ids])}
            
            if kwargs.get('max_length'):
                token_ids = token_ids[:kwargs['max_length']]
                result['input_ids'] = torch.tensor([token_ids])
            
            return result
        
        def decode(self, token_ids, **kwargs):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], int):
                return f"token_{token_ids[-1] % 100}"
            return "generated_text"
    
    # Create mock models
    base_model = MockModel()
    prm = ProcessRewardModel(vocab_size=1000, hidden_size=64, num_layers=2)
    orm = OutcomeRewardModel(vocab_size=1000, hidden_size=64, num_layers=2)
    tokenizer = MockTokenizer()
    
    # Create generation engine
    engine = ConfidentGenerationEngine(base_model, prm, orm, tokenizer)
    
    # Test different strategies
    strategies = [
        GenerationStrategy.GREEDY,
        GenerationStrategy.BEAM_SEARCH,
        GenerationStrategy.GUIDED_SAMPLING,
        GenerationStrategy.BEST_OF_N
    ]
    
    test_prompt = "Solve the equation x² - 4x + 3 = 0"
    
    for strategy in strategies:
        print(f"\n🔬 Testing {strategy.value}...")
        
        config = GenerationConfig(
            strategy=strategy,
            max_length=20,
            num_candidates=3 if strategy == GenerationStrategy.BEST_OF_N else 8
        )
        
        result = engine.generate(test_prompt, config)
        
        print(f"   Generated: {result.generated_text}")
        print(f"   Confidence: {result.overall_confidence:.3f}")
        print(f"   Tokens: {result.tokens_generated}")
        print(f"   Successful: {result.generation_successful}")
        print(f"   Met threshold: {result.met_confidence_threshold}")
    
    print("\n✅ Confidence generation testing completed!")

if __name__ == "__main__":
    main()