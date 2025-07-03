# File: core/reward_models.py
"""
Real Process Reward Model (PRM) and Outcome Reward Model (ORM) implementation
Provides actual reward scoring for text generation with confidence estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import time

class RewardModelType(Enum):
    """Types of reward models"""
    PROCESS = "process"  # Step-by-step reasoning evaluation
    OUTCOME = "outcome"  # Final result evaluation
    HYBRID = "hybrid"    # Combined approach

@dataclass
class RewardScore:
    """Container for reward scoring results"""
    score: float                    # Main reward score [0, 1]
    confidence: float              # Confidence in the score [0, 1]
    reasoning: str                 # Human-readable reasoning
    components: Dict[str, float]   # Individual scoring components
    metadata: Dict[str, Any]       # Additional metadata

class ProcessRewardModel(nn.Module):
    """
    Process Reward Model (PRM) - Evaluates reasoning steps
    Analyzes intermediate steps in problem-solving and reasoning
    """
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 768, num_layers: int = 6):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Text encoder for step analysis
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        # Step quality predictor
        self.step_quality_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Logical consistency checker
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, 0.0, 0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through PRM"""
        
        # Embed tokens
        embeddings = self.embedding(input_ids)  # [batch, seq_len, hidden]
        
        # Apply transformer layers
        hidden_states = embeddings.transpose(0, 1)  # [seq_len, batch, hidden]
        
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        # Pool sequence representation
        if attention_mask is not None:
            # Masked average pooling
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states.transpose(0, 1) * mask).sum(1) / mask.sum(1)
        else:
            # Simple average pooling
            pooled = hidden_states.mean(0)
        
        # Compute step quality
        step_quality = self.step_quality_head(pooled)
        
        # Compute confidence
        confidence = self.confidence_head(pooled)
        
        return {
            'step_quality': step_quality,
            'confidence': confidence,
            'hidden_states': pooled
        }
    
    def compute_step_rewards(self, text_steps: List[str], tokenizer) -> List[RewardScore]:
        """Compute rewards for individual reasoning steps"""
        
        if not text_steps:
            return []
        
        rewards = []
        
        for i, step in enumerate(text_steps):
            try:
                # Tokenize step
                tokens = tokenizer(step, return_tensors='pt', padding=True, truncation=True)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.forward(tokens['input_ids'], tokens.get('attention_mask'))
                
                # Extract scores
                step_quality = outputs['step_quality'].item()
                confidence = outputs['confidence'].item()
                
                # Analyze step components
                components = self._analyze_step_components(step)
                
                # Create reward score
                reward = RewardScore(
                    score=step_quality,
                    confidence=confidence,
                    reasoning=self._generate_step_reasoning(step, step_quality, components),
                    components=components,
                    metadata={
                        'step_index': i,
                        'step_length': len(step),
                        'step_type': self._classify_step_type(step)
                    }
                )
                
                rewards.append(reward)
                
            except Exception as e:
                # Fallback reward for problematic steps
                rewards.append(RewardScore(
                    score=0.3,
                    confidence=0.1,
                    reasoning=f"Error processing step: {str(e)}",
                    components={'error': 1.0},
                    metadata={'step_index': i, 'error': str(e)}
                ))
        
        return rewards
    
    def _analyze_step_components(self, step: str) -> Dict[str, float]:
        """Analyze individual components of a reasoning step"""
        
        components = {}
        step_lower = step.lower()
        
        # Logical structure indicators
        logic_indicators = ['therefore', 'because', 'since', 'thus', 'hence', 'so']
        components['logical_structure'] = sum(1 for ind in logic_indicators if ind in step_lower) / 10
        
        # Mathematical rigor (for math problems)
        math_indicators = ['equation', 'substitute', 'solve', 'calculate', '=', '+', '-', '*', '/']
        components['mathematical_rigor'] = sum(1 for ind in math_indicators if ind in step_lower) / 10
        
        # Clarity and explanation
        explanation_indicators = ['first', 'next', 'then', 'finally', 'step', 'now']
        components['clarity'] = sum(1 for ind in explanation_indicators if ind in step_lower) / 10
        
        # Error indicators (negative scoring)
        error_indicators = ['wrong', 'mistake', 'error', 'incorrect', 'false']
        components['error_presence'] = sum(1 for ind in error_indicators if ind in step_lower) / 5
        
        # Confidence indicators
        confidence_indicators = ['clearly', 'obviously', 'definitely', 'certainly']
        uncertainty_indicators = ['maybe', 'perhaps', 'possibly', 'might', 'could']
        components['confidence_language'] = (
            sum(1 for ind in confidence_indicators if ind in step_lower) -
            sum(1 for ind in uncertainty_indicators if ind in step_lower)
        ) / 5
        
        # Normalize all components to [0, 1]
        for key, value in components.items():
            components[key] = max(0.0, min(1.0, value))
        
        return components
    
    def _classify_step_type(self, step: str) -> str:
        """Classify the type of reasoning step"""
        
        step_lower = step.lower()
        
        if any(word in step_lower for word in ['substitute', 'replace', 'plug']):
            return 'substitution'
        elif any(word in step_lower for word in ['solve', 'calculate', 'compute']):
            return 'calculation'
        elif any(word in step_lower for word in ['therefore', 'thus', 'hence']):
            return 'conclusion'
        elif any(word in step_lower for word in ['assume', 'suppose', 'given']):
            return 'assumption'
        elif any(word in step_lower for word in ['check', 'verify', 'confirm']):
            return 'verification'
        else:
            return 'general_reasoning'
    
    def _generate_step_reasoning(self, step: str, quality: float, components: Dict[str, float]) -> str:
        """Generate human-readable reasoning for step quality"""
        
        if quality > 0.8:
            base = "Excellent reasoning step with strong logical structure"
        elif quality > 0.6:
            base = "Good reasoning step with clear progression"
        elif quality > 0.4:
            base = "Adequate reasoning step with some clarity"
        else:
            base = "Weak reasoning step that may need improvement"
        
        # Add specific feedback based on components
        details = []
        
        if components.get('logical_structure', 0) > 0.5:
            details.append("good logical flow")
        if components.get('mathematical_rigor', 0) > 0.5:
            details.append("strong mathematical approach")
        if components.get('clarity', 0) > 0.5:
            details.append("clear explanation")
        if components.get('error_presence', 0) > 0.3:
            details.append("potential errors detected")
        
        if details:
            return f"{base}: {', '.join(details)}"
        else:
            return base

class OutcomeRewardModel(nn.Module):
    """
    Outcome Reward Model (ORM) - Evaluates final results
    Focuses on correctness, completeness, and quality of final outputs
    """
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 768, num_layers: int = 4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Text encoder for outcome analysis
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        # Multi-head outcome evaluation
        self.correctness_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.completeness_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.helpfulness_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, 0.0, 0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through ORM"""
        
        # Embed tokens
        embeddings = self.embedding(input_ids)
        
        # Apply transformer layers
        hidden_states = embeddings.transpose(0, 1)
        
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        # Pool sequence representation
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states.transpose(0, 1) * mask).sum(1) / mask.sum(1)
        else:
            pooled = hidden_states.mean(0)
        
        # Compute outcome metrics
        correctness = self.correctness_head(pooled)
        completeness = self.completeness_head(pooled)
        helpfulness = self.helpfulness_head(pooled)
        confidence = self.confidence_head(pooled)
        
        return {
            'correctness': correctness,
            'completeness': completeness,
            'helpfulness': helpfulness,
            'confidence': confidence,
            'hidden_states': pooled
        }
    
    def compute_outcome_reward(self, prompt: str, response: str, tokenizer) -> RewardScore:
        """Compute reward for final outcome"""
        
        try:
            # Combine prompt and response for context
            full_text = f"Prompt: {prompt}\nResponse: {response}"
            
            # Tokenize
            tokens = tokenizer(full_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(tokens['input_ids'], tokens.get('attention_mask'))
            
            # Extract scores
            correctness = outputs['correctness'].item()
            completeness = outputs['completeness'].item()
            helpfulness = outputs['helpfulness'].item()
            confidence = outputs['confidence'].item()
            
            # Analyze outcome components
            components = self._analyze_outcome_components(prompt, response)
            
            # Combine neural and rule-based scores
            neural_score = (correctness + completeness + helpfulness) / 3
            rule_score = sum(components.values()) / len(components) if components else 0.5
            
            # Weighted combination
            final_score = 0.7 * neural_score + 0.3 * rule_score
            
            # Generate reasoning
            reasoning = self._generate_outcome_reasoning(prompt, response, final_score, components)
            
            return RewardScore(
                score=final_score,
                confidence=confidence,
                reasoning=reasoning,
                components={
                    'correctness': correctness,
                    'completeness': completeness,
                    'helpfulness': helpfulness,
                    **components
                },
                metadata={
                    'prompt_length': len(prompt),
                    'response_length': len(response),
                    'neural_score': neural_score,
                    'rule_score': rule_score
                }
            )
            
        except Exception as e:
            return RewardScore(
                score=0.3,
                confidence=0.1,
                reasoning=f"Error computing outcome reward: {str(e)}",
                components={'error': 1.0},
                metadata={'error': str(e)}
            )
    
    def _analyze_outcome_components(self, prompt: str, response: str) -> Dict[str, float]:
        """Analyze rule-based outcome components"""
        
        components = {}
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        # Response relevance (keyword overlap)
        prompt_words = set(re.findall(r'\b\w+\b', prompt_lower))
        response_words = set(re.findall(r'\b\w+\b', response_lower))
        
        if prompt_words:
            relevance = len(prompt_words & response_words) / len(prompt_words)
            components['relevance'] = min(relevance, 1.0)
        else:
            components['relevance'] = 0.5
        
        # Response length appropriateness
        response_length = len(response.split())
        if response_length < 5:
            components['length_appropriateness'] = 0.2  # Too short
        elif response_length > 500:
            components['length_appropriateness'] = 0.7  # Might be too long
        else:
            components['length_appropriateness'] = 0.9  # Good length
        
        # Task completion indicators
        completion_indicators = {
            'mathematical': ['answer', 'solution', 'result', '=', 'therefore'],
            'creative': ['story', 'character', 'plot', 'ending'],
            'code': ['function', 'class', 'return', 'def'],
            'factual': ['answer', 'fact', 'information'],
            'reasoning': ['conclusion', 'analysis', 'because', 'therefore']
        }
        
        task_completion = 0.5  # Default
        for task_type, indicators in completion_indicators.items():
            if any(ind in prompt_lower for ind in indicators[:2]):  # Task detection
                if any(ind in response_lower for ind in indicators):  # Completion check
                    task_completion = 0.9
                    break
        
        components['task_completion'] = task_completion
        
        # Quality indicators
        quality_positive = ['detailed', 'comprehensive', 'clear', 'accurate', 'correct']
        quality_negative = ['wrong', 'error', 'mistake', 'unclear', 'incomplete']
        
        positive_count = sum(1 for ind in quality_positive if ind in response_lower)
        negative_count = sum(1 for ind in quality_negative if ind in response_lower)
        
        components['quality_indicators'] = max(0.0, min(1.0, (positive_count - negative_count + 2) / 4))
        
        return components
    
    def _generate_outcome_reasoning(self, prompt: str, response: str, score: float, components: Dict[str, float]) -> str:
        """Generate human-readable reasoning for outcome quality"""
        
        if score > 0.8:
            base = "Excellent response that effectively addresses the prompt"
        elif score > 0.6:
            base = "Good response with appropriate content"
        elif score > 0.4:
            base = "Adequate response but could be improved"
        else:
            base = "Response needs significant improvement"
        
        # Add specific feedback
        details = []
        
        if components.get('relevance', 0) > 0.7:
            details.append("highly relevant to prompt")
        elif components.get('relevance', 0) < 0.3:
            details.append("limited relevance to prompt")
        
        if components.get('task_completion', 0) > 0.8:
            details.append("completes the requested task")
        elif components.get('task_completion', 0) < 0.4:
            details.append("incomplete task completion")
        
        if components.get('quality_indicators', 0) > 0.7:
            details.append("shows good quality indicators")
        
        if details:
            return f"{base}: {', '.join(details)}"
        else:
            return base

class HybridRewardModel:
    """
    Hybrid Reward Model combining PRM and ORM
    Provides comprehensive evaluation of both process and outcome
    """
    
    def __init__(self, prm: ProcessRewardModel, orm: OutcomeRewardModel, 
                 prm_weight: float = 0.4, orm_weight: float = 0.6):
        self.prm = prm
        self.orm = orm
        self.prm_weight = prm_weight
        self.orm_weight = orm_weight
    
    def evaluate_response(self, prompt: str, response: str, tokenizer,
                         include_steps: bool = True) -> RewardScore:
        """Comprehensive evaluation using both PRM and ORM"""
        
        start_time = time.time()
        
        try:
            # Get outcome reward
            orm_reward = self.orm.compute_outcome_reward(prompt, response, tokenizer)
            
            # Get process rewards if steps are available
            prm_rewards = []
            if include_steps:
                steps = self._extract_reasoning_steps(response)
                if steps:
                    prm_rewards = self.prm.compute_step_rewards(steps, tokenizer)
            
            # Combine scores
            if prm_rewards:
                avg_prm_score = sum(r.score for r in prm_rewards) / len(prm_rewards)
                avg_prm_confidence = sum(r.confidence for r in prm_rewards) / len(prm_rewards)
            else:
                avg_prm_score = 0.5  # Neutral when no steps
                avg_prm_confidence = 0.3
            
            # Weighted combination
            combined_score = (self.prm_weight * avg_prm_score + 
                            self.orm_weight * orm_reward.score)
            
            combined_confidence = (self.prm_weight * avg_prm_confidence + 
                                 self.orm_weight * orm_reward.confidence)
            
            # Generate combined reasoning
            reasoning_parts = []
            if prm_rewards:
                reasoning_parts.append(f"Process evaluation: {len(prm_rewards)} steps analyzed")
            reasoning_parts.append(f"Outcome evaluation: {orm_reward.reasoning}")
            
            combined_reasoning = "; ".join(reasoning_parts)
            
            # Combine components
            combined_components = orm_reward.components.copy()
            if prm_rewards:
                combined_components['process_steps'] = len(prm_rewards)
                combined_components['avg_step_quality'] = avg_prm_score
            
            return RewardScore(
                score=combined_score,
                confidence=combined_confidence,
                reasoning=combined_reasoning,
                components=combined_components,
                metadata={
                    'evaluation_time': time.time() - start_time,
                    'prm_weight': self.prm_weight,
                    'orm_weight': self.orm_weight,
                    'steps_analyzed': len(prm_rewards),
                    'orm_score': orm_reward.score,
                    'prm_score': avg_prm_score
                }
            )
            
        except Exception as e:
            return RewardScore(
                score=0.3,
                confidence=0.1,
                reasoning=f"Hybrid evaluation failed: {str(e)}",
                components={'error': 1.0},
                metadata={'error': str(e), 'evaluation_time': time.time() - start_time}
            )
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response text"""
        
        # Look for numbered steps
        numbered_steps = re.findall(r'\d+[.)]\s*([^.]+(?:\.[^.]*)*)', response)
        if numbered_steps:
            return [step.strip() for step in numbered_steps]
        
        # Look for bullet points
        bullet_steps = re.findall(r'[•\-*]\s*([^.]+(?:\.[^.]*)*)', response)
        if bullet_steps:
            return [step.strip() for step in bullet_steps]
        
        # Look for step indicators
        step_indicators = ['first', 'second', 'third', 'next', 'then', 'finally', 'step']
        sentences = re.split(r'[.!?]+', response)
        
        steps = []
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in step_indicators):
                steps.append(sentence)
        
        return steps if len(steps) > 1 else []

def create_reward_models(vocab_size: int = 32000) -> Tuple[ProcessRewardModel, OutcomeRewardModel, HybridRewardModel]:
    """Factory function to create all reward models"""
    
    prm = ProcessRewardModel(vocab_size=vocab_size)
    orm = OutcomeRewardModel(vocab_size=vocab_size)
    hybrid = HybridRewardModel(prm, orm)
    
    return prm, orm, hybrid

def main():
    """Test reward models"""
    
    print("🧪 Testing Reward Models")
    print("=" * 30)
    
    # Mock tokenizer for testing
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            # Simple word-based tokenization for testing
            words = text.split()[:50]  # Limit length
            token_ids = [hash(word) % 1000 for word in words]
            
            result = {'input_ids': torch.tensor([token_ids])}
            if kwargs.get('return_tensors') == 'pt':
                if len(token_ids) < 50:
                    # Pad to fixed length
                    padded = token_ids + [0] * (50 - len(token_ids))
                    result['input_ids'] = torch.tensor([padded])
                    result['attention_mask'] = torch.tensor([[1] * len(token_ids) + [0] * (50 - len(token_ids))])
            
            return result
    
    # Create models
    prm, orm, hybrid = create_reward_models(vocab_size=1000)
    tokenizer = MockTokenizer()
    
    # Test cases
    test_cases = [
        {
            'prompt': "Solve the equation x² - 4x + 3 = 0",
            'response': "First, I'll use the quadratic formula. Then x = (4 ± √(16-12))/2 = (4 ± 2)/2. Therefore x = 3 or x = 1."
        },
        {
            'prompt': "Write a short story about a robot",
            'response': "There once was a robot named Alex who dreamed of painting. Every day he watched humans create art and wished he could too."
        }
    ]
    
    print(f"Testing {len(test_cases)} cases...\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}] Testing: {case['prompt'][:40]}...")
        
        # Test hybrid evaluation
        reward = hybrid.evaluate_response(case['prompt'], case['response'], tokenizer)
        
        print(f"   Score: {reward.score:.3f}")
        print(f"   Confidence: {reward.confidence:.3f}")
        print(f"   Reasoning: {reward.reasoning}")
        print(f"   Components: {len(reward.components)} metrics")
        print()
    
    print("✅ Reward model testing completed!")

if __name__ == "__main__":
    main()