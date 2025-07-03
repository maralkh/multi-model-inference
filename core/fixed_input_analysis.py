# File: core/fixed_input_analysis.py
"""Fixed input analysis with proper attribute handling"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.input_types import TaskType, InputAnalysis

@dataclass
class FixedInputAnalysis:
    """Fixed input analysis with all required attributes"""
    task_type: TaskType = TaskType.CONVERSATIONAL  # Use existing TaskType
    confidence_score: float = 0.5
    manifold_type: str = "euclidean"
    complexity_score: float = 0.5
    specialized_domains: List[str] = None
    reasoning_type: str = "general"
    estimated_tokens: int = 0
    processing_time: float = 0.0
    domain_indicators: List[str] = None
    confidence: float = 0.5  
    pattern_matches: List[str] = None
    linguistic_features: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.specialized_domains is None:
            self.specialized_domains = []

class SafeInputClassifier:
    """Safe input classifier that always returns complete analysis"""
    
    def __init__(self, base_classifier=None):
        self.base_classifier = base_classifier
        self.prediction_cache = {}
        
        # Pattern-based classification patterns
        self.task_patterns = {
            TaskType.MATHEMATICAL: [
                'calculate', 'solve', 'equation', 'derivative', 'integral', 'math', 'formula',
                'algebra', 'geometry', 'trigonometry', 'calculus', 'statistics', 'probability',
                '+', '-', '*', '/', '=', 'x²', 'sin', 'cos', 'log', 'sqrt'
            ],
            TaskType.CODE_GENERATION: [
                'def', 'function', 'class', 'import', 'return', 'print', 'if', 'for', 'while',
                'python', 'javascript', 'java', 'cpp', 'algorithm', 'code', 'program', 'script',
                'variable', 'array', 'loop', 'recursion', 'debug', 'compile'
            ],
            TaskType.CREATIVE_WRITING: [
                'write', 'story', 'poem', 'character', 'plot', 'narrative', 'fiction', 'novel',
                'creative', 'imagination', 'dialogue', 'scene', 'chapter', 'protagonist',
                'haiku', 'sonnet', 'essay', 'article'
            ],
            TaskType.SCIENTIFIC: [
                'explain', 'science', 'biology', 'chemistry', 'physics', 'experiment', 'theory',
                'hypothesis', 'research', 'study', 'analysis', 'molecule', 'atom', 'cell',
                'dna', 'protein', 'photosynthesis', 'evolution', 'quantum', 'energy'
            ],
            TaskType.REASONING: [
                'analyze', 'compare', 'evaluate', 'assess', 'argue', 'pros', 'cons', 'advantage',
                'disadvantage', 'benefit', 'risk', 'impact', 'effect', 'cause', 'consequence',
                'opinion', 'perspective', 'viewpoint', 'consider', 'think'
            ],
            TaskType.FACTUAL_QA: [
                'what', 'when', 'where', 'who', 'which', 'how many', 'capital', 'year',
                'date', 'location', 'country', 'city', 'president', 'invented', 'discovered',
                'fact', 'information', 'data', 'statistic'
            ],
            TaskType.CONVERSATIONAL: [
                'hello', 'hi', 'hey', 'thanks', 'thank you', 'please', 'help', 'assist',
                'good morning', 'good afternoon', 'goodbye', 'bye', 'how are you',
                'nice to meet', 'excuse me', 'sorry', 'welcome'
            ]
        }
    
    def analyze_input(self, text: str) -> FixedInputAnalysis:
        """Analyze input and return complete analysis"""
        
        # Check cache first
        if text in self.prediction_cache:
            return self.prediction_cache[text]
        
        start_time = time.time()
        
        try:
            # Try base classifier first if available
            if self.base_classifier:
                try:
                    base_analysis = self.base_classifier.analyze_input(text)
                    
                    # Convert to our format with safe attribute access
                    fixed_analysis = FixedInputAnalysis(
                        task_type=base_analysis.task_type,
                        confidence_score=getattr(base_analysis, 'confidence_score', 
                                              getattr(base_analysis, 'confidence', 0.7)),
                        manifold_type=getattr(base_analysis, 'manifold_type', 'euclidean'),
                        complexity_score=getattr(base_analysis, 'complexity_score', 0.5),
                        specialized_domains=getattr(base_analysis, 'specialized_domains', []),
                        reasoning_type=getattr(base_analysis, 'reasoning_type', 'general'),
                        estimated_tokens=len(text.split()),
                        processing_time=time.time() - start_time,
                        domain_indicators=getattr(base_analysis, 'domain_indicators', 
                                                getattr(base_analysis, 'specialized_domains', [])),
                        pattern_matches=getattr(base_analysis, 'pattern_matches', []),
                        linguistic_features=getattr(base_analysis, 'linguistic_features', {})
                    )
                    
                    # Cache and return
                    self.prediction_cache[text] = fixed_analysis
                    return fixed_analysis
                    
                except Exception as e:
                    # Fall back to pattern-based classification
                    pass
            
            # Pattern-based classification as fallback
            task_type, confidence = self._pattern_based_classification(text)
            
            # Calculate other attributes
            complexity = self._calculate_complexity(text)
            manifold_type = self._determine_manifold_type(task_type)
            domains = self._identify_domains(text, task_type)
            pattern_matches = self._extract_pattern_matches(text, task_type)
            
            fixed_analysis = FixedInputAnalysis(
                task_type=task_type,
                confidence_score=confidence,
                manifold_type=manifold_type,
                complexity_score=complexity,
                specialized_domains=domains,
                reasoning_type='general',
                estimated_tokens=len(text.split()),
                processing_time=time.time() - start_time,
                domain_indicators=domains.copy(),
                pattern_matches=pattern_matches,
                linguistic_features=self._extract_linguistic_features(text)
            )
            
            # Cache and return
            self.prediction_cache[text] = fixed_analysis
            return fixed_analysis
            
        except Exception as e:
            # Ultimate fallback - return basic analysis
            fallback_analysis = FixedInputAnalysis(
                task_type=TaskType.CONVERSATIONAL,  # Use existing TaskType
                confidence_score=0.3,
                manifold_type='euclidean',
                complexity_score=0.5,
                specialized_domains=[],
                reasoning_type='general',
                estimated_tokens=len(text.split()),
                processing_time=time.time() - start_time
            )
            
            return fallback_analysis
    
    def _pattern_based_classification(self, text: str) -> Tuple[TaskType, float]:
        """Pattern-based classification with confidence scoring"""
        
        text_lower = text.lower()
        task_scores = {}
        
        # Score each task type
        for task_type, patterns in self.task_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    # Weight by pattern length (longer patterns are more specific)
                    score += len(pattern) / 10
            
            # Normalize by number of patterns
            task_scores[task_type] = score / len(patterns)
        
        # Find best match
        if task_scores:
            best_task = max(task_scores.keys(), key=lambda k: task_scores[k])
            best_score = task_scores[best_task]
            
            # Convert score to confidence (0.3 to 0.9 range)
            confidence = min(0.9, max(0.3, best_score * 2 + 0.3))
            
            return best_task, confidence
        else:
            return TaskType.CONVERSATIONAL, 0.5  # Use existing TaskType
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        
        # Simple heuristics for complexity
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        
        # Average word length
        avg_word_length = char_count / max(1, word_count)
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count
        
        # Complexity factors
        length_factor = min(1.0, word_count / 50)  # Longer = more complex
        word_factor = min(1.0, avg_word_length / 8)  # Longer words = more complex
        sentence_factor = min(1.0, avg_sentence_length / 15)  # Longer sentences = more complex
        
        complexity = (length_factor + word_factor + sentence_factor) / 3
        
        return max(0.1, min(1.0, complexity))
    
    def _determine_manifold_type(self, task_type: TaskType) -> str:
        """Determine appropriate manifold type for task"""
        
        manifold_mapping = {
            TaskType.MATHEMATICAL: 'euclidean',
            TaskType.CODE_GENERATION: 'euclidean', 
            TaskType.CREATIVE_WRITING: 'sphere',
            TaskType.SCIENTIFIC: 'euclidean',
            TaskType.REASONING: 'hyperbolic',
            TaskType.FACTUAL_QA: 'euclidean',
            TaskType.CONVERSATIONAL: 'sphere',
            TaskType.REASONING: 'hyperbolic'  # Remove GENERAL
        }
        
        return manifold_mapping.get(task_type, 'euclidean')
    
    def _identify_domains(self, text: str, task_type: TaskType) -> List[str]:
        """Identify specialized domains"""
        
        domain_keywords = {
            'mathematics': ['algebra', 'calculus', 'geometry', 'statistics', 'equation'],
            'programming': ['python', 'javascript', 'algorithm', 'function', 'code'],
            'science': ['biology', 'chemistry', 'physics', 'experiment', 'theory'],
            'literature': ['story', 'poem', 'character', 'narrative', 'fiction'],
            'business': ['analysis', 'strategy', 'market', 'revenue', 'profit'],
            'general': ['help', 'question', 'information', 'explain', 'describe']
        }
        
        identified_domains = []
        text_lower = text.lower()
        
        for domain, keywords in domain_keywords.items():
            domain_score = sum(1 for keyword in keywords if keyword in text_lower)
            if domain_score >= 1:  # At least one keyword match
                identified_domains.append(domain)
        
        # Always have at least one domain
        if not identified_domains:
            identified_domains = ['general']
        
        return identified_domains[:3]  # Limit to top 3 domains
    
    def _extract_pattern_matches(self, text: str, task_type: TaskType) -> List[str]:
        """Extract pattern matches for the detected task type"""
        
        if task_type not in self.task_patterns:
            return []
        
        text_lower = text.lower()
        matches = []
        
        for pattern in self.task_patterns[task_type]:
            if pattern in text_lower:
                matches.append(pattern)
        
        return matches[:5]  # Limit to top 5 matches
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic linguistic features"""
        
        words = text.split()
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        
        return {
            'word_count': len(words),
            'sentence_count': sentences,
            'avg_word_length': sum(len(word) for word in words) / max(1, len(words)),
            'avg_sentence_length': len(words) / sentences,
            'question_words': sum(1 for word in ['what', 'when', 'where', 'who', 'why', 'how'] if word in text.lower()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?')
        }

def create_safe_classifier(base_classifier=None) -> SafeInputClassifier:
    """Create safe classifier wrapper"""
    return SafeInputClassifier(base_classifier)

def test_safe_classifier():
    """Test the safe classifier"""
    
    print("🔒 Testing Safe Input Classifier")
    print("=" * 40)
    
    classifier = create_safe_classifier()
    
    test_cases = [
        "Calculate the derivative of x³ + 2x² - 5x + 7",
        "Write a creative story about a robot learning to paint", 
        "Implement a binary search algorithm in Python",
        "Explain how photosynthesis works in plants",
        "Analyze the pros and cons of renewable energy sources",
        "What is the capital city of Australia?",
        "Hello! How can I assist you today?"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[{i}] Input: '{text[:40]}...'")
        
        analysis = classifier.analyze_input(text)
        
        print(f"    ✅ Task Type: {analysis.task_type.value}")
        print(f"       Confidence: {analysis.confidence_score:.3f}")
        print(f"       Manifold: {analysis.manifold_type}")
        print(f"       Complexity: {analysis.complexity_score:.3f}")
        print(f"       Domains: {', '.join(analysis.specialized_domains)}")
        print(f"       Processing Time: {analysis.processing_time:.4f}s")

if __name__ == "__main__":
    test_safe_classifier()