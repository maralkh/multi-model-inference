# File: core/input_classifier.py
"""Input classification and analysis for multi-model routing"""

import re
import time
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import deque

from .input_types import TaskType, InputAnalysis

class InputClassifier:
    """Analyzes input text to determine task type and characteristics"""
    
    def __init__(self):
        self.task_patterns = {
            TaskType.MATHEMATICAL: {
                'keywords': ['solve', 'calculate', 'equation', 'formula', 'algebra', 'geometry', 
                           'derivative', 'integral', 'probability', 'statistics', 'theorem'],
                'patterns': [r'\d+[x-z]\s*[+\-*/=]', r'[+\-*/=]\s*\d+', r'\b\d+\.\d+\b', 
                           r'[∫∑∏√∞≠≤≥∈∉⊂⊃∩∪]'],
                'complexity_indicators': ['differential', 'matrix', 'vector', 'proof', 'limit']
            },
            TaskType.CREATIVE_WRITING: {
                'keywords': ['story', 'write', 'creative', 'character', 'plot', 'narrative',
                           'poem', 'fiction', 'dialogue', 'scene', 'chapter'],
                'patterns': [r'write\s+a\s+story', r'create\s+a\s+character', r'once\s+upon'],
                'complexity_indicators': ['literary', 'metaphor', 'symbolism', 'genre']
            },
            TaskType.FACTUAL_QA: {
                'keywords': ['what', 'who', 'when', 'where', 'why', 'how', 'explain', 'define',
                           'fact', 'information', 'details', 'describe'],
                'patterns': [r'^(what|who|when|where|why|how)\s+', r'explain\s+', r'define\s+'],
                'complexity_indicators': ['comprehensive', 'detailed', 'analysis', 'research']
            },
            TaskType.REASONING: {
                'keywords': ['analyze', 'compare', 'evaluate', 'assess', 'argue', 'reason',
                           'logic', 'because', 'therefore', 'conclude', 'infer'],
                'patterns': [r'pros\s+and\s+cons', r'advantages\s+and\s+disadvantages', 
                           r'compare\s+.*\s+with', r'analyze\s+'],
                'complexity_indicators': ['critical', 'philosophical', 'ethical', 'complex']
            },
            TaskType.CODE_GENERATION: {
                'keywords': ['code', 'program', 'function', 'algorithm', 'python', 'javascript',
                           'class', 'method', 'variable', 'loop', 'debug'],
                'patterns': [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'#.*code',
                           r'write\s+.*\s+code'],
                'complexity_indicators': ['optimization', 'architecture', 'framework', 'api']
            },
            TaskType.SCIENTIFIC: {
                'keywords': ['research', 'hypothesis', 'experiment', 'data', 'method',
                           'analysis', 'conclusion', 'theory', 'evidence', 'study'],
                'patterns': [r'research\s+', r'study\s+shows', r'according\s+to'],
                'complexity_indicators': ['peer-reviewed', 'methodology', 'statistical', 'empirical']
            },
            TaskType.CONVERSATIONAL: {
                'keywords': ['hello', 'hi', 'thanks', 'please', 'help', 'chat', 'talk',
                           'discuss', 'opinion', 'think'],
                'patterns': [r'^(hi|hello|hey)', r'what\s+do\s+you\s+think', r'can\s+you\s+help'],
                'complexity_indicators': ['personal', 'emotional', 'social', 'casual']
            }
        }
        
        # Initialize TF-IDF for semantic analysis
        self.tfidf_vectorizer = None
        self.domain_embeddings = {}
        
    def analyze_input(self, text: str) -> InputAnalysis:
        """Analyze input text to determine task type and characteristics"""
        
        text_lower = text.lower()
        
        # Calculate scores for each task type
        task_scores = {}
        
        for task_type, patterns in self.task_patterns.items():
            score = 0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in patterns['keywords'] 
                                if keyword in text_lower)
            score += keyword_matches * 2
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in patterns['patterns']
                                if re.search(pattern, text_lower))
            score += pattern_matches * 3
            
            # Complexity indicators
            complexity_matches = sum(1 for indicator in patterns['complexity_indicators']
                                   if indicator in text_lower)
            score += complexity_matches * 1.5
            
            task_scores[task_type] = score
        
        # Determine best task type
        best_task = max(task_scores.keys(), key=lambda k: task_scores[k])
        confidence = task_scores[best_task] / max(sum(task_scores.values()), 1)
        
        # Extract features
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'question_marks': text.count('?'),
            'mathematical_symbols': len(re.findall(r'[+\-*/=<>≤≥∫∑]', text)),
            'code_indicators': len(re.findall(r'[{}();]', text)),
            'task_scores': {t.value: s for t, s in task_scores.items()}
        }
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(text, features)
        
        # Extract keywords and domain indicators
        keywords = self._extract_keywords(text)
        domain_indicators = self._identify_domains(text)
        
        return InputAnalysis(
            task_type=best_task,
            confidence=confidence,
            features=features,
            keywords=keywords,
            complexity_score=complexity_score,
            domain_indicators=domain_indicators
        )
    
    def _calculate_complexity(self, text: str, features: Dict) -> float:
        """Calculate complexity score based on various factors"""
        
        complexity = 0.0
        
        # Length-based complexity
        complexity += min(features['length'] / 1000, 1.0) * 0.3
        
        # Technical term density
        technical_terms = ['algorithm', 'optimization', 'analysis', 'methodology', 
                         'implementation', 'architecture', 'framework']
        tech_density = sum(1 for term in technical_terms if term in text.lower())
        complexity += min(tech_density / 5, 1.0) * 0.4
        
        # Sentence structure complexity
        avg_sentence_length = features['word_count'] / max(features['sentence_count'], 1)
        complexity += min(avg_sentence_length / 20, 1.0) * 0.3
        
        return min(complexity, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        
        # Simple keyword extraction (in practice, use more sophisticated NLP)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = Counter(words)
        
        # Filter out common words
        stopwords = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been',
                    'have', 'were', 'said', 'each', 'which', 'their', 'time', 'would'}
        
        keywords = [word for word, freq in word_freq.most_common(10) 
                   if word not in stopwords and len(word) > 3]
        
        return keywords[:5]
    
    def _identify_domains(self, text: str) -> List[str]:
        """Identify domain-specific indicators"""
        
        domain_keywords = {
            'mathematics': ['equation', 'formula', 'theorem', 'proof', 'calculation'],
            'science': ['hypothesis', 'experiment', 'theory', 'research', 'data'],
            'technology': ['algorithm', 'code', 'software', 'system', 'programming'],
            'business': ['market', 'strategy', 'analysis', 'revenue', 'customer'],
            'education': ['learn', 'teach', 'student', 'curriculum', 'knowledge'],
            'healthcare': ['patient', 'treatment', 'diagnosis', 'medical', 'health']
        }
        
        identified_domains = []
        text_lower = text.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                identified_domains.append(domain)
        
        return identified_domains