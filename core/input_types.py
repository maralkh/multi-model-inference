# File: core/input_types.py
"""Fixed types with all required attributes for multi-model inference"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch.nn as nn
import numpy as np

class TaskType(Enum):
    """Different types of tasks for model specialization"""
    MATHEMATICAL = "mathematical"
    CREATIVE_WRITING = "creative_writing"
    FACTUAL_QA = "factual_qa"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    SCIENTIFIC = "scientific"
    CONVERSATIONAL = "conversational"

@dataclass
class InputAnalysis:
    """Comprehensive analysis results for input text with all required attributes"""
    
    # Core classification results
    task_type: TaskType
    confidence: float
    
    # Text features and metadata
    features: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    domain_indicators: List[str] = field(default_factory=list)
    
    # Additional attributes that might be expected by multi-model engine
    uncertainty_estimate: float = 0.0
    processing_time: float = 0.0
    method_used: str = "default"
    
    # Manifold learning attributes
    best_manifold: str = "euclidean"
    manifold_confidence: float = 0.0
    embedding: Optional[np.ndarray] = None
    
    # Performance tracking
    timestamp: float = 0.0
    model_recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure all required attributes are properly initialized"""
        
        # Set default timestamp if not provided
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()
        
        # Ensure keywords is always a list
        if not isinstance(self.keywords, list):
            self.keywords = []
        
        # Ensure domain_indicators is always a list  
        if not isinstance(self.domain_indicators, list):
            self.domain_indicators = []
        
        # Ensure model_recommendations is always a list
        if not isinstance(self.model_recommendations, list):
            self.model_recommendations = []
        
        # Ensure confidence is within valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Set default uncertainty if not provided
        if self.uncertainty_estimate == 0.0:
            self.uncertainty_estimate = 1.0 - self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'task_type': self.task_type.value,
            'confidence': self.confidence,
            'features': self.features,
            'keywords': self.keywords,
            'complexity_score': self.complexity_score,
            'domain_indicators': self.domain_indicators,
            'uncertainty_estimate': self.uncertainty_estimate,
            'processing_time': self.processing_time,
            'method_used': self.method_used,
            'best_manifold': self.best_manifold,
            'manifold_confidence': self.manifold_confidence,
            'timestamp': self.timestamp,
            'model_recommendations': self.model_recommendations
        }
        
        # Handle numpy array
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        else:
            result['embedding'] = None
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InputAnalysis':
        """Create from dictionary"""
        
        # Convert task_type string back to enum
        if isinstance(data.get('task_type'), str):
            data['task_type'] = TaskType(data['task_type'])
        
        # Convert embedding back to numpy array
        if data.get('embedding') is not None:
            data['embedding'] = np.array(data['embedding'])
        
        return cls(**data)

@dataclass
class ModelSpec:
    """Specification for a specialized model with all required attributes"""
    model_id: str
    model: nn.Module
    prm: Optional[object] = None  # ProcessRewardModel
    orm: Optional[object] = None  # OutcomeRewardModel
    task_types: List[TaskType] = field(default_factory=list)
    specialized_domains: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    
    # Additional metadata
    memory_requirements: str = "8GB"
    inference_speed: str = "medium"
    last_updated: float = 0.0
    
    def __post_init__(self):
        """Initialize default values"""
        if self.last_updated == 0.0:
            import time
            self.last_updated = time.time()
        
        # Ensure task_types is always a list
        if not isinstance(self.task_types, list):
            self.task_types = []
        
        # Ensure specialized_domains is always a list
        if not isinstance(self.specialized_domains, list):
            self.specialized_domains = []

@dataclass
class ManifoldLearningConfig:
    """Configuration for manifold learning with all required attributes"""
    embedding_dim: int = 50
    online_batch_size: int = 32
    offline_update_frequency: int = 100
    memory_size: int = 1000
    clustering_threshold: float = 0.3
    manifold_method: str = "umap"  # "umap", "tsne", "pca", "sphere", "torus", "hyperbolic"
    enable_online_learning: bool = True
    enable_clustering: bool = True
    similarity_threshold: float = 0.7
    
    # Additional configuration options
    enable_caching: bool = True
    cache_size: int = 1000
    uncertainty_estimation: bool = True
    performance_tracking: bool = True

@dataclass
class DataPoint:
    """Represents a data point in the manifold space with all required attributes"""
    text: str
    embedding: np.ndarray
    task_type: TaskType
    selected_model: str
    performance_score: float
    timestamp: float
    complexity: float = 0.0
    cluster_id: int = -1
    
    # Additional attributes
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.0
    
    def __post_init__(self):
        """Initialize default values"""
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()

# Utility functions for creating safe InputAnalysis objects
def create_safe_input_analysis(
    task_type: TaskType,
    confidence: float,
    text: str = "",
    **kwargs
) -> InputAnalysis:
    """Create InputAnalysis with all required attributes safely"""
    
    # Extract features if text provided
    features = kwargs.get('features', {})
    if text and not features:
        features = extract_basic_features(text)
    
    # Extract keywords if text provided
    keywords = kwargs.get('keywords', [])
    if text and not keywords:
        keywords = extract_basic_keywords(text)
    
    # Extract domain indicators
    domain_indicators = kwargs.get('domain_indicators', [])
    if text and not domain_indicators:
        domain_indicators = extract_domain_indicators(text)
    
    return InputAnalysis(
        task_type=task_type,
        confidence=confidence,
        features=features,
        keywords=keywords,
        complexity_score=kwargs.get('complexity_score', 0.0),
        domain_indicators=domain_indicators,
        uncertainty_estimate=kwargs.get('uncertainty_estimate', 1.0 - confidence),
        processing_time=kwargs.get('processing_time', 0.0),
        method_used=kwargs.get('method_used', 'safe_creation'),
        best_manifold=kwargs.get('best_manifold', 'euclidean'),
        manifold_confidence=kwargs.get('manifold_confidence', confidence),
        model_recommendations=kwargs.get('model_recommendations', [])
    )

def extract_basic_features(text: str) -> Dict[str, Any]:
    """Extract basic features from text"""
    import re
    
    if not text:
        return {'length': 0, 'word_count': 0, 'sentence_count': 0}
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'length': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!'),
        'mathematical_symbols': len(re.findall(r'[+\-*/=<>≤≥∫∑]', text)),
        'code_indicators': len(re.findall(r'[{}();]', text)),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'avg_word_length': sum(len(word) for word in words) / max(len(words), 1)
    }

def extract_basic_keywords(text: str) -> List[str]:
    """Extract basic keywords from text"""
    import re
    from collections import Counter
    
    if not text:
        return []
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stopwords = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
        'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'among', 'this', 'that', 'these', 'those', 'have',
        'has', 'had', 'will', 'would', 'could', 'should', 'can', 'may', 'might'
    }
    
    # Filter words and get most common
    filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
    word_counts = Counter(filtered_words)
    
    # Return top 5 keywords
    return [word for word, count in word_counts.most_common(5)]

def extract_domain_indicators(text: str) -> List[str]:
    """Extract domain indicators from text"""
    
    if not text:
        return []
    
    text_lower = text.lower()
    domains = []
    
    # Domain keyword mapping
    domain_keywords = {
        'mathematics': ['equation', 'solve', 'calculate', 'formula', 'algebra', 'geometry'],
        'science': ['hypothesis', 'experiment', 'theory', 'research', 'data'],
        'technology': ['algorithm', 'code', 'software', 'system', 'programming'],
        'creative': ['story', 'character', 'creative', 'write', 'narrative'],
        'business': ['market', 'strategy', 'revenue', 'customer', 'profit']
    }
    
    for domain, keywords in domain_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            domains.append(domain)
    
    return domains

# Backward compatibility - alias for existing code
FixedInputAnalysis = InputAnalysis