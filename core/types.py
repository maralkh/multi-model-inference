# File: core/types.py
"""Core types and enums for the multi-model inference system"""

from enum import Enum
from dataclasses import dataclass
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
class ModelSpec:
    """Specification for a specialized model"""
    model_id: str
    model: nn.Module
    prm: Optional[object]  # ProcessRewardModel
    orm: Optional[object]  # OutcomeRewardModel
    task_types: List[TaskType]
    specialized_domains: List[str]
    performance_metrics: Dict[str, float]
    description: str

@dataclass
class InputAnalysis:
    """Analysis results for input text"""
    task_type: TaskType
    confidence: float
    features: Dict[str, Any]
    keywords: List[str]
    complexity_score: float
    domain_indicators: List[str]

@dataclass
class ManifoldLearningConfig:
    """Configuration for manifold learning"""
    embedding_dim: int = 50
    online_batch_size: int = 32
    offline_update_frequency: int = 100
    memory_size: int = 1000
    clustering_threshold: float = 0.3
    manifold_method: str = "umap"  # "umap", "tsne", "pca"
    enable_online_learning: bool = True
    enable_clustering: bool = True
    similarity_threshold: float = 0.7

@dataclass
class DataPoint:
    """Represents a data point in the manifold space"""
    text: str
    embedding: np.ndarray
    task_type: TaskType
    selected_model: str
    performance_score: float
    timestamp: float
    complexity: float
    cluster_id: int = -1