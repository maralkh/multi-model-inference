# File: config/settings.py
"""Enhanced configuration settings for the multi-model inference system"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

@dataclass
class SystemConfig:
    """Main system configuration"""
    # Model settings
    default_model_id: str = "general_model"
    max_concurrent_requests: int = 10
    model_selection_threshold: float = 0.3
    
    # Performance settings
    max_generation_length: int = 512
    default_temperature: float = 0.7
    timeout_seconds: int = 30
    
    # Logging and monitoring
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_interval: int = 60
    
    # Storage settings
    enable_result_caching: bool = True
    cache_size_mb: int = 100
    save_performance_history: bool = True

@dataclass
class EnhancedManifoldConfig:
    """Enhanced configuration for geometric and Bayesian manifold learning"""
    
    # Basic manifold settings
    embedding_dim: int = 50
    manifold_method: str = "sphere"  # "sphere", "torus", "hyperbolic", "umap", "tsne", "pca"
    
    # Geometric manifold parameters
    sphere_radius: float = 1.0
    torus_major_radius: float = 2.0
    torus_minor_radius: float = 1.0
    hyperbolic_dimension: int = 10
    
    # Bayesian learning settings
    enable_bayesian_learning: bool = True
    gp_kernel_type: str = "rbf"  # "rbf", "matern", "rational_quadratic"
    gp_noise_level: float = 0.1
    gp_length_scale: float = 1.0
    
    # Online learning parameters
    online_batch_size: int = 32
    offline_update_frequency: int = 100
    enable_online_learning: bool = True
    
    # Memory and caching
    memory_size: int = 1000
    enable_embedding_cache: bool = True
    cache_cleanup_frequency: int = 200
    
    # Clustering settings
    enable_clustering: bool = True
    clustering_threshold: float = 0.3
    max_clusters: int = 20
    min_cluster_size: int = 3
    
    # Uncertainty quantification
    uncertainty_estimation_method: str = "bayesian"  # "bayesian", "ensemble", "dropout"
    confidence_calibration: bool = True
    uncertainty_threshold: float = 0.5
    
    # Active learning
    enable_active_learning: bool = True
    acquisition_function: str = "uncertainty"  # "uncertainty", "diversity", "combined"
    diversity_weight: float = 0.3
    
    # Manifold selection strategy
    auto_manifold_selection: bool = True
    manifold_selection_criteria: List[str] = None
    fallback_manifold: str = "sphere"
    
    def __post_init__(self):
        if self.manifold_selection_criteria is None:
            self.manifold_selection_criteria = ["uncertainty", "task_type", "performance"]

@dataclass
class ModelConfig:
    """Configuration for individual models"""
    model_id: str
    model_path: str
    specialized_domains: List[str]
    performance_weights: Dict[str, float]
    resource_requirements: Dict[str, int]
    
    # Enhanced model settings
    enable_reward_guidance: bool = True
    prm_weight: float = 0.5
    orm_weight: float = 0.5
    
    # Manifold-specific settings
    preferred_manifolds: List[str] = None
    uncertainty_tolerance: float = 0.3
    
    def __post_init__(self):
        if self.preferred_manifolds is None:
            self.preferred_manifolds = ["sphere"]

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted"  # New strategy
    RANDOM = "random"

@dataclass
class LoadBalancingConfig:
    """Enhanced load balancing configuration"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.UNCERTAINTY_WEIGHTED
    max_queue_size: int = 50
    timeout_ms: int = 5000
    health_check_interval: int = 30
    
    # Uncertainty-based load balancing
    uncertainty_weight: float = 0.4
    performance_weight: float = 0.6
    adaptive_thresholds: bool = True

@dataclass 
class BayesianLearningConfig:
    """Configuration for Bayesian learning components"""
    
    # Gaussian Process settings
    kernel_parameters: Dict[str, float] = None
    optimize_hyperparameters: bool = True
    hyperparameter_optimization_interval: int = 100
    
    # Prior distributions
    prior_mean_function: str = "zero"  # "zero", "constant", "linear"
    prior_variance: float = 1.0
    
    # Posterior inference
    inference_method: str = "exact"  # "exact", "variational", "mcmc"
    num_inducing_points: int = 100  # For sparse GP
    
    # Uncertainty calibration
    calibration_method: str = "platt"  # "platt", "isotonic", "bayesian"
    calibration_validation_split: float = 0.2
    
    # Active learning
    acquisition_batch_size: int = 5
    exploration_exploitation_tradeoff: float = 0.1
    
    def __post_init__(self):
        if self.kernel_parameters is None:
            self.kernel_parameters = {
                "length_scale": 1.0,
                "variance": 1.0,
                "noise_variance": 0.1
            }

# Default configurations
DEFAULT_SYSTEM_CONFIG = SystemConfig()

DEFAULT_ENHANCED_MANIFOLD_CONFIG = EnhancedManifoldConfig(
    embedding_dim=50,
    manifold_method="sphere",
    enable_bayesian_learning=True,
    enable_online_learning=True,
    enable_clustering=True,
    auto_manifold_selection=True
)

DEFAULT_BAYESIAN_CONFIG = BayesianLearningConfig()

DEFAULT_ENHANCED_MODEL_CONFIGS = [
    ModelConfig(
        model_id="math_specialist",
        model_path="models/math_specialist",
        specialized_domains=["mathematics", "physics", "engineering"],
        performance_weights={"mathematical": 0.95, "scientific": 0.85},
        resource_requirements={"memory_gb": 8, "compute_units": 4},
        preferred_manifolds=["sphere", "hyperbolic"],
        uncertainty_tolerance=0.2
    ),
    ModelConfig(
        model_id="creative_specialist",
        model_path="models/creative_specialist", 
        specialized_domains=["literature", "arts", "entertainment"],
        performance_weights={"creative_writing": 0.90, "conversational": 0.85},
        resource_requirements={"memory_gb": 6, "compute_units": 3},
        preferred_manifolds=["torus", "sphere"],
        uncertainty_tolerance=0.4
    ),
    ModelConfig(
        model_id="reasoning_specialist",
        model_path="models/reasoning_specialist",
        specialized_domains=["philosophy", "logic", "analysis"],
        performance_weights={"reasoning": 0.92, "factual_qa": 0.88},
        resource_requirements={"memory_gb": 10, "compute_units": 5},
        preferred_manifolds=["hyperbolic", "sphere"],
        uncertainty_tolerance=0.25
    ),
    ModelConfig(
        model_id="code_specialist",
        model_path="models/code_specialist",
        specialized_domains=["programming", "software", "algorithms"],
        performance_weights={"code_generation": 0.93, "reasoning": 0.85},
        resource_requirements={"memory_gb": 8, "compute_units": 4},
        preferred_manifolds=["hyperbolic", "sphere"],
        uncertainty_tolerance=0.3
    ),
    ModelConfig(
        model_id="general_model",
        model_path="models/general_model",
        specialized_domains=["general"],
        performance_weights={"general": 0.75},
        resource_requirements={"memory_gb": 12, "compute_units": 6},
        preferred_manifolds=["sphere", "torus", "hyperbolic"],
        uncertainty_tolerance=0.5
    )
]

# Manifold-specific configurations
MANIFOLD_SPECIFIC_CONFIGS = {
    "sphere": {
        "radius": 1.0,
        "best_for_tasks": ["factual_qa", "conversational", "general"],
        "dimensionality_range": (10, 100),
        "bayesian_kernel": "rbf"
    },
    "torus": {
        "major_radius": 2.0,
        "minor_radius": 1.0,
        "best_for_tasks": ["creative_writing", "conversational"],
        "dimensionality_range": (5, 50),
        "bayesian_kernel": "matern"
    },
    "hyperbolic": {
        "dimension_limit": 20,
        "best_for_tasks": ["reasoning", "code_generation", "mathematical"],
        "dimensionality_range": (5, 30),
        "bayesian_kernel": "rational_quadratic"
    }
}

# Performance tuning configurations
PERFORMANCE_CONFIGS = {
    "high_accuracy": EnhancedManifoldConfig(
        embedding_dim=100,
        enable_bayesian_learning=True,
        gp_noise_level=0.05,
        uncertainty_threshold=0.3,
        offline_update_frequency=50
    ),
    "balanced": EnhancedManifoldConfig(
        embedding_dim=50,
        enable_bayesian_learning=True,
        gp_noise_level=0.1,
        uncertainty_threshold=0.5,
        offline_update_frequency=100
    ),
    "high_speed": EnhancedManifoldConfig(
        embedding_dim=20,
        enable_bayesian_learning=False,
        enable_clustering=False,
        offline_update_frequency=200
    )
}