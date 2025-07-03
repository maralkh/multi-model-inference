# File: config/data_config.py
"""Missing configuration classes for data preprocessing"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline"""
    
    # Text preprocessing
    max_sequence_length: int = 512
    min_sequence_length: int = 10
    lowercase: bool = True
    remove_special_chars: bool = False
    remove_numbers: bool = False
    
    # Tokenization
    tokenizer_type: str = "basic"  # "basic", "whitespace", "bert"
    vocab_size: int = 30000
    add_special_tokens: bool = True
    
    # Data filtering
    filter_empty_texts: bool = True
    filter_duplicates: bool = True
    min_word_count: int = 3
    
    # Batching
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = False
    
    # Validation
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42

@dataclass 
class ModelLoadingConfig:
    """Configuration for model loading"""
    
    model_path: str = "models/"
    device: str = "auto"  # "auto", "cpu", "cuda"
    precision: str = "float16"  # "float32", "float16", "int8"
    max_memory: Optional[str] = None
    
    # Model-specific settings
    trust_remote_code: bool = False
    use_auth_token: bool = False
    revision: str = "main"
    
    # Performance
    torch_dtype: str = "auto"
    low_cpu_mem_usage: bool = True
    device_map: str = "auto"

@dataclass
class DataPipelineConfig:
    """Configuration for integrated data pipeline"""
    
    preprocessing: PreprocessingConfig = None
    model_loading: ModelLoadingConfig = None
    
    # Pipeline settings
    parallel_processing: bool = True
    num_workers: int = 4
    cache_processed_data: bool = True
    cache_dir: str = "cache/"
    
    # Logging
    log_level: str = "INFO"
    log_progress: bool = True
    
    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.model_loading is None:
            self.model_loading = ModelLoadingConfig()

# Default configurations for quick setup
DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()
DEFAULT_MODEL_LOADING_CONFIG = ModelLoadingConfig()
DEFAULT_DATA_PIPELINE_CONFIG = DataPipelineConfig()