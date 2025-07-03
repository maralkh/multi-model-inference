# File: data/integrated_data_pipeline.py
"""
Comprehensive integrated data processing pipeline for multi-model inference system
Supports both research and production environments
"""

import logging
import time
import warnings
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import asdict
import threading
from enum import Enum

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import configurations
try:
    from config.data_config import PreprocessingConfig, ModelLoadingConfig, DataPipelineConfig
except ImportError:
    # Fallback if config module not available
    from dataclasses import dataclass
    
    @dataclass
    class PreprocessingConfig:
        max_sequence_length: int = 512
        batch_size: int = 32
        lowercase: bool = True
        filter_empty_texts: bool = True
        
    @dataclass 
    class ModelLoadingConfig:
        model_path: str = "models/"
        device: str = "auto"
        precision: str = "float16"
        
    @dataclass
    class DataPipelineConfig:
        preprocessing: PreprocessingConfig = None
        model_loading: ModelLoadingConfig = None
        parallel_processing: bool = True
        num_workers: int = 4
        cache_processed_data: bool = True
        cache_dir: str = "cache/"
        log_level: str = "INFO"

class PipelineStage(Enum):
    """Pipeline processing stages"""
    INITIALIZATION = "initialization"
    TEXT_PREPROCESSING = "text_preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    MODEL_LOADING = "model_loading"
    BATCH_PROCESSING = "batch_processing"
    VALIDATION = "validation"
    CACHING = "caching"
    COMPLETED = "completed"

class DataProcessor:
    """Advanced data processing utilities"""
    
    @staticmethod
    def clean_text(text: str, config: PreprocessingConfig) -> str:
        """Advanced text cleaning"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        processed = text
        
        # Basic cleaning
        if config.lowercase:
            processed = processed.lower()
        
        # Remove special characters if configured
        if getattr(config, 'remove_special_chars', False):
            import re
            processed = re.sub(r'[^\w\s]', ' ', processed)
        
        # Remove extra whitespace
        processed = ' '.join(processed.split())
        
        # Apply length constraints
        if len(processed) > config.max_sequence_length:
            # Intelligent truncation at word boundaries
            words = processed.split()
            truncated_words = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= config.max_sequence_length:
                    truncated_words.append(word)
                    current_length += len(word) + 1
                else:
                    break
            
            processed = ' '.join(truncated_words)
        
        return processed
    
    @staticmethod
    def extract_features(text: str) -> Dict[str, Any]:
        """Extract comprehensive text features"""
        if not text:
            return {
                'length': 0, 'word_count': 0, 'sentence_count': 0,
                'avg_word_length': 0, 'complexity_score': 0
            }
        
        import re
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Basic statistics
        features = {
            'length': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1),
            'unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / max(len(words), 1)
        }
        
        # Language indicators
        features.update({
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'punctuation_density': len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
        })
        
        # Complexity estimation
        avg_sentence_length = features['word_count'] / max(features['sentence_count'], 1)
        complexity_score = min(
            (avg_sentence_length / 20) * 0.4 +
            (features['avg_word_length'] / 10) * 0.3 +
            (features['vocabulary_richness']) * 0.3,
            1.0
        )
        features['complexity_score'] = complexity_score
        
        return features
    
    @staticmethod
    def validate_data(data_item: Dict[str, Any], config: PreprocessingConfig) -> bool:
        """Validate processed data item"""
        text = data_item.get('processed_text', '')
        
        # Check minimum requirements
        if not text or len(text.strip()) < getattr(config, 'min_sequence_length', 10):
            return False
        
        # Check word count
        word_count = len(text.split())
        if word_count < getattr(config, 'min_word_count', 3):
            return False
        
        return True

class ModelManager:
    """Advanced model loading and management"""
    
    def __init__(self, config: ModelLoadingConfig):
        self.config = config
        self.models_metadata = {}
        self.load_lock = threading.Lock()
    
    def load_models_metadata(self) -> Dict[str, Any]:
        """Load comprehensive models metadata"""
        with self.load_lock:
            if self.models_metadata:
                return self.models_metadata
            
            # Enhanced model metadata
            self.models_metadata = {
                'models': {
                    'math_specialist': {
                        'type': 'specialist',
                        'domain': 'mathematics',
                        'tasks': ['mathematical', 'scientific'],
                        'performance_metrics': {'accuracy': 0.95, 'speed': 'fast'},
                        'memory_requirements': '8GB',
                        'specialization_score': 0.95
                    },
                    'creative_specialist': {
                        'type': 'specialist', 
                        'domain': 'creative_writing',
                        'tasks': ['creative_writing', 'conversational'],
                        'performance_metrics': {'creativity': 0.90, 'fluency': 0.88},
                        'memory_requirements': '6GB',
                        'specialization_score': 0.90
                    },
                    'reasoning_specialist': {
                        'type': 'specialist',
                        'domain': 'reasoning',
                        'tasks': ['reasoning', 'factual_qa'],
                        'performance_metrics': {'logical_consistency': 0.92, 'accuracy': 0.88},
                        'memory_requirements': '10GB',
                        'specialization_score': 0.92
                    },
                    'code_specialist': {
                        'type': 'specialist',
                        'domain': 'programming', 
                        'tasks': ['code_generation', 'reasoning'],
                        'performance_metrics': {'code_quality': 0.93, 'efficiency': 0.85},
                        'memory_requirements': '8GB',
                        'specialization_score': 0.93
                    },
                    'general_model': {
                        'type': 'general',
                        'domain': 'all',
                        'tasks': ['mathematical', 'creative_writing', 'factual_qa', 'reasoning', 'code_generation', 'scientific', 'conversational'],
                        'performance_metrics': {'general_capability': 0.75, 'versatility': 0.95},
                        'memory_requirements': '12GB',
                        'specialization_score': 0.75
                    }
                },
                'loading_config': asdict(self.config),
                'system_info': {
                    'available_models': 5,
                    'total_memory_required': '44GB',
                    'recommended_device': self.config.device,
                    'precision': self.config.precision
                },
                'capabilities': {
                    'multi_model_routing': True,
                    'load_balancing': True,
                    'automatic_fallback': True,
                    'performance_monitoring': True
                },
                'last_updated': time.time(),
                'status': 'ready'
            }
        
        return self.models_metadata
    
    def get_model_recommendations(self, task_type: str) -> List[str]:
        """Get recommended models for a specific task type"""
        if not self.models_metadata:
            self.load_models_metadata()
        
        recommendations = []
        for model_id, model_info in self.models_metadata['models'].items():
            if task_type in model_info['tasks']:
                recommendations.append({
                    'model_id': model_id,
                    'specialization_score': model_info['specialization_score'],
                    'performance_metrics': model_info['performance_metrics']
                })
        
        # Sort by specialization score
        recommendations.sort(key=lambda x: x['specialization_score'], reverse=True)
        return [rec['model_id'] for rec in recommendations]

class CacheManager:
    """Advanced caching system for processed data"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache_index(self):
        """Save cache index"""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key for data"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached data"""
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    # Remove invalid cache entry
                    self.cache_index.pop(cache_key, None)
                    cache_file.unlink(missing_ok=True)
        return None
    
    def cache_data(self, cache_key: str, data: Any, metadata: Dict[str, Any] = None):
        """Cache processed data"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.cache_index[cache_key] = {
                'timestamp': time.time(),
                'size': len(pickle.dumps(data)),
                'metadata': metadata or {}
            }
            self._save_cache_index()
            
        except Exception as e:
            print(f"Warning: Could not cache data: {e}")
    
    def cleanup_old_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        to_remove = []
        for cache_key, info in self.cache_index.items():
            if current_time - info['timestamp'] > max_age_seconds:
                to_remove.append(cache_key)
        
        for cache_key in to_remove:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_file.unlink(missing_ok=True)
            self.cache_index.pop(cache_key, None)
        
        if to_remove:
            self._save_cache_index()
            print(f"Cleaned up {len(to_remove)} old cache entries")

class IntegratedDataPipeline:
    """
    Production-ready integrated data processing pipeline
    Supports both research and production environments with full monitoring
    """
    
    def __init__(self, config: Optional[DataPipelineConfig] = None):
        self.config = config or DataPipelineConfig()
        self.logger = self._setup_logging()
        self.processor = DataProcessor()
        self.model_manager = ModelManager(self.config.model_loading)
        self.cache_manager = CacheManager(self.config.cache_dir) if self.config.cache_processed_data else None
        
        # Pipeline state
        self.current_stage = PipelineStage.INITIALIZATION
        self.is_initialized = False
        self.processing_stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'cache_hits': 0,
            'processing_time': 0.0,
            'last_batch_size': 0
        }
        
        # Thread safety
        self.processing_lock = threading.Lock()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(getattr(logging, self.config.log_level, logging.INFO))
        
        if not logger.handlers:
            # File handler
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / "data_pipeline.log")
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        return logger
    
    def initialize(self) -> bool:
        """Initialize the comprehensive data pipeline"""
        try:
            self.current_stage = PipelineStage.INITIALIZATION
            self.logger.info("🚀 Initializing integrated data pipeline...")
            
            # Create necessary directories
            self._create_directories()
            
            # Validate configuration
            self._validate_config()
            
            # Initialize components
            self._initialize_components()
            
            # Load models metadata
            self.model_manager.load_models_metadata()
            
            # Cleanup old cache if enabled
            if self.cache_manager:
                self.cache_manager.cleanup_old_cache()
            
            self.is_initialized = True
            self.current_stage = PipelineStage.COMPLETED
            self.logger.info("✅ Data pipeline initialized successfully")
            
            # Log system capabilities
            self._log_system_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline initialization failed: {e}")
            self.current_stage = PipelineStage.INITIALIZATION
            return False
    
    def _create_directories(self):
        """Create comprehensive directory structure"""
        dirs_to_create = [
            self.config.cache_dir,
            "data/processed",
            "data/raw",
            "data/features", 
            "logs",
            "models",
            "exports",
            "temp"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"📁 Created {len(dirs_to_create)} directories")
    
    def _validate_config(self):
        """Comprehensive configuration validation"""
        if self.config.preprocessing is None:
            self.config.preprocessing = PreprocessingConfig()
            self.logger.warning("⚠️ Using default preprocessing config")
            
        if self.config.model_loading is None:
            self.config.model_loading = ModelLoadingConfig()
            self.logger.warning("⚠️ Using default model loading config")
        
        # Validate numeric values
        assert self.config.preprocessing.batch_size > 0, "Batch size must be positive"
        assert self.config.preprocessing.max_sequence_length > 0, "Max sequence length must be positive"
        assert self.config.num_workers >= 1, "Number of workers must be at least 1"
        
        self.logger.info("✅ Configuration validated")
    
    def _initialize_components(self):
        """Initialize pipeline components"""
        self.logger.info("🔧 Initializing components...")
        
        # Test parallel processing capability
        if self.config.parallel_processing:
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future = executor.submit(lambda: time.sleep(0.1))
                    future.result(timeout=1.0)
                self.logger.info("✅ Parallel processing enabled")
            except Exception as e:
                self.logger.warning(f"⚠️ Parallel processing disabled: {e}")
                self.config.parallel_processing = False
    
    def _log_system_info(self):
        """Log comprehensive system information"""
        models_metadata = self.model_manager.models_metadata
        
        self.logger.info("📊 System Information:")
        self.logger.info(f"   Available models: {len(models_metadata.get('models', {}))}")
        self.logger.info(f"   Parallel processing: {self.config.parallel_processing}")
        self.logger.info(f"   Cache enabled: {self.cache_manager is not None}")
        self.logger.info(f"   Batch size: {self.config.preprocessing.batch_size}")
        self.logger.info(f"   Workers: {self.config.num_workers}")
    
    def process_text_data(self, texts: List[str], 
                         batch_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Process text data through the comprehensive pipeline
        
        Args:
            texts: List of input texts
            batch_callback: Optional callback for batch processing updates
            
        Returns:
            List of processed data dictionaries with comprehensive features
        """
        if not self.is_initialized:
            self.logger.warning("Pipeline not initialized, auto-initializing...")
            if not self.initialize():
                raise RuntimeError("Failed to initialize pipeline")
        
        start_time = time.time()
        self.current_stage = PipelineStage.TEXT_PREPROCESSING
        
        with self.processing_lock:
            self.logger.info(f"🔄 Processing {len(texts)} text samples...")
            
            # Check cache first
            cache_key = None
            if self.cache_manager:
                cache_key = self.cache_manager.get_cache_key(texts)
                cached_result = self.cache_manager.get_cached_data(cache_key)
                if cached_result:
                    self.processing_stats['cache_hits'] += 1
                    self.logger.info("✅ Retrieved results from cache")
                    return cached_result
            
            # Process data
            if self.config.parallel_processing and len(texts) > self.config.num_workers:
                processed_data = self._process_parallel(texts, batch_callback)
            else:
                processed_data = self._process_sequential(texts, batch_callback)
            
            # Validate and filter results
            self.current_stage = PipelineStage.VALIDATION
            valid_data = []
            for item in processed_data:
                if self.processor.validate_data(item, self.config.preprocessing):
                    valid_data.append(item)
                else:
                    self.processing_stats['failed_processed'] += 1
            
            # Cache results
            if self.cache_manager and cache_key:
                self.current_stage = PipelineStage.CACHING
                metadata = {
                    'original_count': len(texts),
                    'processed_count': len(valid_data),
                    'processing_time': time.time() - start_time
                }
                self.cache_manager.cache_data(cache_key, valid_data, metadata)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats.update({
                'total_processed': self.processing_stats['total_processed'] + len(texts),
                'successful_processed': self.processing_stats['successful_processed'] + len(valid_data),
                'processing_time': self.processing_stats['processing_time'] + processing_time,
                'last_batch_size': len(texts)
            })
            
            self.current_stage = PipelineStage.COMPLETED
            self.logger.info(f"✅ Successfully processed {len(valid_data)}/{len(texts)} samples in {processing_time:.2f}s")
            
            return valid_data
    
    def _process_sequential(self, texts: List[str], 
                           batch_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Sequential processing implementation"""
        processed_data = []
        batch_size = self.config.preprocessing.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for j, text in enumerate(batch):
                try:
                    result = self._process_single_text(text, i + j)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Error processing text {i+j}: {e}")
                    continue
            
            processed_data.extend(batch_results)
            
            if batch_callback:
                batch_callback(i // batch_size + 1, len(texts) // batch_size + 1)
            
            if self.config.log_progress:
                progress = (i + len(batch)) / len(texts) * 100
                self.logger.info(f"Progress: {progress:.1f}% ({i + len(batch)}/{len(texts)})")
        
        return processed_data
    
    def _process_parallel(self, texts: List[str], 
                         batch_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Parallel processing implementation"""
        processed_data = []
        batch_size = self.config.preprocessing.batch_size
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit batches for parallel processing
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                future = executor.submit(self._process_batch, batch, i)
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    batch_results = future.result(timeout=300)  # 5 minute timeout
                    processed_data.extend(batch_results)
                    
                    if batch_callback:
                        batch_callback(i + 1, len(futures))
                        
                except Exception as e:
                    self.logger.error(f"Batch {i} processing failed: {e}")
        
        return processed_data
    
    def _process_batch(self, batch: List[str], start_index: int) -> List[Dict[str, Any]]:
        """Process a single batch of texts"""
        results = []
        for j, text in enumerate(batch):
            try:
                result = self._process_single_text(text, start_index + j)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Error in batch processing text {start_index + j}: {e}")
                continue
        return results
    
    def _process_single_text(self, text: str, index: int) -> Dict[str, Any]:
        """Process a single text with comprehensive feature extraction"""
        
        # Stage 1: Text preprocessing
        processed_text = self.processor.clean_text(text, self.config.preprocessing)
        
        # Stage 2: Feature extraction
        self.current_stage = PipelineStage.FEATURE_EXTRACTION
        features = self.processor.extract_features(processed_text)
        
        # Stage 3: Create comprehensive data item
        data_item = {
            'id': index,
            'original_text': text,
            'processed_text': processed_text,
            'features': features,
            'metadata': {
                'processing_timestamp': time.time(),
                'pipeline_version': '2.0',
                'config_hash': hash(str(asdict(self.config.preprocessing))),
                'stage': self.current_stage.value
            }
        }
        
        # Add convenience fields for backward compatibility
        data_item.update({
            'length': features['length'],
            'word_count': features['word_count'],
            'timestamp': data_item['metadata']['processing_timestamp']
        })
        
        return data_item
    
    def get_model_recommendations(self, task_type: str) -> List[str]:
        """Get model recommendations for specific task type"""
        return self.model_manager.get_model_recommendations(task_type)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        base_stats = {
            'initialized': self.is_initialized,
            'current_stage': self.current_stage.value,
            'processing_stats': self.processing_stats.copy(),
            'config': {
                'batch_size': self.config.preprocessing.batch_size,
                'max_sequence_length': self.config.preprocessing.max_sequence_length,
                'parallel_processing': self.config.parallel_processing,
                'num_workers': self.config.num_workers,
                'cache_enabled': self.cache_manager is not None,
                'cache_dir': self.config.cache_dir
            },
            'capabilities': [
                'advanced_text_preprocessing',
                'comprehensive_feature_extraction', 
                'parallel_processing',
                'intelligent_caching',
                'model_metadata_management',
                'batch_processing',
                'progress_monitoring',
                'error_handling',
                'performance_tracking'
            ]
        }
        
        # Add models information
        if hasattr(self.model_manager, 'models_metadata') and self.model_manager.models_metadata:
            base_stats['models'] = {
                'available_count': len(self.model_manager.models_metadata.get('models', {})),
                'model_list': list(self.model_manager.models_metadata.get('models', {}).keys()),
                'system_info': self.model_manager.models_metadata.get('system_info', {})
            }
        
        # Add cache statistics
        if self.cache_manager:
            base_stats['cache'] = {
                'entries_count': len(self.cache_manager.cache_index),
                'cache_directory': str(self.cache_manager.cache_dir),
                'cache_hits': self.processing_stats['cache_hits']
            }
        
        return base_stats
    
    def export_processed_data(self, data: List[Dict[str, Any]], 
                             export_path: str, format: str = 'json') -> bool:
        """Export processed data to various formats"""
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            elif format.lower() == 'pickle':
                with open(export_file, 'wb') as f:
                    pickle.dump(data, f)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"✅ Exported {len(data)} items to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup pipeline resources"""
        self.logger.info("🧹 Cleaning up pipeline resources...")
        
        if self.cache_manager:
            self.cache_manager.cleanup_old_cache(max_age_hours=1)
        
        # Clean temp files
        temp_dir = Path("temp")
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(exist_ok=True)
        
        self.logger.info("✅ Cleanup completed")

# Convenience functions for easy import and backward compatibility
def create_pipeline(config: Optional[DataPipelineConfig] = None) -> IntegratedDataPipeline:
    """Create and initialize a comprehensive data pipeline"""
    pipeline = IntegratedDataPipeline(config)
    pipeline.initialize()
    return pipeline

def quick_process_texts(texts: List[str], 
                       parallel: bool = True) -> List[Dict[str, Any]]:
    """Quick text processing with optimized default config"""
    config = DataPipelineConfig()
    config.parallel_processing = parallel
    config.log_level = "WARNING"  # Reduce noise for quick processing
    
    pipeline = create_pipeline(config)
    return pipeline.process_text_data(texts)

def batch_process_files(file_paths: List[str], 
                       output_dir: str = "data/processed") -> bool:
    """Process multiple text files in batch"""
    pipeline = create_pipeline()
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            processed = pipeline.process_text_data([content])
            
            # Export processed data
            output_file = Path(output_dir) / f"{Path(file_path).stem}_processed.json"
            pipeline.export_processed_data(processed, str(output_file))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    return True

# Backward compatibility aliases
DataPipeline = IntegratedDataPipeline
ProcessingPipeline = IntegratedDataPipeline

def main():
    """Main function for testing and demonstration"""
    print("🚀 Integrated Data Pipeline - Demo Mode")
    print("=" * 50)
    
    # Demo data
    sample_texts = [
        "Calculate the derivative of f(x) = x^2 + 3x + 1",
        "Write a creative story about a time-traveling scientist",
        "Implement a binary search algorithm in Python", 
        "Analyze the pros and cons of renewable energy",
        "What is the capital of France?",
        "Hello! How are you doing today?",
        "Explain quantum mechanics in simple terms"
    ]
    
    try:
        # Create and test pipeline
        print("\n🔧 Creating pipeline...")
        pipeline = create_pipeline()
        
        # Test processing
        print(f"\n📝 Processing {len(sample_texts)} sample texts...")
        results = pipeline.process_text_data(sample_texts)
        
        # Show results
        print(f"\n✅ Successfully processed {len(results)} texts")
        
        # Display sample result
        if results:
            print(f"\n📊 Sample result:")
            sample = results[0]
            print(f"   Original: {sample['original_text'][:50]}...")
            print(f"   Processed: {sample['processed_text'][:50]}...")
            print(f"   Features: {len(sample['features'])} extracted")
            print(f"   Word count: {sample['word_count']}")
            print(f"   Complexity: {sample['features']['complexity_score']:.3f}")
        
        # Test model recommendations
        print(f"\n🤖 Model recommendations:")
        for task in ['mathematical', 'creative_writing', 'code_generation']:
            recommendations = pipeline.get_model_recommendations(task)
            print(f"   {task}: {recommendations[:2]}")
        
        # Show pipeline stats
        print(f"\n📈 Pipeline statistics:")
        stats = pipeline.get_pipeline_stats()
        print(f"   Processed: {stats['processing_stats']['successful_processed']}")
        print(f"   Cache hits: {stats['processing_stats']['cache_hits']}")
        print(f"   Available models: {stats.get('models', {}).get('available_count', 0)}")
        
        # Test export
        print(f"\n💾 Testing export functionality...")
        export_path = "temp/demo_export.json"
        success = pipeline.export_processed_data(results, export_path)
        if success:
            print(f"   ✅ Exported to {export_path}")
        
        print(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        if 'pipeline' in locals():
            pipeline.cleanup()

if __name__ == "__main__":
    main()