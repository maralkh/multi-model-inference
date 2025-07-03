# File: core/updated_classifier_manifold.py
"""Updated classifier and manifold learner with real data integration"""

import os
import sys
import json
import time
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, asdict
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import original components
from core.input_types import TaskType, InputAnalysis, ManifoldLearningConfig
from core.enhanced_input_classifier import EnhancedInputClassifier
from core.manifold_learner import ManifoldLearner
from core.geometric_embeddings import GeometricBayesianManifoldLearner

# Import data components
try:
    from data.integrated_data_pipeline import ProcessedDataSample
    DATA_TYPES_AVAILABLE = True
except ImportError:
    DATA_TYPES_AVAILABLE = False
    # Create minimal substitute
    ProcessedDataSample = dict

logger = logging.getLogger(__name__)

@dataclass 
class RealDataTrainingConfig:
    """Configuration for training on real data"""
    # Data sources
    use_processed_data: bool = True
    use_hf_data: bool = True
    use_synthetic_data: bool = True
    
    # Training parameters
    max_training_samples: int = 5000
    validation_split: float = 0.2
    min_samples_per_task: int = 50
    
    # Quality filtering
    min_quality_score: float = 0.5
    filter_duplicates: bool = True
    filter_short_texts: bool = True
    min_text_length: int = 20
    
    # Manifold learning
    enable_manifold_pretraining: bool = True
    manifold_batch_size: int = 100
    online_learning_rate: float = 0.1
    
    # Performance tracking
    enable_validation: bool = True
    validation_frequency: int = 100
    early_stopping_patience: int = 5

class UpdatedInputClassifier(EnhancedInputClassifier):
    """Enhanced input classifier with real data integration"""
    
    def __init__(self, 
                 real_data_config: RealDataTrainingConfig = None,
                 enable_manifold_learning: bool = True,
                 manifold_config: ManifoldLearningConfig = None,
                 auto_load_data: bool = True):
        
        # Adjust manifold config based on expected data size
        if manifold_config is None:
            manifold_config = ManifoldLearningConfig(
                embedding_dim=16,  # Smaller dimension for small datasets
                manifold_method="auto",
                enable_online_learning=True,
                enable_clustering=True
            )
        
        # Initialize parent
        super().__init__(
            enable_manifold_learning=enable_manifold_learning,
            manifold_config=manifold_config
        )
        
        self.real_data_config = real_data_config or RealDataTrainingConfig()
        
        # Enhanced tracking
        self.training_metadata = {
            'total_samples_processed': 0,
            'samples_per_task': defaultdict(int),
            'data_sources_used': [],
            'training_history': [],
            'validation_scores': [],
            'last_training_time': None
        }
        
        self.real_data_cache = {}
        self.validation_data = []
        
        # Auto-load real data if available
        if auto_load_data:
            self.load_and_train_on_real_data()
    
    def load_and_train_on_real_data(self) -> Dict[str, Any]:
        """Load and train on available real data"""
        
        logger.info("📚 Loading real data for classifier training...")
        
        training_samples = []
        data_sources = []
        
        # Load processed data
        if self.real_data_config.use_processed_data:
            processed_samples = self._load_processed_data()
            if processed_samples:
                training_samples.extend(processed_samples)
                data_sources.append("processed_data")
                logger.info(f"✅ Loaded {len(processed_samples)} processed samples")
        
        # Load HF data
        if self.real_data_config.use_hf_data:
            hf_samples = self._load_hf_data()
            if hf_samples:
                training_samples.extend(hf_samples)
                data_sources.append("huggingface_data")
                logger.info(f"✅ Loaded {len(hf_samples)} HuggingFace samples")
        
        # Generate synthetic data if needed
        if self.real_data_config.use_synthetic_data or len(training_samples) < 100:
            synthetic_samples = self._generate_enhanced_synthetic_data()
            training_samples.extend(synthetic_samples)
            data_sources.append("synthetic_data")
            logger.info(f"✅ Generated {len(synthetic_samples)} synthetic samples")
        
        if not training_samples:
            logger.warning("⚠️ No training data found, using minimal synthetic data")
            return {'error': 'No training data available'}
        
        # Process and train
        training_results = self._process_and_train(training_samples, data_sources)
        
        logger.info(f"🎓 Training complete: {training_results['total_samples']} samples from {len(data_sources)} sources")
        return training_results
    
    def _load_processed_data(self) -> List[Dict[str, Any]]:
        """Load processed data from integrated pipeline"""
        
        samples = []
        processed_data_paths = [
            "./processed_data/processed/train_processed.json",
            "./combined_data/processed/train_processed.json", 
            "./real_data/processed/train_processed.json"
        ]
        
        for data_path in processed_data_paths:
            if Path(data_path).exists():
                try:
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Convert to standard format
                    for item in data:
                        if isinstance(item, dict) and 'processed_text' in item:
                            samples.append({
                                'text': item['processed_text'],
                                'task_type': item.get('task_type', 'general'),
                                'quality_score': item.get('quality_score', 0.7),
                                'source': 'processed_data'
                            })
                    
                    logger.info(f"Loaded {len(data)} samples from {data_path}")
                    break  # Use first available file
                    
                except Exception as e:
                    logger.warning(f"Failed to load {data_path}: {e}")
                    continue
        
        return samples
    
    def _load_hf_data(self) -> List[Dict[str, Any]]:
        """Load HuggingFace data"""
        
        samples = []
        hf_data_paths = [
            "./hf_data",
            "./combined_data/hf_data",
            "./processed_data/raw_data/hf_data"
        ]
        
        for base_path in hf_data_paths:
            base_dir = Path(base_path)
            if base_dir.exists():
                # Look for HF data files
                for category_dir in base_dir.glob("*/"):
                    if category_dir.is_dir():
                        for json_file in category_dir.glob("hf_*.json"):
                            try:
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                
                                # Convert HF format
                                for item in data:
                                    if isinstance(item, dict) and 'text' in item:
                                        samples.append({
                                            'text': item['text'],
                                            'task_type': item.get('task_type', 'general'),
                                            'quality_score': item.get('quality_score', 0.6),
                                            'source': 'huggingface'
                                        })
                                
                                logger.debug(f"Loaded {len(data)} samples from {json_file.name}")
                                
                            except Exception as e:
                                logger.warning(f"Failed to load {json_file}: {e}")
                                continue
                
                if samples:
                    break  # Use first available data source
        
        return samples
    
    def _generate_enhanced_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate enhanced synthetic training data"""
        
        synthetic_data = {
            'mathematical': [
                "Calculate the area of a circle with radius 7 meters",
                "Solve the quadratic equation 3x² - 12x + 9 = 0",
                "Find the derivative of f(x) = 2x³ - 4x² + 7x - 1",
                "What is the integral of cos(2x) dx?",
                "Determine the limit of (x² - 1)/(x - 1) as x approaches 1",
                "Calculate the volume of a sphere with diameter 10 cm",
                "Find the slope of the line passing through (2,3) and (5,9)",
                "Solve the system: 2x + y = 7, x - y = 2",
                "What is 25% of 340 plus 15% of 280?",
                "Calculate compound interest: $1000 at 5% for 3 years"
            ],
            
            'creative_writing': [
                "Write a short story about a robot learning to paint",
                "Create a poem describing a sunset over the ocean",
                "Develop a character who can communicate with animals",
                "Write dialogue between two friends meeting after 10 years",
                "Describe a magical library where books come alive",
                "Create a story beginning with 'The last door finally opened'",
                "Write a haiku about artificial intelligence",
                "Describe a futuristic city from a child's perspective",
                "Create a mystery story set in a bookstore",
                "Write a love letter from the perspective of an old tree"
            ],
            
            'code_generation': [
                "Implement a binary search algorithm in Python",
                "Write a function to reverse a linked list",
                "Create a REST API endpoint for user authentication",
                "Implement a quicksort algorithm with error handling",
                "Write a Python class for a shopping cart",
                "Create a function to find all prime numbers up to n",
                "Implement a stack data structure with push/pop operations",
                "Write SQL query to find top 10 customers by revenue",
                "Create a JavaScript function for form validation",
                "Implement depth-first search for a graph"
            ],
            
            'scientific': [
                "Explain the process of photosynthesis in detail",
                "How do vaccines help build immunity?",
                "Describe the structure and function of DNA",
                "What causes the greenhouse effect?",
                "Explain how neurons transmit signals",
                "What is the water cycle and its importance?",
                "How do antibiotics work against bacteria?",
                "Explain the theory of plate tectonics",
                "What is genetic engineering and its applications?",
                "How do solar panels convert sunlight to electricity?"
            ],
            
            'reasoning': [
                "Analyze the pros and cons of renewable energy",
                "Evaluate the impact of social media on society",
                "Compare the advantages of remote vs office work",
                "Assess the ethical implications of genetic editing",
                "Analyze the causes and effects of climate change",
                "Evaluate different approaches to healthcare systems",
                "Compare democratic and authoritarian governance",
                "Assess the impact of automation on employment",
                "Analyze the benefits and risks of cryptocurrency",
                "Evaluate strategies for sustainable urban development"
            ],
            
            'factual_qa': [
                "What is the capital city of New Zealand?",
                "When did the Berlin Wall fall?",
                "Who painted the Mona Lisa?",
                "What is the largest ocean on Earth?",
                "How many planets are in our solar system?",
                "What is the chemical symbol for gold?",
                "Who wrote the novel '1984'?",
                "What is the highest mountain in North America?",
                "When was the World Wide Web invented?",
                "What is the smallest unit of matter?"
            ],
            
            'conversational': [
                "Hello! How can I assist you today?",
                "Thank you for your help, I really appreciate it",
                "Could you please explain that in simpler terms?",
                "I'm having trouble understanding this concept",
                "That's a great point, I hadn't considered that",
                "Can you recommend a good book to read?",
                "What's your opinion on this topic?",
                "I hope you have a wonderful day!",
                "Sorry, could you repeat that please?",
                "That sounds really interesting, tell me more"
            ]
        }
        
        samples = []
        for task_type, texts in synthetic_data.items():
            for text in texts:
                samples.append({
                    'text': text,
                    'task_type': task_type,
                    'quality_score': 0.8,
                    'source': 'synthetic'
                })
        
        return samples
    
    def _process_and_train(self, training_samples: List[Dict[str, Any]], data_sources: List[str]) -> Dict[str, Any]:
        """Process samples and train classifier"""
        
        # Filter samples
        filtered_samples = self._filter_training_samples(training_samples)
        
        # Split into train/validation
        train_samples, val_samples = self._split_train_validation(filtered_samples)
        
        # Extract texts and labels
        train_texts = [sample['text'] for sample in train_samples]
        train_labels = [sample['task_type'] for sample in train_samples]
        
        # Fit classifier
        self.fit_training_data(train_texts, train_labels)
        
        # Validation
        validation_results = {}
        if val_samples and self.real_data_config.enable_validation:
            validation_results = self._validate_classifier(val_samples)
        
        # Update metadata
        self.training_metadata.update({
            'total_samples_processed': len(filtered_samples),
            'samples_per_task': dict(Counter(sample['task_type'] for sample in filtered_samples)),
            'data_sources_used': data_sources,
            'last_training_time': time.time(),
            'validation_results': validation_results
        })
        
        # Pretrain manifold learner if enabled
        if self.real_data_config.enable_manifold_pretraining and self.manifold_learner:
            self._pretrain_manifold_learner(train_texts)
        
        return {
            'total_samples': len(filtered_samples),
            'train_samples': len(train_samples),
            'validation_samples': len(val_samples),
            'data_sources': data_sources,
            'validation_results': validation_results,
            'training_successful': True
        }
    
    def _filter_training_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter training samples based on quality"""
        
        filtered = []
        
        for sample in samples:
            # Quality score filter
            if sample.get('quality_score', 0) < self.real_data_config.min_quality_score:
                continue
            
            # Text length filter
            if self.real_data_config.filter_short_texts:
                if len(sample['text']) < self.real_data_config.min_text_length:
                    continue
            
            # Task type validation
            try:
                TaskType(sample['task_type'])
            except ValueError:
                # Skip samples with invalid task types
                continue
            
            filtered.append(sample)
        
        # Remove duplicates if enabled
        if self.real_data_config.filter_duplicates:
            seen_texts = set()
            deduplicated = []
            for sample in filtered:
                text_hash = hash(sample['text'])
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    deduplicated.append(sample)
            filtered = deduplicated
        
        # Ensure minimum samples per task
        task_counts = Counter(sample['task_type'] for sample in filtered)
        balanced_filtered = []
        task_samples = defaultdict(list)
        
        # Group by task
        for sample in filtered:
            task_samples[sample['task_type']].append(sample)
        
        # Balance tasks
        for task_type, samples in task_samples.items():
            min_samples = self.real_data_config.min_samples_per_task
            if len(samples) >= min_samples:
                # Take up to max samples
                max_samples = min(len(samples), self.real_data_config.max_training_samples // 7)  # 7 task types
                balanced_filtered.extend(samples[:max_samples])
            else:
                # Include all available samples
                balanced_filtered.extend(samples)
        
        logger.info(f"Filtered {len(samples)} → {len(filtered)} → {len(balanced_filtered)} samples")
        return balanced_filtered
    
    def _split_train_validation(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split samples into training and validation sets"""
        
        if not self.real_data_config.enable_validation:
            return samples, []
        
        val_split = self.real_data_config.validation_split
        split_index = int(len(samples) * (1 - val_split))
        
        # Shuffle samples
        import random
        random.shuffle(samples)
        
        train_samples = samples[:split_index]
        val_samples = samples[split_index:]
        
        return train_samples, val_samples
    
    def _validate_classifier(self, val_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate classifier performance"""
        
        correct = 0
        total = len(val_samples)
        task_results = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for sample in val_samples:
            try:
                analysis = self.analyze_input(sample['text'])
                predicted_task = analysis.task_type.value
                actual_task = sample['task_type']
                
                if predicted_task == actual_task:
                    correct += 1
                    task_results[actual_task]['correct'] += 1
                
                task_results[actual_task]['total'] += 1
                
            except Exception as e:
                logger.warning(f"Validation failed for sample: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Task-specific accuracies
        task_accuracies = {}
        for task, results in task_results.items():
            if results['total'] > 0:
                task_accuracies[task] = results['correct'] / results['total']
        
        validation_results = {
            'overall_accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'task_specific_accuracies': task_accuracies
        }
        
        logger.info(f"📊 Validation accuracy: {accuracy:.3f} ({correct}/{total})")
        return validation_results
    
    def _pretrain_manifold_learner(self, texts: List[str]):
        """Pretrain manifold learner on text data"""
        
        if not self.manifold_learner:
            return
        
        try:
            logger.info("🌐 Pretraining manifold learner...")
            
            # Batch processing for large datasets
            batch_size = self.real_data_config.manifold_batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                self.manifold_learner.learn_manifold_offline(batch_texts)
                
                if i % (batch_size * 5) == 0:  # Progress update every 5 batches
                    logger.info(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
            
            logger.info("✅ Manifold learner pretraining complete")
            
        except Exception as e:
            logger.warning(f"⚠️ Manifold pretraining failed: {e}")
    
    def get_training_metadata(self) -> Dict[str, Any]:
        """Get detailed training metadata"""
        
        metadata = self.training_metadata.copy()
        
        # Add classifier stats
        metadata['classifier_stats'] = self.get_classification_statistics()
        
        # Add manifold learner diagnostics if available
        if self.manifold_learner:
            try:
                metadata['manifold_diagnostics'] = self.manifold_learner.get_manifold_diagnostics()
            except Exception:
                metadata['manifold_diagnostics'] = {'error': 'Not available'}
        
        return metadata
    
    def save_classifier_state(self, filepath: str = None):
        """Save classifier state to file"""
        
        if filepath is None:
            filepath = f"classifier_state_{int(time.time())}.pkl"
        
        state = {
            'training_metadata': self.training_metadata,
            'real_data_config': asdict(self.real_data_config),
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'is_fitted': self.is_fitted,
            'training_history': list(self.training_history)
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"💾 Classifier state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save classifier state: {e}")
    
    def load_classifier_state(self, filepath: str):
        """Load classifier state from file"""
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.training_metadata = state.get('training_metadata', {})
            self.tfidf_vectorizer = state.get('tfidf_vectorizer', self.tfidf_vectorizer)
            self.is_fitted = state.get('is_fitted', False)
            
            if 'training_history' in state:
                self.training_history.extend(state['training_history'])
            
            logger.info(f"📂 Classifier state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load classifier state: {e}")

def create_updated_classifier(auto_train: bool = True, 
                            config: RealDataTrainingConfig = None) -> UpdatedInputClassifier:
    """Factory function to create updated classifier"""
    
    training_config = config or RealDataTrainingConfig(
        max_training_samples=500,  # Reasonable for demo
        min_quality_score=0.4,
        enable_manifold_pretraining=True
    )
    
    # Create manifold config that adapts to data size
    manifold_config = ManifoldLearningConfig(
        embedding_dim=12,  # Small dimension for robustness
        manifold_method="auto",
        enable_online_learning=True,
        enable_clustering=True
    )
    
    classifier = UpdatedInputClassifier(
        real_data_config=training_config,
        enable_manifold_learning=True,
        manifold_config=manifold_config,
        auto_load_data=auto_train
    )
    
    return classifier

def demo_updated_classifier():
    """Demo of updated classifier with real data"""
    
    print("🧠 Updated Classifier Demo")
    print("=" * 40)
    
    # Create classifier
    classifier = create_updated_classifier(auto_train=True)
    
    # Test classification
    test_cases = [
        "Solve the equation 2x + 5 = 13",
        "Write a story about space exploration", 
        "Implement a hash table in Python",
        "Explain how DNA replication works",
        "Analyze the impact of globalization",
        "What is the capital of Canada?",
        "Hello, how can I help you?"
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} cases...")
    
    for i, text in enumerate(test_cases, 1):
        analysis = classifier.analyze_input(text)
        print(f"[{i}] '{text[:30]}...'")
        print(f"    Task: {analysis.task_type.value}")
        print(f"    Confidence: {analysis.confidence_score:.3f}")
        print(f"    Manifold: {analysis.manifold_type}")
    
    # Show training metadata
    print(f"\n📊 Training Metadata:")
    metadata = classifier.get_training_metadata()
    print(f"    Total samples: {metadata['total_samples_processed']}")
    print(f"    Data sources: {metadata['data_sources_used']}")
    print(f"    Samples per task: {dict(metadata['samples_per_task'])}")
    
    if 'validation_results' in metadata and metadata['validation_results']:
        val_results = metadata['validation_results']
        print(f"    Validation accuracy: {val_results['overall_accuracy']:.3f}")
    
    return classifier

if __name__ == "__main__":
    demo_updated_classifier()