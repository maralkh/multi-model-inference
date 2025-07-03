# File: data/huggingface_data_loader.py
"""Load and process datasets from Hugging Face Hub"""

import json
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging

try:
    from datasets import load_dataset, Dataset, DatasetDict
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    print("⚠️ Hugging Face datasets not available. Install with: pip install datasets transformers")
    HF_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_data_generator import RealDataGenerator, DataSample

logger = logging.getLogger(__name__)

@dataclass
class HFDatasetConfig:
    """Configuration for Hugging Face dataset loading"""
    dataset_name: str
    subset: Optional[str] = None
    split: str = "train"
    max_samples: int = 1000
    text_column: str = "text"
    label_column: Optional[str] = None
    task_type: str = "general"
    preprocessing: Optional[str] = None  # None, "clean", "tokenize"

class PopularDatasets:
    """Popular Hugging Face datasets for different tasks"""
    
    # Text Classification / NLU
    CLASSIFICATION = [
        HFDatasetConfig("imdb", split="train", text_column="text", label_column="label", task_type="reasoning", max_samples=500),
        HFDatasetConfig("emotion", split="train", text_column="text", label_column="label", task_type="conversational", max_samples=300),
        HFDatasetConfig("ag_news", split="train", text_column="text", label_column="label", task_type="factual_qa", max_samples=400),
    ]
    
    # Question Answering
    QUESTION_ANSWERING = [
        HFDatasetConfig("squad", split="train", text_column="question", task_type="factual_qa", max_samples=300),
        HFDatasetConfig("squad_v2", split="train", text_column="question", task_type="factual_qa", max_samples=300),
        HFDatasetConfig("natural_questions", subset="default", split="train", text_column="question", task_type="factual_qa", max_samples=200),
    ]
    
    # Code Generation
    CODE_DATASETS = [
        HFDatasetConfig("code_search_net", subset="python", split="train", text_column="func_documentation_string", task_type="code_generation", max_samples=300),
        HFDatasetConfig("codeparrot/github-code", subset="Python", split="train", text_column="code", task_type="code_generation", max_samples=200),
    ]
    
    # Mathematics
    MATH_DATASETS = [
        HFDatasetConfig("competition_math", split="train", text_column="problem", task_type="mathematical", max_samples=300),
        HFDatasetConfig("gsm8k", split="train", text_column="question", task_type="mathematical", max_samples=400),
        HFDatasetConfig("math_qa", split="train", text_column="Problem", task_type="mathematical", max_samples=300),
    ]
    
    # Creative Writing
    CREATIVE_DATASETS = [
        HFDatasetConfig("roneneldan/TinyStories", split="train", text_column="text", task_type="creative_writing", max_samples=300),
        HFDatasetConfig("wikitext", subset="wikitext-2-v1", split="train", text_column="text", task_type="creative_writing", max_samples=200),
        HFDatasetConfig("bookcorpus", split="train", text_column="text", task_type="creative_writing", max_samples=100),
    ]
    
    # Science
    SCIENCE_DATASETS = [
        HFDatasetConfig("sciq", split="train", text_column="question", task_type="scientific", max_samples=300),
        HFDatasetConfig("ai2_arc", subset="ARC-Easy", split="train", text_column="question", task_type="scientific", max_samples=200),
        HFDatasetConfig("ai2_arc", subset="ARC-Challenge", split="train", text_column="question", task_type="scientific", max_samples=200),
    ]
    
    # Conversational
    CONVERSATIONAL_DATASETS = [
        HFDatasetConfig("daily_dialog", split="train", text_column="dialog", task_type="conversational", max_samples=300),
        HFDatasetConfig("empathetic_dialogues", split="train", text_column="utterance", task_type="conversational", max_samples=300),
        HFDatasetConfig("blended_skill_talk", split="train", text_column="free_messages", task_type="conversational", max_samples=200),
    ]

class HuggingFaceDataLoader:
    """Load and process Hugging Face datasets"""
    
    def __init__(self, cache_dir: str = "./hf_cache", output_dir: str = "./hf_data"):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.popular_datasets = PopularDatasets()
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not HF_AVAILABLE:
            logger.warning("Hugging Face datasets not available")
    
    def load_single_dataset(self, config: HFDatasetConfig) -> List[DataSample]:
        """Load a single dataset from Hugging Face"""
        
        if not HF_AVAILABLE:
            logger.error("Hugging Face datasets not available")
            return []
        
        try:
            logger.info(f"Loading {config.dataset_name}...")
            
            # Load dataset
            if config.subset:
                dataset = load_dataset(
                    config.dataset_name, 
                    config.subset, 
                    split=config.split,
                    cache_dir=str(self.cache_dir)
                )
            else:
                dataset = load_dataset(
                    config.dataset_name,
                    split=config.split, 
                    cache_dir=str(self.cache_dir)
                )
            
            # Sample if too large
            if len(dataset) > config.max_samples:
                indices = random.sample(range(len(dataset)), config.max_samples)
                dataset = dataset.select(indices)
            
            # Convert to DataSample format
            samples = []
            
            for i, item in enumerate(dataset):
                try:
                    # Extract text
                    text = self._extract_text(item, config)
                    
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    # Create sample
                    sample = DataSample(
                        text=text,
                        task_type=config.task_type,
                        domain=f"hf_{config.dataset_name}",
                        difficulty="medium",  # Default
                        expected_length=50,   # Estimate
                        quality_score=0.75,   # Default
                        metadata={
                            'source': 'huggingface',
                            'dataset_name': config.dataset_name,
                            'subset': config.subset,
                            'split': config.split,
                            'original_index': i,
                            'preprocessing': config.preprocessing,
                            'loaded_at': time.time()
                        }
                    )
                    
                    # Add label if available
                    if config.label_column and config.label_column in item:
                        sample.metadata['label'] = item[config.label_column]
                    
                    samples.append(sample)
                    
                except Exception as e:
                    logger.warning(f"Failed to process item {i}: {e}")
                    continue
            
            logger.info(f"✅ Loaded {len(samples)} samples from {config.dataset_name}")
            return samples
            
        except Exception as e:
            logger.error(f"❌ Failed to load {config.dataset_name}: {e}")
            return []
    
    def _extract_text(self, item: Dict, config: HFDatasetConfig) -> str:
        """Extract text from dataset item"""
        
        # Try primary text column
        if config.text_column in item:
            text = item[config.text_column]
            
            # Handle different data types
            if isinstance(text, str):
                return text.strip()
            elif isinstance(text, list):
                return " ".join(str(x) for x in text)
            else:
                return str(text)
        
        # Fallback strategies
        text_fields = ['text', 'question', 'problem', 'dialog', 'utterance', 'content', 'body', 'description']
        
        for field in text_fields:
            if field in item:
                text = item[field]
                if isinstance(text, str) and text.strip():
                    return text.strip()
        
        # Last resort: combine available text fields
        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and len(value.strip()) > 5:
                text_parts.append(value.strip())
        
        return " ".join(text_parts[:3])  # Limit to avoid too long text
    
    def load_popular_datasets(self, categories: List[str] = None) -> Dict[str, List[DataSample]]:
        """Load popular datasets by category"""
        
        if not HF_AVAILABLE:
            logger.error("Hugging Face datasets not available")
            return {}
        
        available_categories = {
            'classification': self.popular_datasets.CLASSIFICATION,
            'qa': self.popular_datasets.QUESTION_ANSWERING,
            'code': self.popular_datasets.CODE_DATASETS,
            'math': self.popular_datasets.MATH_DATASETS,
            'creative': self.popular_datasets.CREATIVE_DATASETS,
            'science': self.popular_datasets.SCIENCE_DATASETS,
            'conversational': self.popular_datasets.CONVERSATIONAL_DATASETS
        }
        
        if categories is None:
            categories = list(available_categories.keys())
        
        results = {}
        
        for category in categories:
            if category not in available_categories:
                logger.warning(f"Unknown category: {category}")
                continue
            
            logger.info(f"📚 Loading {category} datasets...")
            category_samples = []
            
            for config in available_categories[category]:
                samples = self.load_single_dataset(config)
                category_samples.extend(samples)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            results[category] = category_samples
            logger.info(f"✅ Loaded {len(category_samples)} samples for {category}")
        
        return results
    
    def load_custom_dataset(self, dataset_name: str, **kwargs) -> List[DataSample]:
        """Load a custom dataset with flexible parameters"""
        
        config = HFDatasetConfig(
            dataset_name=dataset_name,
            **kwargs
        )
        
        return self.load_single_dataset(config)
    
    def save_hf_data(self, data: Dict[str, List[DataSample]], prefix: str = "hf") -> Dict[str, int]:
        """Save Hugging Face data to files"""
        
        stats = {}
        
        for category, samples in data.items():
            if not samples:
                continue
            
            # Create category directory
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Save as JSON
            json_file = category_dir / f"{prefix}_{category}_data.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(sample) for sample in samples], f, indent=2, ensure_ascii=False)
            
            # Save simplified CSV
            csv_file = category_dir / f"{prefix}_{category}_data.csv"
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("text,task_type,domain,source\n")
                for sample in samples:
                    # Escape CSV properly
                    text = sample.text.replace('"', '""').replace('\n', ' ')
                    f.write(f'"{text}",{sample.task_type},{sample.domain},{sample.metadata["source"]}\n')
            
            stats[category] = len(samples)
        
        # Save summary
        summary = {
            'source': 'huggingface',
            'loaded_at': time.time(),
            'categories': stats,
            'total_samples': sum(stats.values())
        }
        
        summary_file = self.output_dir / f"{prefix}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return stats

class CombinedDataManager:
    """Manage both real generated data and Hugging Face data"""
    
    def __init__(self, base_dir: str = "./combined_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.real_generator = RealDataGenerator(str(self.base_dir / "real_data"))
        self.hf_loader = HuggingFaceDataLoader(
            cache_dir=str(self.base_dir / "hf_cache"),
            output_dir=str(self.base_dir / "hf_data")
        )
    
    def generate_all_data(self, include_hf: bool = True, hf_categories: List[str] = None) -> Dict[str, Any]:
        """Generate both real and HF data"""
        
        logger.info("🚀 Starting Combined Data Generation")
        
        results = {
            'real_data': {},
            'hf_data': {},
            'combined_stats': {}
        }
        
        # Generate real data
        logger.info("📝 Generating real data...")
        try:
            real_stats = self.real_generator.generate_all_data()
            results['real_data'] = real_stats
            logger.info(f"✅ Real data: {sum(real_stats.values())} samples")
        except Exception as e:
            logger.error(f"❌ Real data generation failed: {e}")
            results['real_data'] = {}
        
        # Load HF data
        if include_hf and HF_AVAILABLE:
            logger.info("📚 Loading Hugging Face data...")
            try:
                hf_data = self.hf_loader.load_popular_datasets(hf_categories)
                hf_stats = self.hf_loader.save_hf_data(hf_data)
                results['hf_data'] = hf_stats
                logger.info(f"✅ HF data: {sum(hf_stats.values())} samples")
            except Exception as e:
                logger.error(f"❌ HF data loading failed: {e}")
                results['hf_data'] = {}
        
        # Create combined datasets
        logger.info("🔄 Creating combined datasets...")
        combined_stats = self._create_combined_datasets(results['real_data'], results['hf_data'])
        results['combined_stats'] = combined_stats
        
        # Generate final summary
        self._generate_final_summary(results)
        
        total_samples = sum(results['real_data'].values()) + sum(results['hf_data'].values())
        logger.info(f"🎉 Combined data generation complete! Total: {total_samples} samples")
        
        return results
    
    def _create_combined_datasets(self, real_stats: Dict, hf_stats: Dict) -> Dict[str, int]:
        """Create combined training/testing datasets"""
        
        combined_dir = self.base_dir / "combined"
        combined_dir.mkdir(exist_ok=True)
        
        # Task type mapping for combining
        task_mapping = {
            'mathematical': ['math'],
            'creative_writing': ['creative'],
            'code_generation': ['code'],
            'scientific': ['science'],
            'reasoning': ['classification', 'qa'],
            'factual_qa': ['qa'],
            'conversational': ['conversational']
        }
        
        combined_datasets = {}
        
        for real_task, hf_categories in task_mapping.items():
            combined_samples = []
            
            # Load real data
            try:
                real_file = self.real_generator.base_dir / real_task / f"{real_task}_data.json"
                if real_file.exists():
                    with open(real_file, 'r', encoding='utf-8') as f:
                        real_samples = json.load(f)
                        combined_samples.extend(real_samples)
            except Exception as e:
                logger.warning(f"Failed to load real {real_task}: {e}")
            
            # Load HF data
            for hf_cat in hf_categories:
                try:
                    hf_file = self.hf_loader.output_dir / hf_cat / f"hf_{hf_cat}_data.json"
                    if hf_file.exists():
                        with open(hf_file, 'r', encoding='utf-8') as f:
                            hf_samples = json.load(f)
                            # Convert to common format
                            for sample in hf_samples:
                                sample['source'] = 'huggingface'
                            combined_samples.extend(hf_samples)
                except Exception as e:
                    logger.warning(f"Failed to load HF {hf_cat}: {e}")
            
            # Save combined dataset
            if combined_samples:
                # Shuffle for training
                random.shuffle(combined_samples)
                
                combined_file = combined_dir / f"combined_{real_task}.json"
                with open(combined_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_samples, f, indent=2, ensure_ascii=False)
                
                combined_datasets[f"combined_{real_task}"] = len(combined_samples)
        
        return combined_datasets
    
    def _generate_final_summary(self, results: Dict[str, Any]):
        """Generate final comprehensive summary"""
        
        summary = {
            'generation_time': time.time(),
            'data_sources': {
                'real_data': {
                    'total_samples': sum(results['real_data'].values()),
                    'datasets': results['real_data']
                },
                'huggingface_data': {
                    'total_samples': sum(results['hf_data'].values()),
                    'datasets': results['hf_data']
                },
                'combined_data': {
                    'total_samples': sum(results['combined_stats'].values()),
                    'datasets': results['combined_stats']
                }
            },
            'grand_total': sum(results['real_data'].values()) + sum(results['hf_data'].values()),
            'data_quality': {
                'real_data_quality': 'high (curated)',
                'hf_data_quality': 'variable (popular datasets)',
                'diversity': 'very high',
                'coverage': 'comprehensive'
            }
        }
        
        summary_file = self.base_dir / "comprehensive_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Generate README
        readme_content = self._generate_combined_readme(summary)
        readme_file = self.base_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _generate_combined_readme(self, summary: Dict) -> str:
        """Generate comprehensive README"""
        
        return f"""# Combined Dataset Collection

Generated on: {time.ctime(summary['generation_time'])}
Grand Total: {summary['grand_total']} samples

## Data Sources

### 📝 Real Generated Data
- **Total**: {summary['data_sources']['real_data']['total_samples']} samples
- **Quality**: {summary['data_quality']['real_data_quality']}
- **Content**: Carefully curated examples across all task types

### 📚 Hugging Face Data  
- **Total**: {summary['data_sources']['huggingface_data']['total_samples']} samples
- **Quality**: {summary['data_quality']['hf_data_quality']}
- **Content**: Popular datasets from HF Hub

### 🔄 Combined Data
- **Total**: {summary['data_sources']['combined_data']['total_samples']} datasets
- **Format**: Merged real + HF data by task type
- **Usage**: Ready for training/testing

## Directory Structure

```
combined_data/
├── real_data/              # Generated real data
│   ├── mathematical/
│   ├── creative_writing/
│   ├── code_generation/
│   └── ...
├── hf_data/               # Hugging Face data
│   ├── math/
│   ├── qa/
│   ├── code/
│   └── ...
├── combined/              # Merged datasets
│   ├── combined_mathematical.json
│   ├── combined_creative_writing.json
│   └── ...
├── hf_cache/             # HF dataset cache
└── comprehensive_summary.json
```

## Usage Examples

```python
# Load combined dataset
import json
with open('combined_data/combined/combined_mathematical.json') as f:
    math_data = json.load(f)

# Use with data manager
from data.huggingface_data_loader import CombinedDataManager
manager = CombinedDataManager()
results = manager.generate_all_data(include_hf=True)
```

## Data Quality

- **Diversity**: {summary['data_quality']['diversity']}
- **Coverage**: {summary['data_quality']['coverage']}
- **Real Data**: Curated for quality and accuracy
- **HF Data**: Popular, community-validated datasets

## Supported Tasks

- Mathematical problem solving
- Creative writing and storytelling  
- Code generation and programming
- Scientific explanations
- Logical reasoning and analysis
- Factual question answering
- Conversational dialogue

---

Generated by Multi-Model Inference Data Pipeline
"""

def run_real_data_only():
    """Run with real data generator only"""
    
    print("🚀 Running Real Data Generation Only")
    print("=" * 50)
    
    generator = RealDataGenerator()
    stats = generator.generate_all_data()
    
    print(f"\n📊 Results:")
    for dataset, count in stats.items():
        print(f"   {dataset}: {count} samples")
    
    return stats

def run_hf_data_only():
    """Run with Hugging Face data only"""
    
    print("🚀 Running Hugging Face Data Loading Only")
    print("=" * 50)
    
    if not HF_AVAILABLE:
        print("❌ Hugging Face datasets not available")
        print("Install with: pip install datasets transformers")
        return {}
    
    loader = HuggingFaceDataLoader()
    
    # Load subset for testing
    categories = ['math', 'qa', 'code']
    data = loader.load_popular_datasets(categories)
    stats = loader.save_hf_data(data)
    
    print(f"\n📊 Results:")
    for category, count in stats.items():
        print(f"   {category}: {count} samples")
    
    return stats

def run_combined_data():
    """Run combined data generation"""
    
    print("🚀 Running Combined Data Generation")
    print("=" * 50)
    
    manager = CombinedDataManager()
    results = manager.generate_all_data(
        include_hf=HF_AVAILABLE,
        hf_categories=['math', 'qa', 'code', 'science']  # Subset for testing
    )
    
    print(f"\n📊 Final Results:")
    print(f"Real Data: {sum(results['real_data'].values())} samples")
    print(f"HF Data: {sum(results['hf_data'].values())} samples") 
    print(f"Combined: {sum(results['combined_stats'].values())} datasets")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Loading Options")
    parser.add_argument("--mode", choices=["real", "hf", "combined"], default="combined",
                       help="Data loading mode")
    
    args = parser.parse_args()
    
    if args.mode == "real":
        run_real_data_only()
    elif args.mode == "hf":
        run_hf_data_only()
    else:
        run_combined_data()