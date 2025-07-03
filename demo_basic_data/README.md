# Real Data for Multi-Model Inference

Generated on: Thu Jul  3 04:25:16 2025
Total Samples: 110

## Dataset Structure

### Task Types
- **Mathematical**: 7 samples
- **Creative Writing**: 6 samples  
- **Code Generation**: 5 samples
- **Scientific**: 6 samples
- **Reasoning**: 4 samples
- **Factual Q&A**: 4 samples
- **Conversational**: 3 samples

### Difficulty Levels
- **Easy**: 12 samples
- **Medium**: 19 samples
- **Hard**: 4 samples

### Domain Coverage
- **Mathematics**: Arithmetic, Algebra, Calculus, Statistics
- **Science**: Biology, Chemistry, Physics  
- **Programming**: Python, JavaScript, SQL, Data Structures
- **Creative**: Fiction, Poetry, Character Development, Dialogue
- **Analysis**: Logic, Critical Thinking, Problem Solving, Ethics

## File Structure

```
real_data/
├── mathematical/           # Math-specific datasets
├── creative_writing/       # Creative content datasets
├── code_generation/        # Programming datasets
├── scientific/            # Science datasets
├── reasoning/             # Analysis datasets
├── factual_qa/           # Q&A datasets
├── conversational/       # Chat datasets
├── mixed/                # Mixed training/testing sets
├── benchmarks/           # Difficulty-based benchmarks
├── domains/              # Domain-specific collections
├── data_summary.json     # Statistics and metadata
└── README.md            # This file
```

## Usage

```python
from data.real_data_generator import RealDataGenerator

# Generate all data
generator = RealDataGenerator()
stats = generator.generate_all_data()

# Load specific dataset
import json
with open('real_data/mathematical/mathematical_data.json') as f:
    math_data = json.load(f)
```

## Quality Metrics

- **Average Quality Score**: 0.88
- **Coverage**: comprehensive
- **Diversity**: high

## Data Sources

All data samples are carefully crafted to represent real-world scenarios and academic standards across multiple domains.
