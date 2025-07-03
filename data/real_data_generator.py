# File: data/real_data_generator.py
"""Real data generator for multiple topics and domains"""

import json
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataSample:
    """Single data sample with metadata"""
    text: str
    task_type: str
    domain: str
    difficulty: str  # easy, medium, hard
    expected_length: int
    quality_score: float
    metadata: Dict[str, Any]

class RealDataTopics:
    """Real data for different topics and domains"""
    
    # Mathematical data
    MATHEMATICAL_DATA = [
        # Basic Arithmetic
        {
            'text': "Calculate 25% of 840 and then add 156 to the result.",
            'domain': 'arithmetic',
            'difficulty': 'easy',
            'expected_output': "25% of 840 = 0.25 × 840 = 210. Adding 156: 210 + 156 = 366.",
            'quality_score': 0.9
        },
        {
            'text': "A store offers a 30% discount on a $120 item. What is the final price after a 8% tax on the discounted amount?",
            'domain': 'word_problems',
            'difficulty': 'medium',
            'expected_output': "Discount: $120 × 0.30 = $36. Discounted price: $120 - $36 = $84. Tax: $84 × 0.08 = $6.72. Final price: $84 + $6.72 = $90.72.",
            'quality_score': 0.85
        },
        
        # Algebra
        {
            'text': "Solve for x: 3x + 7 = 2x - 4",
            'domain': 'algebra',
            'difficulty': 'easy',
            'expected_output': "3x + 7 = 2x - 4\n3x - 2x = -4 - 7\nx = -11",
            'quality_score': 0.95
        },
        {
            'text': "Find the vertex of the parabola y = 2x² - 8x + 3",
            'domain': 'algebra',
            'difficulty': 'medium',
            'expected_output': "For y = ax² + bx + c, vertex x = -b/(2a) = 8/(2×2) = 2. y = 2(2)² - 8(2) + 3 = 8 - 16 + 3 = -5. Vertex: (2, -5)",
            'quality_score': 0.88
        },
        
        # Calculus
        {
            'text': "Find the derivative of f(x) = 3x⁴ - 2x³ + x² - 5x + 7",
            'domain': 'calculus',
            'difficulty': 'medium',
            'expected_output': "f'(x) = 12x³ - 6x² + 2x - 5",
            'quality_score': 0.92
        },
        {
            'text': "Evaluate the integral ∫(2x + 3)dx from 0 to 4",
            'domain': 'calculus',
            'difficulty': 'medium',
            'expected_output': "∫(2x + 3)dx = x² + 3x + C. From 0 to 4: [4² + 3(4)] - [0² + 3(0)] = 16 + 12 = 28",
            'quality_score': 0.90
        },
        
        # Statistics
        {
            'text': "Calculate the mean, median, and mode of the dataset: 12, 15, 18, 15, 22, 19, 15, 25",
            'domain': 'statistics',
            'difficulty': 'easy',
            'expected_output': "Mean = (12+15+18+15+22+19+15+25)/8 = 141/8 = 17.625. Median = (15+18)/2 = 16.5. Mode = 15 (appears 3 times).",
            'quality_score': 0.87
        }
    ]
    
    # Creative Writing data
    CREATIVE_WRITING_DATA = [
        # Short Stories
        {
            'text': "Write a 200-word short story about a librarian who discovers that books in her library come alive at midnight.",
            'domain': 'fiction',
            'difficulty': 'medium',
            'expected_output': "Margaret had worked at the Riverside Library for fifteen years, but nothing had prepared her for what she witnessed that Tuesday night. As the grandfather clock chimed midnight, the books began to rustle...",
            'quality_score': 0.85
        },
        {
            'text': "Create a dystopian story beginning with: 'The last tree on Earth was planted today.'",
            'domain': 'science_fiction',
            'difficulty': 'hard',
            'expected_output': "The last tree on Earth was planted today, in a ceremony broadcast to the remaining domed cities. Sarah pressed her palm against the protective glass...",
            'quality_score': 0.90
        },
        
        # Poetry
        {
            'text': "Write a haiku about autumn rain.",
            'domain': 'poetry',
            'difficulty': 'easy',
            'expected_output': "Gentle autumn rain\nTaps against my window pane\nNature's lullaby",
            'quality_score': 0.88
        },
        {
            'text': "Compose a sonnet about the passage of time.",
            'domain': 'poetry',
            'difficulty': 'hard',
            'expected_output': "Time flows like a river swift and deep,\nCarrying moments on its endless stream...",
            'quality_score': 0.82
        },
        
        # Character Development
        {
            'text': "Create a detailed character profile for a 35-year-old detective who specializes in cold cases.",
            'domain': 'character_development',
            'difficulty': 'medium',
            'expected_output': "Detective Sarah Chen, 35, has been with the NYPD for twelve years. Known for her methodical approach and photographic memory, she transferred to Cold Cases after...",
            'quality_score': 0.91
        },
        
        # Dialogue
        {
            'text': "Write a tense dialogue between two characters: one is trying to convince the other to leave town immediately.",
            'domain': 'dialogue',
            'difficulty': 'medium',
            'expected_output': "\"You need to leave. Tonight.\" Marcus grabbed Elena's shoulders. \"I'm serious.\"\n\"What are you talking about?\" Elena pulled away...",
            'quality_score': 0.86
        }
    ]
    
    # Programming/Code data
    CODE_GENERATION_DATA = [
        # Python
        {
            'text': "Write a Python function to find the largest element in a list without using the built-in max() function.",
            'domain': 'python',
            'difficulty': 'easy',
            'expected_output': "def find_largest(lst):\n    if not lst:\n        return None\n    largest = lst[0]\n    for item in lst[1:]:\n        if item > largest:\n            largest = item\n    return largest",
            'quality_score': 0.92
        },
        {
            'text': "Implement a binary search algorithm in Python with proper error handling.",
            'domain': 'python',
            'difficulty': 'medium',
            'expected_output': "def binary_search(arr, target):\n    if not arr:\n        return -1\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            'quality_score': 0.95
        },
        
        # JavaScript
        {
            'text': "Create a JavaScript function that debounces another function (delays execution until after a specified time has passed).",
            'domain': 'javascript',
            'difficulty': 'medium',
            'expected_output': "function debounce(func, delay) {\n    let timeoutId;\n    return function(...args) {\n        clearTimeout(timeoutId);\n        timeoutId = setTimeout(() => func.apply(this, args), delay);\n    };\n}",
            'quality_score': 0.88
        },
        
        # SQL
        {
            'text': "Write a SQL query to find the top 5 customers by total purchase amount from tables 'customers' and 'orders'.",
            'domain': 'sql',
            'difficulty': 'medium',
            'expected_output': "SELECT c.customer_id, c.customer_name, SUM(o.amount) as total_amount\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nGROUP BY c.customer_id, c.customer_name\nORDER BY total_amount DESC\nLIMIT 5;",
            'quality_score': 0.90
        },
        
        # Data Structures
        {
            'text': "Implement a simple stack class in Python with push, pop, peek, and is_empty methods.",
            'domain': 'data_structures',
            'difficulty': 'easy',
            'expected_output': "class Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        self.items.append(item)\n    \n    def pop(self):\n        if self.is_empty():\n            raise IndexError(\"Stack is empty\")\n        return self.items.pop()\n    \n    def peek(self):\n        if self.is_empty():\n            raise IndexError(\"Stack is empty\")\n        return self.items[-1]\n    \n    def is_empty(self):\n        return len(self.items) == 0",
            'quality_score': 0.93
        }
    ]
    
    # Scientific data
    SCIENTIFIC_DATA = [
        # Biology
        {
            'text': "Explain the process of cellular respiration and its three main stages.",
            'domain': 'biology',
            'difficulty': 'medium',
            'expected_output': "Cellular respiration is the process by which cells break down glucose to produce ATP energy. The three stages are: 1) Glycolysis (glucose → pyruvate in cytoplasm), 2) Krebs Cycle (pyruvate → CO₂ in mitochondria), 3) Electron Transport Chain (oxygen → water, produces most ATP).",
            'quality_score': 0.89
        },
        {
            'text': "Describe the structure and function of DNA, including base pairing rules.",
            'domain': 'biology',
            'difficulty': 'medium',
            'expected_output': "DNA is a double-helix structure made of nucleotides containing phosphate, sugar (deoxyribose), and nitrogen bases (A, T, G, C). Base pairing rules: Adenine pairs with Thymine, Guanine pairs with Cytosine. DNA stores genetic information and serves as template for protein synthesis.",
            'quality_score': 0.91
        },
        
        # Chemistry
        {
            'text': "Balance the chemical equation: C₃H₈ + O₂ → CO₂ + H₂O",
            'domain': 'chemistry',
            'difficulty': 'easy',
            'expected_output': "C₃H₈ + 5O₂ → 3CO₂ + 4H₂O\nCarbon: 3 on each side\nHydrogen: 8 on each side\nOxygen: 10 on each side",
            'quality_score': 0.94
        },
        {
            'text': "Explain the difference between ionic and covalent bonds with examples.",
            'domain': 'chemistry',
            'difficulty': 'medium',
            'expected_output': "Ionic bonds form between metals and non-metals through electron transfer (e.g., NaCl: Na⁺ + Cl⁻). Covalent bonds form between non-metals through electron sharing (e.g., H₂O, CO₂). Ionic compounds conduct electricity when dissolved; covalent compounds typically don't.",
            'quality_score': 0.87
        },
        
        # Physics
        {
            'text': "Calculate the kinetic energy of a 1500 kg car traveling at 25 m/s.",
            'domain': 'physics',
            'difficulty': 'easy',
            'expected_output': "KE = ½mv²\nKE = ½ × 1500 kg × (25 m/s)²\nKE = ½ × 1500 × 625\nKE = 468,750 J = 468.75 kJ",
            'quality_score': 0.93
        },
        {
            'text': "Explain Einstein's theory of special relativity and its key principles.",
            'domain': 'physics',
            'difficulty': 'hard',
            'expected_output': "Special relativity has two key postulates: 1) Laws of physics are identical in all inertial reference frames, 2) Speed of light in vacuum is constant for all observers. Key consequences: time dilation, length contraction, mass-energy equivalence (E=mc²), and nothing can exceed light speed.",
            'quality_score': 0.88
        }
    ]
    
    # Reasoning/Analysis data
    REASONING_DATA = [
        # Logical Reasoning
        {
            'text': "Analyze this argument: 'All birds can fly. Penguins are birds. Therefore, penguins can fly.' Identify the logical error.",
            'domain': 'logical_reasoning',
            'difficulty': 'easy',
            'expected_output': "This is a syllogism with a false premise. The major premise 'All birds can fly' is incorrect - some birds like penguins, ostriches, and kiwis cannot fly. The logical structure is valid, but the conclusion is false due to the incorrect premise.",
            'quality_score': 0.90
        },
        
        # Critical Thinking
        {
            'text': "Evaluate the pros and cons of remote work from both employee and employer perspectives.",
            'domain': 'critical_thinking',
            'difficulty': 'medium',
            'expected_output': "Employee pros: flexibility, no commute, better work-life balance. Cons: isolation, communication challenges, home distractions. Employer pros: reduced office costs, wider talent pool, potentially higher productivity. Cons: management challenges, security concerns, team cohesion issues.",
            'quality_score': 0.85
        },
        
        # Problem Solving
        {
            'text': "A farmer has chickens and cows. The animals have 50 heads and 140 legs total. How many chickens and cows are there?",
            'domain': 'problem_solving',
            'difficulty': 'medium',
            'expected_output': "Let x = chickens, y = cows. Equations: x + y = 50 (heads), 2x + 4y = 140 (legs). From first equation: x = 50 - y. Substitute: 2(50-y) + 4y = 140, 100 - 2y + 4y = 140, 2y = 40, y = 20. Therefore: 20 cows and 30 chickens.",
            'quality_score': 0.92
        },
        
        # Ethics
        {
            'text': "Discuss the ethical implications of AI systems making decisions about medical treatments.",
            'domain': 'ethics',
            'difficulty': 'hard',
            'expected_output': "Key ethical concerns: accountability (who's responsible for AI decisions?), bias (training data may reflect societal biases), transparency (black-box decision making), autonomy (patient choice vs AI recommendations), justice (equal access to AI benefits), and the need for human oversight in life-critical decisions.",
            'quality_score': 0.87
        }
    ]
    
    # Factual Q&A data
    FACTUAL_QA_DATA = [
        # History
        {
            'text': "What were the main causes of World War I?",
            'domain': 'history',
            'difficulty': 'medium',
            'expected_output': "Main causes: 1) Militarism (arms race), 2) Alliance system (Triple Alliance vs Triple Entente), 3) Imperialism (colonial competition), 4) Nationalism (ethnic tensions in Balkans). The immediate trigger was assassination of Archduke Franz Ferdinand in Sarajevo, June 1914.",
            'quality_score': 0.89
        },
        
        # Geography
        {
            'text': "What is the longest river in South America and where does it flow?",
            'domain': 'geography',
            'difficulty': 'easy',
            'expected_output': "The Amazon River is the longest in South America (~6,400 km). It flows from the Peruvian Andes eastward across Brazil to the Atlantic Ocean, passing through Peru, Colombia, and Brazil.",
            'quality_score': 0.91
        },
        
        # Science Facts
        {
            'text': "How many bones are in the adult human body?",
            'domain': 'science_facts',
            'difficulty': 'easy',
            'expected_output': "An adult human has 206 bones. Babies are born with about 270 bones, but many fuse together as they grow.",
            'quality_score': 0.95
        },
        
        # Current Events (general knowledge)
        {
            'text': "What is cryptocurrency and how does blockchain technology work?",
            'domain': 'technology',
            'difficulty': 'medium',
            'expected_output': "Cryptocurrency is digital currency secured by cryptography. Blockchain is a distributed ledger technology that records transactions across multiple computers. Each 'block' contains transaction data and is linked to previous blocks, creating an immutable chain that prevents tampering.",
            'quality_score': 0.86
        }
    ]
    
    # Conversational data
    CONVERSATIONAL_DATA = [
        # Greetings
        {
            'text': "Hello! How are you doing today?",
            'domain': 'greetings',
            'difficulty': 'easy',
            'expected_output': "Hello! I'm doing well, thank you for asking. How are you today? Is there anything I can help you with?",
            'quality_score': 0.85
        },
        
        # Help Requests
        {
            'text': "I'm feeling overwhelmed with my workload. Any advice?",
            'domain': 'support',
            'difficulty': 'medium',
            'expected_output': "I understand that feeling. Try these strategies: 1) List all tasks and prioritize by urgency/importance, 2) Break large tasks into smaller steps, 3) Take regular breaks, 4) Delegate when possible, 5) Consider talking to your supervisor about workload. Remember, it's okay to ask for help.",
            'quality_score': 0.88
        },
        
        # Casual Chat
        {
            'text': "What do you think makes a good conversation?",
            'domain': 'casual_chat',
            'difficulty': 'medium',
            'expected_output': "Good conversations typically involve: active listening, genuine curiosity about the other person, asking thoughtful questions, sharing appropriate personal experiences, maintaining good eye contact, and being present in the moment. Balance between talking and listening is key.",
            'quality_score': 0.82
        }
    ]

class RealDataGenerator:
    """Generates comprehensive real data for training and testing"""
    
    def __init__(self, base_dir: str = "./real_data"):
        self.base_dir = Path(base_dir)
        self.topics = RealDataTopics()
        
        # Ensure directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all_data(self) -> Dict[str, int]:
        """Generate all real data files"""
        
        logger.info("🏭 Generating comprehensive real data...")
        
        stats = {}
        
        # Generate data for each task type
        stats['mathematical'] = self._generate_task_data('mathematical', self.topics.MATHEMATICAL_DATA)
        stats['creative_writing'] = self._generate_task_data('creative_writing', self.topics.CREATIVE_WRITING_DATA)
        stats['code_generation'] = self._generate_task_data('code_generation', self.topics.CODE_GENERATION_DATA)
        stats['scientific'] = self._generate_task_data('scientific', self.topics.SCIENTIFIC_DATA)
        stats['reasoning'] = self._generate_task_data('reasoning', self.topics.REASONING_DATA)
        stats['factual_qa'] = self._generate_task_data('factual_qa', self.topics.FACTUAL_QA_DATA)
        stats['conversational'] = self._generate_task_data('conversational', self.topics.CONVERSATIONAL_DATA)
        
        # Generate mixed datasets
        stats['mixed_training'] = self._generate_mixed_training_data()
        stats['mixed_testing'] = self._generate_mixed_testing_data()
        
        # Generate benchmark datasets
        stats['benchmark_easy'] = self._generate_benchmark_data('easy')
        stats['benchmark_medium'] = self._generate_benchmark_data('medium')
        stats['benchmark_hard'] = self._generate_benchmark_data('hard')
        
        # Generate domain-specific datasets
        stats['domain_math'] = self._generate_domain_specific_data('mathematics')
        stats['domain_science'] = self._generate_domain_specific_data('science')
        stats['domain_programming'] = self._generate_domain_specific_data('programming')
        
        # Generate summary
        self._generate_data_summary(stats)
        
        total_samples = sum(stats.values())
        logger.info(f"✅ Generated {total_samples} real data samples across {len(stats)} datasets")
        
        return stats
    
    def _generate_task_data(self, task_type: str, raw_data: List[Dict]) -> int:
        """Generate data for a specific task type"""
        
        task_dir = self.base_dir / task_type
        task_dir.mkdir(exist_ok=True)
        
        samples = []
        
        for item in raw_data:
            sample = DataSample(
                text=item['text'],
                task_type=task_type,
                domain=item['domain'],
                difficulty=item['difficulty'],
                expected_length=len(item['expected_output'].split()),
                quality_score=item['quality_score'],
                metadata={
                    'expected_output': item['expected_output'],
                    'created_at': time.time(),
                    'source': 'real_data_generator'
                }
            )
            samples.append(sample)
        
        # Save as JSON
        json_file = task_dir / f"{task_type}_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(sample) for sample in samples], f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        csv_file = task_dir / f"{task_type}_data.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'task_type', 'domain', 'difficulty', 'expected_length', 'quality_score', 'expected_output'])
            
            for sample in samples:
                writer.writerow([
                    sample.text,
                    sample.task_type,
                    sample.domain,
                    sample.difficulty,
                    sample.expected_length,
                    sample.quality_score,
                    sample.metadata['expected_output']
                ])
        
        logger.info(f"✅ Generated {len(samples)} samples for {task_type}")
        return len(samples)
    
    def _generate_mixed_training_data(self) -> int:
        """Generate mixed training dataset"""
        
        mixed_dir = self.base_dir / "mixed"
        mixed_dir.mkdir(exist_ok=True)
        
        # Collect samples from all categories
        all_samples = []
        
        for task_data in [
            self.topics.MATHEMATICAL_DATA,
            self.topics.CREATIVE_WRITING_DATA,
            self.topics.CODE_GENERATION_DATA,
            self.topics.SCIENTIFIC_DATA,
            self.topics.REASONING_DATA,
            self.topics.FACTUAL_QA_DATA,
            self.topics.CONVERSATIONAL_DATA
        ]:
            # Take subset of each category
            subset = random.sample(task_data, min(3, len(task_data)))
            all_samples.extend(subset)
        
        # Shuffle for training
        random.shuffle(all_samples)
        
        # Save mixed training data
        mixed_file = mixed_dir / "mixed_training.json"
        with open(mixed_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        return len(all_samples)
    
    def _generate_mixed_testing_data(self) -> int:
        """Generate mixed testing dataset"""
        
        # Create challenging test cases that span multiple domains
        test_cases = [
            {
                'text': "Write a Python function to calculate the compound interest, then explain the mathematical concept behind compound interest.",
                'task_type': 'mixed',
                'domain': 'math_programming',
                'difficulty': 'hard',
                'expected_output': "def compound_interest(principal, rate, time, n=1):\n    return principal * (1 + rate/n)**(n*time)\n\nCompound interest means earning interest on both the initial principal and previously earned interest. The formula A = P(1 + r/n)^(nt) shows exponential growth.",
                'quality_score': 0.92
            },
            {
                'text': "Analyze the ethics of genetic engineering and write a short dialogue between a scientist and an ethicist discussing CRISPR technology.",
                'task_type': 'mixed',
                'domain': 'science_ethics_creative',
                'difficulty': 'hard',
                'expected_output': "Genetic engineering raises concerns about safety, equality, and playing God. Dialogue: 'CRISPR could cure genetic diseases,' said Dr. Smith. 'But who decides what needs curing?' replied the ethicist. 'We must consider unintended consequences and societal impacts.'",
                'quality_score': 0.88
            }
        ]
        
        mixed_dir = self.base_dir / "mixed"
        test_file = mixed_dir / "mixed_testing.json"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        return len(test_cases)
    
    def _generate_benchmark_data(self, difficulty: str) -> int:
        """Generate benchmark data by difficulty level"""
        
        benchmark_dir = self.base_dir / "benchmarks"
        benchmark_dir.mkdir(exist_ok=True)
        
        # Collect samples of specific difficulty
        difficulty_samples = []
        
        all_data = [
            ('mathematical', self.topics.MATHEMATICAL_DATA),
            ('creative_writing', self.topics.CREATIVE_WRITING_DATA),
            ('code_generation', self.topics.CODE_GENERATION_DATA),
            ('scientific', self.topics.SCIENTIFIC_DATA),
            ('reasoning', self.topics.REASONING_DATA),
            ('factual_qa', self.topics.FACTUAL_QA_DATA),
            ('conversational', self.topics.CONVERSATIONAL_DATA)
        ]
        
        for task_type, data in all_data:
            filtered = [item for item in data if item['difficulty'] == difficulty]
            for item in filtered:
                item['task_type'] = task_type
            difficulty_samples.extend(filtered)
        
        # Save benchmark data
        benchmark_file = benchmark_dir / f"benchmark_{difficulty}.json"
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(difficulty_samples, f, indent=2, ensure_ascii=False)
        
        return len(difficulty_samples)
    
    def _generate_domain_specific_data(self, domain: str) -> int:
        """Generate domain-specific datasets"""
        
        domain_dir = self.base_dir / "domains"
        domain_dir.mkdir(exist_ok=True)
        
        domain_mapping = {
            'mathematics': ['arithmetic', 'algebra', 'calculus', 'statistics'],
            'science': ['biology', 'chemistry', 'physics'],
            'programming': ['python', 'javascript', 'sql', 'data_structures']
        }
        
        if domain not in domain_mapping:
            return 0
        
        target_domains = domain_mapping[domain]
        domain_samples = []
        
        # Collect samples from relevant domains
        all_data = [
            self.topics.MATHEMATICAL_DATA,
            self.topics.SCIENTIFIC_DATA,
            self.topics.CODE_GENERATION_DATA
        ]
        
        for data in all_data:
            filtered = [item for item in data if item['domain'] in target_domains]
            domain_samples.extend(filtered)
        
        # Save domain-specific data
        domain_file = domain_dir / f"domain_{domain}.json"
        with open(domain_file, 'w', encoding='utf-8') as f:
            json.dump(domain_samples, f, indent=2, ensure_ascii=False)
        
        return len(domain_samples)
    
    def _generate_data_summary(self, stats: Dict[str, int]):
        """Generate data summary and statistics"""
        
        summary = {
            'generation_time': time.time(),
            'total_samples': sum(stats.values()),
            'datasets': stats,
            'data_distribution': {
                'task_types': {
                    'mathematical': stats.get('mathematical', 0),
                    'creative_writing': stats.get('creative_writing', 0),
                    'code_generation': stats.get('code_generation', 0),
                    'scientific': stats.get('scientific', 0),
                    'reasoning': stats.get('reasoning', 0),
                    'factual_qa': stats.get('factual_qa', 0),
                    'conversational': stats.get('conversational', 0)
                },
                'difficulty_levels': {
                    'easy': stats.get('benchmark_easy', 0),
                    'medium': stats.get('benchmark_medium', 0),
                    'hard': stats.get('benchmark_hard', 0)
                },
                'domains': {
                    'mathematics': stats.get('domain_math', 0),
                    'science': stats.get('domain_science', 0),
                    'programming': stats.get('domain_programming', 0)
                }
            },
            'data_quality': {
                'avg_quality_score': 0.88,  # Estimated from samples
                'coverage': 'comprehensive',
                'diversity': 'high'
            }
        }
        
        summary_file = self.base_dir / "data_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Also create README
        readme_content = self._generate_readme(summary)
        readme_file = self.base_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _generate_readme(self, summary: Dict) -> str:
        """Generate README for the real data"""
        
        return f"""# Real Data for Multi-Model Inference

Generated on: {time.ctime(summary['generation_time'])}
Total Samples: {summary['total_samples']}

## Dataset Structure

### Task Types
- **Mathematical**: {summary['data_distribution']['task_types']['mathematical']} samples
- **Creative Writing**: {summary['data_distribution']['task_types']['creative_writing']} samples  
- **Code Generation**: {summary['data_distribution']['task_types']['code_generation']} samples
- **Scientific**: {summary['data_distribution']['task_types']['scientific']} samples
- **Reasoning**: {summary['data_distribution']['task_types']['reasoning']} samples
- **Factual Q&A**: {summary['data_distribution']['task_types']['factual_qa']} samples
- **Conversational**: {summary['data_distribution']['task_types']['conversational']} samples

### Difficulty Levels
- **Easy**: {summary['data_distribution']['difficulty_levels']['easy']} samples
- **Medium**: {summary['data_distribution']['difficulty_levels']['medium']} samples
- **Hard**: {summary['data_distribution']['difficulty_levels']['hard']} samples

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

- **Average Quality Score**: {summary['data_quality']['avg_quality_score']}
- **Coverage**: {summary['data_quality']['coverage']}
- **Diversity**: {summary['data_quality']['diversity']}

## Data Sources

All data samples are carefully crafted to represent real-world scenarios and academic standards across multiple domains.
"""

def generate_real_data():
    """Main function to generate real data"""
    
    print("🚀 Real Data Generation")
    print("=" * 40)
    
    generator = RealDataGenerator()
    stats = generator.generate_all_data()
    
    print(f"\n📊 Generation Summary:")
    for dataset, count in stats.items():
        print(f"   {dataset}: {count} samples")
    
    print(f"\n📁 Data saved to: {generator.base_dir}")
    print("🎉 Real data generation complete!")
    
    return stats

if __name__ == "__main__":
    generate_real_data()