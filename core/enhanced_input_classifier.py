# File: core/input_classifier.py
"""Enhanced input classification with manifold learning integration"""

import re
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import zipfile
import os

from .types import TaskType, InputAnalysis, ManifoldLearningConfig
from .manifold_learner import ManifoldLearner
from .geometric_embeddings import GeometricBayesianManifoldLearner

class GloVeEmbeddings:
    """GloVe word embeddings loader and manager"""
    
    def __init__(self, embedding_dim: int = 100, max_vocab_size: int = 10000):
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.word_vectors = {}
        self.vocab = set()
        self.is_loaded = False
        
        # GloVe download URLs (using smaller versions for demo)
        self.glove_urls = {
            50: "https://nlp.stanford.edu/data/glove.6B.zip",
            100: "https://nlp.stanford.edu/data/glove.6B.zip",
            200: "https://nlp.stanford.edu/data/glove.6B.zip",
            300: "https://nlp.stanford.edu/data/glove.6B.zip"
        }
        
        # Fallback: create synthetic embeddings if download fails
        self._create_synthetic_embeddings()
    
    def _create_synthetic_embeddings(self):
        """Create synthetic word embeddings as fallback"""
        
        print("🔧 Creating synthetic GloVe-like embeddings...")
        
        # Common words for different task types
        task_vocabularies = {
            'mathematical': ['solve', 'equation', 'calculate', 'formula', 'algebra', 'geometry',
                           'derivative', 'integral', 'matrix', 'vector', 'theorem', 'proof',
                           'function', 'variable', 'polynomial', 'trigonometry', 'calculus'],
            
            'creative': ['story', 'character', 'plot', 'narrative', 'fiction', 'creative',
                        'write', 'imagination', 'dialogue', 'scene', 'novel', 'poem',
                        'literary', 'metaphor', 'symbolism', 'theme', 'genre'],
            
            'code': ['algorithm', 'function', 'class', 'method', 'variable', 'loop',
                    'python', 'javascript', 'programming', 'code', 'debug', 'implement',
                    'software', 'system', 'framework', 'api', 'database'],
            
            'reasoning': ['analyze', 'compare', 'evaluate', 'assess', 'argue', 'reason',
                         'logic', 'critical', 'philosophical', 'ethical', 'complex',
                         'conclusion', 'evidence', 'argument', 'perspective'],
            
            'scientific': ['research', 'hypothesis', 'experiment', 'theory', 'data',
                          'analysis', 'method', 'study', 'evidence', 'observation',
                          'measurement', 'scientific', 'empirical', 'methodology'],
            
            'factual': ['explain', 'define', 'describe', 'information', 'fact',
                       'details', 'knowledge', 'what', 'who', 'when', 'where',
                       'why', 'how', 'tell', 'know'],
            
            'conversational': ['hello', 'hi', 'thanks', 'please', 'help', 'chat',
                             'talk', 'discuss', 'opinion', 'think', 'feel',
                             'conversation', 'social', 'friendly', 'casual']
        }
        
        # Create embeddings with task-type clustering
        np.random.seed(42)  # For reproducibility
        
        # Generate task-specific centroids
        task_centroids = {}
        for i, (task, words) in enumerate(task_vocabularies.items()):
            # Create centroid for each task type
            angle = (i / len(task_vocabularies)) * 2 * np.pi
            centroid = np.array([np.cos(angle), np.sin(angle)] + 
                              [np.random.normal(0, 0.1) for _ in range(self.embedding_dim - 2)])
            task_centroids[task] = centroid
        
        # Generate word vectors clustered around task centroids
        for task, words in task_vocabularies.items():
            centroid = task_centroids[task]
            
            for word in words:
                if word not in self.word_vectors:
                    # Add noise around centroid
                    noise = np.random.normal(0, 0.3, self.embedding_dim)
                    vector = centroid + noise
                    # Normalize
                    vector = vector / np.linalg.norm(vector)
                    self.word_vectors[word] = vector
                    self.vocab.add(word)
        
        # Add some common words
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'with', 'by', 'from', 'about', 'into', 'through', 'during',
                       'before', 'after', 'above', 'below', 'between', 'among']
        
        for word in common_words:
            if word not in self.word_vectors:
                vector = np.random.normal(0, 0.1, self.embedding_dim)
                vector = vector / np.linalg.norm(vector)
                self.word_vectors[word] = vector
                self.vocab.add(word)
        
        self.is_loaded = True
        print(f"✅ Created {len(self.word_vectors)} synthetic word vectors (dim={self.embedding_dim})")
    
    def load_glove_embeddings(self, glove_file_path: Optional[str] = None):
        """Load pre-trained GloVe embeddings from file"""
        
        if glove_file_path and os.path.exists(glove_file_path):
            print(f"📁 Loading GloVe embeddings from {glove_file_path}...")
            
            try:
                loaded_count = 0
                with open(glove_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if loaded_count >= self.max_vocab_size:
                            break
                            
                        parts = line.strip().split()
                        if len(parts) == self.embedding_dim + 1:
                            word = parts[0]
                            vector = np.array([float(x) for x in parts[1:]])
                            self.word_vectors[word] = vector
                            self.vocab.add(word)
                            loaded_count += 1
                
                self.is_loaded = True
                print(f"✅ Loaded {loaded_count} GloVe word vectors")
                return True
                
            except Exception as e:
                print(f"❌ Failed to load GloVe embeddings: {e}")
                print("🔧 Falling back to synthetic embeddings...")
                return False
        else:
            print("📁 GloVe file not found, using synthetic embeddings")
            return False
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a single word"""
        word_lower = word.lower()
        return self.word_vectors.get(word_lower)
    
    def get_sentence_embedding(self, text: str, method: str = 'mean') -> np.ndarray:
        """Get sentence embedding by aggregating word vectors"""
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        vectors = []
        
        for word in words:
            vector = self.get_word_vector(word)
            if vector is not None:
                vectors.append(vector)
        
        if not vectors:
            # Return zero vector if no words found
            return np.zeros(self.embedding_dim)
        
        vectors = np.array(vectors)
        
        if method == 'mean':
            return np.mean(vectors, axis=0)
        elif method == 'max':
            return np.max(vectors, axis=0)
        elif method == 'weighted_mean':
            # Weight by inverse frequency (simple TF-IDF like weighting)
            word_counts = Counter(words)
            weights = np.array([1.0 / word_counts[word] for word in words 
                              if self.get_word_vector(word) is not None])
            if len(weights) > 0:
                weights = weights / np.sum(weights)
                return np.average(vectors, axis=0, weights=weights)
            else:
                return np.mean(vectors, axis=0)
        else:
            return np.mean(vectors, axis=0)
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """Get cosine similarity between two words"""
        
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def find_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words to given word"""
        
        target_vector = self.get_word_vector(word)
        if target_vector is None:
            return []
        
        similarities = []
        for vocab_word, vector in self.word_vectors.items():
            if vocab_word != word.lower():
                similarity = np.dot(target_vector, vector) / (
                    np.linalg.norm(target_vector) * np.linalg.norm(vector)
                )
                similarities.append((vocab_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class EnhancedInputClassifier:
    """Enhanced input classifier using manifold learning for better task classification"""
    
    def __init__(self, enable_manifold_learning: bool = True, manifold_config: Optional[ManifoldLearningConfig] = None,
                 glove_dim: int = 100, glove_file_path: Optional[str] = None):
        self.enable_manifold_learning = enable_manifold_learning
        
        # Initialize GloVe embeddings
        self.glove_embeddings = GloVeEmbeddings(embedding_dim=glove_dim)
        if glove_file_path:
            self.glove_embeddings.load_glove_embeddings(glove_file_path)
        
        # Initialize TF-IDF for semantic analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Traditional pattern-based classification
        self.task_patterns = {
            TaskType.MATHEMATICAL: {
                'keywords': ['solve', 'calculate', 'equation', 'formula', 'algebra', 'geometry', 
                           'derivative', 'integral', 'probability', 'statistics', 'theorem', 'matrix',
                           'vector', 'calculus', 'linear', 'polynomial', 'trigonometry'],
                'patterns': [r'\d+[x-z]\s*[+\-*/=]', r'[+\-*/=]\s*\d+', r'\b\d+\.\d+\b', 
                           r'[∫∑∏√∞≠≤≥∈∉⊂⊃∩∪]', r'dx\b', r'dy\b', r'f\(x\)'],
                'complexity_indicators': ['differential', 'eigenvalue', 'proof', 'limit', 'convergence'],
                'manifold_type': 'euclidean'  # Mathematical problems often have euclidean structure
            },
            TaskType.CREATIVE_WRITING: {
                'keywords': ['story', 'write', 'creative', 'character', 'plot', 'narrative',
                           'poem', 'fiction', 'dialogue', 'scene', 'chapter', 'novel', 'prose'],
                'patterns': [r'write\s+a\s+story', r'create\s+a\s+character', r'once\s+upon',
                           r'tell\s+me\s+about', r'imagine\s+a'],
                'complexity_indicators': ['literary', 'metaphor', 'symbolism', 'genre', 'narrative'],
                'manifold_type': 'sphere'  # Creative content often has directional/thematic structure
            },
            TaskType.FACTUAL_QA: {
                'keywords': ['what', 'who', 'when', 'where', 'why', 'how', 'explain', 'define',
                           'fact', 'information', 'details', 'describe', 'tell', 'know'],
                'patterns': [r'^(what|who|when|where|why|how)\s+', r'explain\s+', r'define\s+',
                           r'what\s+is', r'who\s+was', r'how\s+does'],
                'complexity_indicators': ['comprehensive', 'detailed', 'analysis', 'research'],
                'manifold_type': 'euclidean'  # Factual content has clear dimensional structure
            },
            TaskType.REASONING: {
                'keywords': ['analyze', 'compare', 'evaluate', 'assess', 'argue', 'reason',
                           'logic', 'because', 'therefore', 'conclude', 'infer', 'judge'],
                'patterns': [r'pros\s+and\s+cons', r'advantages\s+and\s+disadvantages', 
                           r'compare\s+.*\s+with', r'analyze\s+', r'what\s+do\s+you\s+think'],
                'complexity_indicators': ['critical', 'philosophical', 'ethical', 'complex'],
                'manifold_type': 'hyperbolic'  # Reasoning often has hierarchical structure
            },
            TaskType.CODE_GENERATION: {
                'keywords': ['code', 'program', 'function', 'algorithm', 'python', 'javascript',
                           'class', 'method', 'variable', 'loop', 'debug', 'implement', 'build'],
                'patterns': [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'#.*code',
                           r'write\s+.*\s+code', r'implement\s+', r'function\s+'],
                'complexity_indicators': ['optimization', 'architecture', 'framework', 'api'],
                'manifold_type': 'euclidean'  # Code has structured, logical organization
            },
            TaskType.SCIENTIFIC: {
                'keywords': ['research', 'hypothesis', 'experiment', 'data', 'method',
                           'analysis', 'conclusion', 'theory', 'evidence', 'study', 'science'],
                'patterns': [r'research\s+', r'study\s+shows', r'according\s+to',
                           r'hypothesis\s+', r'experiment\s+'],
                'complexity_indicators': ['peer-reviewed', 'methodology', 'statistical', 'empirical'],
                'manifold_type': 'sphere'  # Scientific concepts often have global relationships
            },
            TaskType.CONVERSATIONAL: {
                'keywords': ['hello', 'hi', 'thanks', 'please', 'help', 'chat', 'talk',
                           'discuss', 'opinion', 'think', 'feel', 'like'],
                'patterns': [r'^(hi|hello|hey)', r'what\s+do\s+you\s+think', r'can\s+you\s+help',
                           r'thank\s+you', r'please\s+'],
                'complexity_indicators': ['personal', 'emotional', 'social', 'casual'],
                'manifold_type': 'torus'  # Conversations have cyclical, repetitive patterns
            }
        }
        
        # Initialize TF-IDF for semantic analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.domain_embeddings = {}
        self.is_fitted = False
        
        # Initialize manifold learner if enabled
        self.manifold_learner = None
        if self.enable_manifold_learning:
            if manifold_config is None:
                manifold_config = ManifoldLearningConfig(
                    embedding_dim=20,
                    manifold_method="auto",  # Will select best manifold automatically
                    enable_online_learning=True,
                    enable_clustering=True
                )
            self.manifold_learner = ManifoldLearner(manifold_config)
        
        # Cache for performance
        self.analysis_cache = {}
        self.cache_size_limit = 1000
        
        # Training history for adaptive learning
        self.training_history = deque(maxlen=10000)
        self.performance_feedback = {}
    
    def fit_training_data(self, training_texts: List[str], training_labels: Optional[List[str]] = None):
        """Fit the classifier on training data"""
        
        print(f"🔧 Fitting classifier on {len(training_texts)} training examples...")
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer.fit(training_texts)
        self.is_fitted = True
        
        # Train manifold learner if enabled
        if self.manifold_learner is not None:
            print("🌐 Training manifold learner...")
            self.manifold_learner.learn_manifold_offline(training_texts)
            
            # If labels are provided, use them for supervised learning
            if training_labels is not None:
                for text, label in zip(training_texts, training_labels):
                    try:
                        task_type = TaskType(label)
                        self.training_history.append({
                            'text': text,
                            'true_label': task_type,
                            'timestamp': time.time()
                        })
                    except ValueError:
                        continue  # Skip invalid labels
        
        print("✅ Classifier fitting complete!")
    
    def analyze_input(self, text: str, use_cache: bool = True) -> InputAnalysis:
        """Enhanced input analysis using both traditional and manifold-based methods"""
        
        # Check cache first
        if use_cache and text in self.analysis_cache:
            return self.analysis_cache[text]
        
        # Traditional pattern-based analysis
        traditional_analysis = self._traditional_analysis(text)
        
        # Manifold-based analysis (if enabled and fitted)
        manifold_analysis = None
        if self.manifold_learner is not None and self.is_fitted:
            manifold_analysis = self._manifold_analysis(text)
        
        # Combine analyses
        final_analysis = self._combine_analyses(text, traditional_analysis, manifold_analysis)
        
        # Cache result
        if use_cache:
            if len(self.analysis_cache) >= self.cache_size_limit:
                # Remove oldest entries
                oldest_key = next(iter(self.analysis_cache))
                del self.analysis_cache[oldest_key]
            self.analysis_cache[text] = final_analysis
        
        return final_analysis
    
    def _traditional_analysis(self, text: str) -> Dict[str, Any]:
        """Traditional pattern-based analysis"""
        
        text_lower = text.lower()
        
        # Calculate scores for each task type
        task_scores = {}
        manifold_preferences = {}
        
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
            manifold_preferences[task_type] = patterns['manifold_type']
        
        # Determine best task type
        best_task = max(task_scores.keys(), key=lambda k: task_scores[k])
        confidence = task_scores[best_task] / max(sum(task_scores.values()), 1)
        
        # Extract traditional features
        features = self._extract_traditional_features(text, task_scores)
        
        return {
            'task_type': best_task,
            'confidence': confidence,
            'task_scores': task_scores,
            'features': features,
            'manifold_preferences': manifold_preferences,
            'method': 'traditional'
        }
    
    def _manifold_analysis(self, text: str) -> Dict[str, Any]:
        """Manifold-based analysis using geometric learning"""
        
        try:
            # Get manifold recommendations
            recommendations = self.manifold_learner.get_recommendations(text)
            
            # Extract manifold-specific information
            best_manifold = recommendations.get('best_manifold', 'unknown')
            uncertainty = recommendations.get('uncertainty_estimate', 0.0)
            bayesian_confidence = recommendations.get('confidence', 0.0)
            recommended_models = recommendations.get('recommended_models', [])
            complexity_estimate = recommendations.get('complexity_estimate', 0.0)
            
            # Map manifold type to task type
            manifold_to_task = {
                'sphere': TaskType.CREATIVE_WRITING,
                'torus': TaskType.CONVERSATIONAL,
                'hyperbolic': TaskType.REASONING,
                'euclidean': TaskType.MATHEMATICAL,
                'pca': TaskType.FACTUAL_QA,
                'isomap': TaskType.SCIENTIFIC,
                'tsne': TaskType.CODE_GENERATION
            }
            
            suggested_task = manifold_to_task.get(best_manifold, TaskType.FACTUAL_QA)
            
            # Get embedding if available
            embedding = None
            embedding_variance = None
            if hasattr(self.manifold_learner, 'transform'):
                try:
                    # Simple text to feature conversion (in practice, use proper embeddings)
                    features = self._text_to_features(text)
                    embedding = self.manifold_learner.traditional_manifold.transform([features])[0]
                    
                    # Get uncertainty from Bayesian sampling if available
                    if hasattr(self.manifold_learner, 'sample_embeddings_bayesian'):
                        samples = self.manifold_learner.sample_embeddings_bayesian(text, n_samples=10)
                        embedding_variance = np.var(samples, axis=0).mean()
                    else:
                        embedding_variance = uncertainty
                        
                except Exception as e:
                    print(f"Warning: Could not get embedding: {e}")
                    embedding = None
                    embedding_variance = None
            
            return {
                'task_type': suggested_task,
                'confidence': bayesian_confidence,
                'uncertainty': uncertainty,
                'best_manifold': best_manifold,
                'recommended_models': recommended_models,
                'complexity_estimate': complexity_estimate,
                'embedding': embedding,
                'embedding_variance': embedding_variance,
                'method': 'manifold'
            }
            
        except Exception as e:
            print(f"Warning: Manifold analysis failed: {e}")
            return {
                'task_type': TaskType.FACTUAL_QA,
                'confidence': 0.0,
                'uncertainty': 1.0,
                'best_manifold': 'unknown',
                'method': 'manifold_fallback'
            }
    
    def _combine_analyses(self, text: str, traditional: Dict, manifold: Optional[Dict]) -> InputAnalysis:
        """Combine traditional and manifold analyses into final result"""
        
        if manifold is None:
            # Use only traditional analysis
            task_type = traditional['task_type']
            confidence = traditional['confidence']
            features = traditional['features']
            
            # Add basic uncertainty estimate based on confidence
            uncertainty = 1.0 - confidence
            best_manifold = traditional['manifold_preferences'].get(task_type, 'euclidean')
            
        else:
            # Combine both analyses
            traditional_confidence = traditional['confidence']
            manifold_confidence = manifold['confidence']
            
            # Weight the confidences (manifold learning often more accurate for complex tasks)
            manifold_weight = 0.7 if manifold_confidence > 0.3 else 0.3
            traditional_weight = 1.0 - manifold_weight
            
            # Choose task type based on weighted confidence
            if manifold_confidence * manifold_weight > traditional_confidence * traditional_weight:
                task_type = manifold['task_type']
                confidence = manifold_confidence * manifold_weight + traditional_confidence * traditional_weight
            else:
                task_type = traditional['task_type']
                confidence = traditional_confidence * traditional_weight + manifold_confidence * manifold_weight
            
        if manifold is not None:
            features.update({
                'manifold_confidence': manifold_confidence,
                'traditional_confidence': traditional_confidence,
                'uncertainty_estimate': manifold.get('uncertainty', 0.0),
                'embedding_variance': manifold.get('embedding_variance', 0.0),
                'best_manifold': manifold.get('best_manifold', 'unknown'),
                'recommended_models': manifold.get('recommended_models', [])
            })
            
            uncertainty = manifold.get('uncertainty', 1.0 - confidence)
            best_manifold = manifold.get('best_manifold', 'euclidean')
        
        # Extract final keywords and domain indicators
        keywords = self._extract_keywords(text)
        domain_indicators = self._identify_domains(text)
        complexity_score = self._calculate_complexity(text, features)
        
        # Create enhanced analysis
        analysis = InputAnalysis(
            task_type=task_type,
            confidence=confidence,
            features=features,
            keywords=keywords,
            complexity_score=complexity_score,
            domain_indicators=domain_indicators
        )
        
        # Add manifold-specific attributes
        if hasattr(analysis, '__dict__'):
            analysis.__dict__.update({
                'uncertainty_estimate': uncertainty,
                'best_manifold': best_manifold,
                'manifold_enabled': manifold is not None,
                'analysis_method': 'combined' if manifold else 'traditional'
            })
        
        return analysis
    
    def _extract_traditional_features(self, text: str, task_scores: Dict) -> Dict[str, Any]:
        """Extract traditional feature set"""
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'mathematical_symbols': len(re.findall(r'[+\-*/=<>≤≥∫∑]', text)),
            'code_indicators': len(re.findall(r'[{}();]', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'task_scores': {t.value: s for t, s in task_scores.items()}
        }
        
        # Add TF-IDF features if fitted
        if self.is_fitted:
            try:
                tfidf_features = self.tfidf_vectorizer.transform([text])
                features['tfidf_norm'] = np.linalg.norm(tfidf_features.toarray())
                features['tfidf_max'] = tfidf_features.max()
                features['tfidf_nnz'] = tfidf_features.nnz
            except Exception:
                pass  # Skip TF-IDF features if error
        
        # Add GloVe features
        if hasattr(self, 'glove_embeddings') and self.glove_embeddings.is_loaded:
            glove_embedding = self.glove_embeddings.get_sentence_embedding(text)
            glove_features = self._extract_glove_features(text)
            
            features.update({
                'glove_embedding_norm': np.linalg.norm(glove_embedding),
                'glove_embedding_mean': np.mean(glove_embedding),
                'glove_embedding_std': np.std(glove_embedding),
                'glove_semantic_coherence': glove_features[4] if len(glove_features) > 4 else 0.0,
                'glove_math_similarity': glove_features[6] if len(glove_features) > 6 else 0.0,
                'glove_creative_similarity': glove_features[7] if len(glove_features) > 7 else 0.0,
                'glove_code_similarity': glove_features[8] if len(glove_features) > 8 else 0.0
            })
        
        return features
    
    def _text_to_features(self, text: str) -> np.ndarray:
        """Convert text to numerical features combining TF-IDF and GloVe"""
        
        features = []
        
        # TF-IDF features
        if self.is_fitted:
            try:
                tfidf_features = self.tfidf_vectorizer.transform([text])
                tfidf_array = tfidf_features.toarray()[0]
                features.extend(tfidf_array)
            except Exception:
                # Fallback if TF-IDF fails
                features.extend([0.0] * 1000)  # Assuming max_features=1000
        else:
            features.extend([0.0] * 1000)
        
        # GloVe sentence embedding features
        glove_embedding = self.glove_embeddings.get_sentence_embedding(text, method='weighted_mean')
        features.extend(glove_embedding)
        
        # Additional GloVe-based features
        glove_features = self._extract_glove_features(text)
        features.extend(glove_features)
        
        # Traditional features
        traditional_features = self._extract_simple_features(text)
        features.extend(traditional_features)
        
        return np.array(features)
    
    def _extract_glove_features(self, text: str) -> List[float]:
        """Extract GloVe-based semantic features"""
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return [0.0] * 15  # Return zero features if no words
        
        # Get word vectors
        word_vectors = []
        for word in words:
            vector = self.glove_embeddings.get_word_vector(word)
            if vector is not None:
                word_vectors.append(vector)
        
        if not word_vectors:
            return [0.0] * 15
        
        word_vectors = np.array(word_vectors)
        
        features = []
        
        # Statistical features of word embeddings
        features.append(np.mean(word_vectors))  # Mean activation
        features.append(np.std(word_vectors))   # Standard deviation
        features.append(np.max(word_vectors))   # Max activation
        features.append(np.min(word_vectors))   # Min activation
        
        # Semantic coherence (average pairwise similarity)
        if len(word_vectors) > 1:
            similarities = []
            for i in range(len(word_vectors)):
                for j in range(i+1, len(word_vectors)):
                    sim = np.dot(word_vectors[i], word_vectors[j]) / (
                        np.linalg.norm(word_vectors[i]) * np.linalg.norm(word_vectors[j])
                    )
                    similarities.append(sim)
            features.append(np.mean(similarities))  # Semantic coherence
            features.append(np.std(similarities))   # Coherence variance
        else:
            features.extend([0.0, 0.0])
        
        # Task-specific similarity scores
        task_keywords = {
            'math': ['solve', 'equation', 'calculate', 'formula'],
            'creative': ['story', 'character', 'write', 'creative'],
            'code': ['algorithm', 'function', 'programming', 'implement'],
            'reasoning': ['analyze', 'compare', 'evaluate', 'assess'],
            'scientific': ['research', 'hypothesis', 'experiment', 'theory'],
            'factual': ['explain', 'define', 'information', 'fact'],
            'conversational': ['hello', 'help', 'chat', 'thanks']
        }
        
        for task_name, keywords in task_keywords.items():
            task_similarities = []
            for word in words:
                word_vector = self.glove_embeddings.get_word_vector(word)
                if word_vector is not None:
                    for keyword in keywords:
                        keyword_vector = self.glove_embeddings.get_word_vector(keyword)
                        if keyword_vector is not None:
                            sim = np.dot(word_vector, keyword_vector) / (
                                np.linalg.norm(word_vector) * np.linalg.norm(keyword_vector)
                            )
                            task_similarities.append(sim)
            
            if task_similarities:
                features.append(np.max(task_similarities))  # Best similarity to task
            else:
                features.append(0.0)
        
        # Ensure we return exactly 15 features
        while len(features) < 15:
            features.append(0.0)
        
        return features[:15]
    
    def _extract_simple_features(self, text: str) -> List[float]:
        """Extract simple numerical features"""
        
        features = [
            len(text),                                          # Text length
            len(text.split()),                                  # Word count
            len(set(text.lower().split())) / max(len(text.split()), 1),  # Vocabulary diversity
            text.count('?'),                                    # Question marks
            text.count('!'),                                    # Exclamation marks
            len(re.findall(r'[+\-*/=<>≤≥∫∑]', text)),          # Math symbols
            len(re.findall(r'[{}();]', text)),                  # Code indicators
            sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
            sum(1 for c in text if c.isdigit()) / max(len(text), 1),  # Digit ratio
            len(re.findall(r'\b[A-Z][a-z]+\b', text))          # Proper nouns
        ]
        
        return features
    
    def _calculate_complexity(self, text: str, features: Dict) -> float:
        """Calculate complexity score based on various factors"""
        
        complexity = 0.0
        
        # Length-based complexity
        complexity += min(features['length'] / 1000, 1.0) * 0.2
        
        # Technical term density
        technical_terms = ['algorithm', 'optimization', 'analysis', 'methodology', 
                         'implementation', 'architecture', 'framework', 'derivative',
                         'integral', 'matrix', 'vector', 'hypothesis', 'theorem']
        tech_density = sum(1 for term in technical_terms if term in text.lower())
        complexity += min(tech_density / 5, 1.0) * 0.3
        
        # Sentence structure complexity
        word_count = features.get('word_count', 0)
        sentence_count = max(features.get('sentence_count', 1), 1)
        avg_sentence_length = word_count / sentence_count
        complexity += min(avg_sentence_length / 20, 1.0) * 0.2
        
        # Symbol complexity
        math_symbols = features.get('mathematical_symbols', 0)
        code_indicators = features.get('code_indicators', 0)
        complexity += min((math_symbols + code_indicators) / 10, 1.0) * 0.3
        
        return min(complexity, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = Counter(words)
        
        # Filter out common words
        stopwords = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been',
                    'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there',
                    'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first'}
        
        keywords = [word for word, freq in word_freq.most_common(15) 
                   if word not in stopwords and len(word) > 3]
        
        return keywords[:8]
    
    def _identify_domains(self, text: str) -> List[str]:
        """Identify domain-specific indicators"""
        
        domain_keywords = {
            'mathematics': ['equation', 'formula', 'theorem', 'proof', 'calculation', 'algebra',
                          'geometry', 'calculus', 'derivative', 'integral', 'matrix', 'vector'],
            'science': ['hypothesis', 'experiment', 'theory', 'research', 'data', 'analysis',
                       'method', 'study', 'evidence', 'observation', 'measurement'],
            'technology': ['algorithm', 'code', 'software', 'system', 'programming', 'computer',
                         'function', 'implementation', 'framework', 'api', 'database'],
            'business': ['market', 'strategy', 'analysis', 'revenue', 'customer', 'sales',
                        'profit', 'investment', 'management', 'organization'],
            'education': ['learn', 'teach', 'student', 'curriculum', 'knowledge', 'skill',
                         'training', 'course', 'lesson', 'instruction'],
            'healthcare': ['patient', 'treatment', 'diagnosis', 'medical', 'health', 'disease',
                          'therapy', 'clinical', 'medicine', 'symptoms'],
            'creative': ['story', 'character', 'plot', 'narrative', 'poem', 'creative',
                        'fiction', 'art', 'design', 'imagination']
        }
        
        identified_domains = []
        text_lower = text.lower()
        
        for domain, keywords in domain_keywords.items():
            domain_score = sum(1 for keyword in keywords if keyword in text_lower)
            if domain_score >= 2:  # Require at least 2 matches
                identified_domains.append(domain)
        
        return identified_domains
    
    def update_performance(self, text: str, predicted_task: TaskType, actual_task: TaskType, 
                          performance_score: float):
        """Update classifier performance based on feedback"""
        
        feedback = {
            'text': text,
            'predicted': predicted_task,
            'actual': actual_task,
            'score': performance_score,
            'timestamp': time.time(),
            'correct': predicted_task == actual_task
        }
        
        self.training_history.append(feedback)
        
        # Update manifold learner if available
        if self.manifold_learner is not None:
            self.manifold_learner.update_online(
                text=text,
                task_type=actual_task,
                selected_model=predicted_task.value,
                performance_score=performance_score
            )
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get classification performance statistics"""
        
        if not self.training_history:
            return {'message': 'No training history available'}
        
        recent_history = list(self.training_history)[-1000:]  # Last 1000 examples
        
        total_predictions = len(recent_history)
        correct_predictions = sum(1 for h in recent_history if h['correct'])
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Task-specific accuracy
        task_stats = {}
        for task_type in TaskType:
            task_predictions = [h for h in recent_history if h['actual'] == task_type]
            if task_predictions:
                task_correct = sum(1 for h in task_predictions if h['correct'])
                task_accuracy = task_correct / len(task_predictions)
                task_stats[task_type.value] = {
                    'accuracy': task_accuracy,
                    'total_samples': len(task_predictions),
                    'correct_samples': task_correct
                }
        
        # Average performance score
        avg_performance = np.mean([h['score'] for h in recent_history])
        
        stats = {
            'overall_accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'average_performance_score': avg_performance,
            'task_specific_stats': task_stats,
            'manifold_enabled': self.manifold_learner is not None,
            'cache_size': len(self.analysis_cache)
        }
        
        # Add manifold learner statistics if available
        if self.manifold_learner is not None:
            try:
                manifold_diagnostics = self.manifold_learner.get_manifold_diagnostics()
                stats['manifold_diagnostics'] = manifold_diagnostics
            except Exception:
                pass
        
        return stats
    
    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
    
    def get_recommended_manifold(self, text: str) -> str:
        """Get recommended manifold type for given text"""
        
        if self.manifold_learner is None:
            # Fallback to traditional pattern matching
            analysis = self._traditional_analysis(text)
            return analysis['manifold_preferences'].get(analysis['task_type'], 'euclidean')
        
        recommendations = self.manifold_learner.get_recommendations(text)
        return recommendations.get('best_manifold', 'euclidean')
    
    def export_training_data(self) -> List[Dict[str, Any]]:
        """Export training history for analysis"""
        return list(self.training_history)


# Backward compatibility
class InputClassifier(EnhancedInputClassifier):
    """Backward compatible wrapper for the enhanced classifier"""
    
    def __init__(self):
        super().__init__(enable_manifold_learning=False)