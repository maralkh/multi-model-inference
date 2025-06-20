# File: core/glove_embeddings.py
"""GloVe word embeddings implementation with synthetic fallback"""

import numpy as np
import re
import os
import urllib.request
import zipfile
from typing import List, Dict, Optional, Tuple
from collections import Counter

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
    
    def extract_semantic_features(self, text: str) -> List[float]:
        """Extract semantic features from text using GloVe embeddings"""
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return [0.0] * 15  # Return zero features if no words
        
        # Get word vectors
        word_vectors = []
        for word in words:
            vector = self.get_word_vector(word)
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
                word_vector = self.get_word_vector(word)
                if word_vector is not None:
                    for keyword in keywords:
                        keyword_vector = self.get_word_vector(keyword)
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
    
    def get_embedding_statistics(self) -> Dict[str, any]:
        """Get statistics about loaded embeddings"""
        
        if not self.is_loaded:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'vocab_size': len(self.vocab),
            'embedding_dim': self.embedding_dim,
            'sample_words': list(self.vocab)[:10],
            'is_synthetic': not hasattr(self, '_real_glove_loaded')
        }