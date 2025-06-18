# File: core/manifold_learner.py
"""Advanced manifold learning for input distribution analysis"""

import time
import pickle
import threading
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import re

from .types import ManifoldLearningConfig, DataPoint, TaskType

class ManifoldLearner:
    """Advanced manifold learning for input distribution analysis"""
    
    def __init__(self, config: ManifoldLearningConfig):
        self.config = config
        self.embeddings_history = deque(maxlen=config.memory_size)
        self.data_points = []
        
        # Feature extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.feature_scaler = StandardScaler()
        
        # Manifold learning models
        self.manifold_model = None
        self.incremental_pca = IncrementalPCA(n_components=config.embedding_dim)
        self.clustering_model = MiniBatchKMeans(n_clusters=8, random_state=42)
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        
        # Online learning components
        self.online_buffer = deque(maxlen=config.online_batch_size)
        self.update_counter = 0
        self.manifold_fitted = False
        self.cluster_centers = {}
        self.cluster_performance = {}
        
        # Threading for async updates
        self.update_lock = threading.Lock()
        
    def extract_features(self, text: str) -> np.ndarray:
        """Extract comprehensive features from text"""
        
        # TF-IDF features
        try:
            if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                tfidf_features = self.tfidf_vectorizer.transform([text]).toarray()[0]
            else:
                # First time fitting
                tfidf_features = self.tfidf_vectorizer.fit_transform([text]).toarray()[0]
        except:
            # Fallback to basic features
            tfidf_features = np.zeros(1000)
        
        # Linguistic features
        linguistic_features = self._extract_linguistic_features(text)
        
        # Task-specific features
        task_features = self._extract_task_features(text)
        
        # Combine all features
        combined_features = np.concatenate([
            tfidf_features,
            linguistic_features,
            task_features
        ])
        
        return combined_features
    
    def _extract_linguistic_features(self, text: str) -> np.ndarray:
        """Extract linguistic and structural features"""
        
        features = []
        
        # Basic statistics
        features.extend([
            len(text),                          # Text length
            len(text.split()),                  # Word count
            len(re.findall(r'[.!?]+', text)),  # Sentence count
            text.count('?'),                    # Question marks
            text.count('!'),                    # Exclamation marks
            len(re.findall(r'[A-Z]', text)),   # Capital letters
        ])
        
        # Complexity indicators
        avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features.append(avg_word_length)
        
        # Readability approximation
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum(len(re.findall(r'[aeiouAEIOU]', word)) for word in text.split())
        
        if sentences > 0 and words > 0:
            flesch_score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
            features.append(flesch_score / 100)  # Normalize
        else:
            features.append(0.5)  # Neutral score
        
        # Character-level features
        features.extend([
            text.count('(') + text.count(')'),         # Parentheses
            text.count('[') + text.count(']'),         # Brackets
            text.count('{') + text.count('}'),         # Braces
            len(re.findall(r'[0-9]', text)) / max(len(text), 1),  # Digit density
            len(re.findall(r'[+\-*/=<>]', text)) / max(len(text), 1),  # Math symbols
        ])
        
        return np.array(features)
    
    def _extract_task_features(self, text: str) -> np.ndarray:
        """Extract task-specific features"""
        
        text_lower = text.lower()
        features = []
        
        # Mathematical indicators
        math_patterns = [
            r'\d+[x-z]\s*[+\-*/=]',  # Algebraic expressions
            r'[∫∑∏√∞≠≤≥∈∉⊂⊃∩∪]',      # Math symbols
            r'\b(solve|calculate|equation|formula)\b',  # Math keywords
            r'\b(derivative|integral|limit|matrix)\b'   # Advanced math
        ]
        math_score = sum(len(re.findall(pattern, text_lower)) for pattern in math_patterns)
        features.append(math_score / max(len(text), 1))
        
        # Code indicators
        code_patterns = [
            r'def\s+\w+',           # Function definitions
            r'class\s+\w+',         # Class definitions
            r'import\s+\w+',        # Imports
            r'[{}();]',             # Code punctuation
            r'\b(function|method|algorithm|code)\b'  # Code keywords
        ]
        code_score = sum(len(re.findall(pattern, text_lower)) for pattern in code_patterns)
        features.append(code_score / max(len(text), 1))
        
        # Creative writing indicators
        creative_patterns = [
            r'\b(story|character|plot|narrative)\b',
            r'\b(once upon|long ago|in a|there was)\b',
            r'\b(write|create|imagine|describe)\b'
        ]
        creative_score = sum(len(re.findall(pattern, text_lower)) for pattern in creative_patterns)
        features.append(creative_score / max(len(text), 1))
        
        # Reasoning indicators
        reasoning_patterns = [
            r'\b(analyze|compare|evaluate|assess)\b',
            r'\b(because|therefore|however|although)\b',
            r'\b(pros and cons|advantages|disadvantages)\b'
        ]
        reasoning_score = sum(len(re.findall(pattern, text_lower)) for pattern in reasoning_patterns)
        features.append(reasoning_score / max(len(text), 1))
        
        return np.array(features)
    
    def learn_manifold_offline(self, texts: List[str], labels: List[str] = None) -> None:
        """Learn manifold structure from offline data"""
        
        print(f"🧠 Learning manifold from {len(texts)} offline samples...")
        
        # Extract features for all texts
        features_list = []
        for text in texts:
            features = self.extract_features(text)
            features_list.append(features)
        
        features_matrix = np.array(features_list)
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features_matrix)
        
        # Learn manifold structure
        if self.config.manifold_method == "umap":
            try:
                import umap
                self.manifold_model = umap.UMAP(
                    n_components=self.config.embedding_dim,
                    random_state=42,
                    n_neighbors=15,
                    min_dist=0.1
                )
            except ImportError:
                print("⚠️ UMAP not available, falling back to PCA")
                self.manifold_model = PCA(n_components=self.config.embedding_dim, random_state=42)
        elif self.config.manifold_method == "tsne":
            self.manifold_model = TSNE(
                n_components=self.config.embedding_dim,
                random_state=42,
                perplexity=min(30, len(texts) // 4)
            )
        else:  # PCA
            self.manifold_model = PCA(n_components=self.config.embedding_dim, random_state=42)
        
        # Fit manifold model
        embeddings = self.manifold_model.fit_transform(scaled_features)
        
        # Perform clustering if enabled
        if self.config.enable_clustering:
            cluster_labels = self.clustering_model.fit_predict(embeddings)
            
            # Store cluster information
            for i, (text, embedding, cluster_id) in enumerate(zip(texts, embeddings, cluster_labels)):
                data_point = DataPoint(
                    text=text,
                    embedding=embedding,
                    task_type=TaskType.CONVERSATIONAL,  # Default, will be updated
                    selected_model="unknown",
                    performance_score=0.0,
                    timestamp=time.time(),
                    complexity=0.0,
                    cluster_id=cluster_id
                )
                self.data_points.append(data_point)
                
                # Update cluster centers
                if cluster_id not in self.cluster_centers:
                    self.cluster_centers[cluster_id] = []
                self.cluster_centers[cluster_id].append(embedding)
        
        # Fit nearest neighbors model
        self.nn_model.fit(embeddings)
        self.manifold_fitted = True
        
        print(f"✅ Manifold learning completed. Found {len(set(cluster_labels))} clusters.")
    
    def update_online(self, text: str, task_type: TaskType, selected_model: str, 
                     performance_score: float) -> None:
        """Update manifold learning with new online data"""
        
        if not self.config.enable_online_learning:
            return
        
        with self.update_lock:
            # Extract features
            features = self.extract_features(text)
            
            # Get embedding
            if self.manifold_fitted:
                scaled_features = self.feature_scaler.transform([features])
                if hasattr(self.manifold_model, 'transform'):
                    embedding = self.manifold_model.transform(scaled_features)[0]
                else:
                    # For methods like t-SNE that don't support transform
                    embedding = self._approximate_embedding(features)
            else:
                # Use incremental PCA for initial embeddings
                scaled_features = self.feature_scaler.fit_transform([features])
                embedding = self.incremental_pca.fit_transform(scaled_features)[0]
            
            # Create data point
            data_point = DataPoint(
                text=text,
                embedding=embedding,
                task_type=task_type,
                selected_model=selected_model,
                performance_score=performance_score,
                timestamp=time.time(),
                complexity=self._estimate_complexity(text),
                cluster_id=-1
            )
            
            # Add to buffer and history
            self.online_buffer.append(data_point)
            self.embeddings_history.append(embedding)
            self.data_points.append(data_point)
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate text complexity"""
        
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        if sentence_count == 0:
            return 0.5
        
        avg_sentence_length = word_count / sentence_count
        
        # Technical term indicators
        technical_indicators = [
            'algorithm', 'optimization', 'methodology', 'implementation',
            'architecture', 'framework', 'derivative', 'integral'
        ]
        
        tech_density = sum(1 for term in technical_indicators if term in text.lower())
        
        # Combine factors
        complexity = min(
            (avg_sentence_length / 20) * 0.4 +
            (tech_density / 5) * 0.4 +
            (len(text) / 1000) * 0.2,
            1.0
        )
        
        return complexity
    
    def _approximate_embedding(self, features: np.ndarray) -> np.ndarray:
        """Approximate embedding for new data when transform is not available"""
        
        if not self.data_points:
            return np.random.normal(0, 0.1, self.config.embedding_dim)
        
        # Find most similar existing data point
        scaled_features = self.feature_scaler.transform([features])
        
        best_similarity = -1
        best_embedding = None
        
        for data_point in self.data_points[-100:]:  # Check last 100 points for efficiency
            existing_features = self.extract_features(data_point.text)
            existing_scaled = self.feature_scaler.transform([existing_features])
            
            similarity = cosine_similarity(scaled_features, existing_scaled)[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_embedding = data_point.embedding
        
        if best_embedding is not None:
            # Add some noise to avoid exact duplicates
            noise = np.random.normal(0, 0.05, len(best_embedding))
            return best_embedding + noise
        
        return np.random.normal(0, 0.1, self.config.embedding_dim)
    
    def get_recommendations(self, text: str) -> Dict[str, Any]:
        """Get model recommendations based on manifold analysis"""
        
        if not self.manifold_fitted:
            return {'recommended_models': ['general_model'], 'confidence': 0.5}
        
        # Extract features and get embedding
        features = self.extract_features(text)
        scaled_features = self.feature_scaler.transform([features])
        
        if hasattr(self.manifold_model, 'transform'):
            embedding = self.manifold_model.transform(scaled_features)[0]
        else:
            embedding = self._approximate_embedding(features)
        
        # Find similar data points
        if len(self.embeddings_history) > 0:
            embeddings_matrix = np.array(list(self.embeddings_history))
            distances, indices = self.nn_model.kneighbors([embedding])
            
            # Analyze similar points
            similar_points = [self.data_points[i] for i in indices[0] 
                            if i < len(self.data_points)]
            
            # Model recommendations based on similar points
            model_scores = {}
            for point in similar_points:
                model = point.selected_model
                score = point.performance_score
                
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(score)
            
            # Calculate average scores
            model_recommendations = {}
            for model, scores in model_scores.items():
                model_recommendations[model] = np.mean(scores)
            
            # Sort by score
            sorted_models = sorted(model_recommendations.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            recommended_models = [model for model, score in sorted_models[:3]]
            confidence = np.mean([score for model, score in sorted_models[:1]]) if sorted_models else 0.5
            
            return {
                'recommended_models': recommended_models or ['general_model'],
                'confidence': confidence,
                'similar_points_count': len(similar_points),
                'embedding': embedding.tolist(),
                'complexity_estimate': self._estimate_complexity(text)
            }
        
        return {'recommended_models': ['general_model'], 'confidence': 0.5}
    
    def save_model(self, filepath: str) -> None:
        """Save the manifold learning model"""
        
        model_data = {
            'config': self.config,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_scaler': self.feature_scaler,
            'manifold_model': self.manifold_model,
            'data_points': self.data_points[-1000:],  # Save last 1000 points
            'manifold_fitted': self.manifold_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Manifold model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved manifold learning model"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.feature_scaler = model_data['feature_scaler']
        self.manifold_model = model_data['manifold_model']
        self.data_points = model_data['data_points']
        self.manifold_fitted = model_data['manifold_fitted']
        
        # Rebuild embeddings history
        self.embeddings_history = deque(
            [dp.embedding for dp in self.data_points], 
            maxlen=self.config.memory_size
        )
        
        # Refit nearest neighbors if we have data
        if self.data_points:
            embeddings = [dp.embedding for dp in self.data_points]
            self.nn_model.fit(embeddings)
        
        print(f"📂 Manifold model loaded from {filepath}")