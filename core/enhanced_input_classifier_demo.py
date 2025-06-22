# File: examples/complete_enhanced_classifier_demo.py
"""Complete demonstration of enhanced input classifier with TF-IDF + GloVe + Manifold learning"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import time
from sklearn.metrics.pairwise import cosine_similarity

from .enhanced_input_classifier import EnhancedInputClassifier
from .types import TaskType, ManifoldLearningConfig
from .glove_embeddings import GloVeEmbeddings

def demonstrate_enhanced_classification():
    """Demonstrate enhanced classification capabilities"""
    
    print("🧠 Enhanced Input Classifier with TF-IDF + GloVe + Manifold Learning")
    print("=" * 70)
    
    # Create enhanced classifier with GloVe embeddings
    manifold_config = ManifoldLearningConfig(
        embedding_dim=25,
        manifold_method="auto",
        enable_online_learning=True,
        enable_clustering=True
    )
    
    classifier = EnhancedInputClassifier(
        enable_manifold_learning=True,
        manifold_config=manifold_config,
        glove_dim=100,  # Use 100-dimensional GloVe embeddings
        glove_file_path=None  # Will use synthetic embeddings
    )
    
    # Training data with diverse examples
    training_data = [
        # Mathematical tasks
        "Solve the quadratic equation x² + 5x + 6 = 0",
        "Calculate the derivative of sin(x) * cos(x)",
        "Find the eigenvalues of the 2x2 matrix [[2,1],[1,2]]",
        "Prove that the sum of angles in a triangle is 180°",
        "Integrate the function f(x) = x³ from 0 to 2",
        
        # Creative writing tasks
        "Write a short story about a magical forest",
        "Create a character description for a space explorer",
        "Compose a poem about the changing seasons",
        "Develop a plot for a mystery novel",
        "Write dialogue between two old friends meeting again",
        
        # Code generation tasks
        "Implement a binary search algorithm in Python",
        "Create a function to sort a list of dictionaries",
        "Write a class for managing a shopping cart",
        "Debug this recursive function for calculating factorial",
        "Design a REST API for user authentication",
        
        # Reasoning tasks
        "Analyze the pros and cons of renewable energy",
        "Compare democracy and autocracy as governance systems",
        "Evaluate the ethical implications of AI in healthcare",
        "What are the causes and effects of climate change?",
        "Assess the impact of social media on mental health",
        
        # Scientific tasks
        "Explain the process of photosynthesis in plants",
        "Describe the structure of DNA and its function",
        "What is the theory of relativity and its implications?",
        "How do vaccines work to prevent diseases?",
        "Analyze the lifecycle of stars in the universe",
        
        # Factual Q&A
        "What is the capital of Australia?",
        "Who invented the telephone?",
        "When did World War II end?",
        "Where is the Great Wall of China located?",
        "How many continents are there?",
        
        # Conversational
        "Hi there! How are you doing today?",
        "Thanks for your help with that problem",
        "What do you think about this weather?",
        "Can you help me with something?",
        "Have a great day!"
    ]
    
    # Corresponding labels for training
    training_labels = [
        # Mathematical (5 examples)
        'mathematical', 'mathematical', 'mathematical', 'mathematical', 'mathematical',
        # Creative writing (5 examples)
        'creative_writing', 'creative_writing', 'creative_writing', 'creative_writing', 'creative_writing',
        # Code generation (5 examples)
        'code_generation', 'code_generation', 'code_generation', 'code_generation', 'code_generation',
        # Reasoning (5 examples)
        'reasoning', 'reasoning', 'reasoning', 'reasoning', 'reasoning',
        # Scientific (5 examples)
        'scientific', 'scientific', 'scientific', 'scientific', 'scientific',
        # Factual Q&A (5 examples)
        'factual_qa', 'factual_qa', 'factual_qa', 'factual_qa', 'factual_qa',
        # Conversational (5 examples)
        'conversational', 'conversational', 'conversational', 'conversational', 'conversational'
    ]
    
    print(f"🔧 Training classifier with {len(training_data)} examples...")
    classifier.fit_training_data(training_data, training_labels)
    
    # Test cases for demonstration
    test_cases = [
        # Clear mathematical
        "Solve the differential equation dy/dx = 2x + 3",
        
        # Clear creative
        "Write a science fiction story about time travel",
        
        # Clear code
        "Implement a graph traversal algorithm using DFS",
        
        # Clear reasoning
        "Analyze the advantages and disadvantages of remote work",
        
        # Clear scientific
        "Explain quantum entanglement and its applications",
        
        # Clear factual
        "What are the main causes of the American Civil War?",
        
        # Clear conversational
        "Hello! I hope you're having a wonderful day!",
        
        # Ambiguous cases
        "How do neural networks learn patterns in data?",  # Could be scientific or code
        "Create an algorithm to optimize investment portfolios",  # Could be math or code
        "What makes a compelling narrative structure?",  # Could be creative or reasoning
        "Calculate the ROI of a marketing campaign",  # Could be math or business
        "Debug and explain this machine learning model"  # Could be code or scientific
    ]
    
    print(f"\n🧪 Testing enhanced classifier with {len(test_cases)} cases...")
    print("=" * 70)
    
    results = []
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n[{i}] Testing: {test_text[:50]}...")
        
        # Analyze with enhanced classifier
        start_time = time.time()
        analysis = classifier.analyze_input(test_text)
        analysis_time = time.time() - start_time
        
        # Extract results
        task_type = analysis.task_type.value
        confidence = analysis.confidence
        complexity = analysis.complexity_score
        keywords = analysis.keywords[:3]  # Top 3 keywords
        domains = analysis.domain_indicators
        
        # Extract manifold-specific and GloVe information
        uncertainty = getattr(analysis, 'uncertainty_estimate', 0.0)
        best_manifold = getattr(analysis, 'best_manifold', 'unknown')
        manifold_confidence = analysis.features.get('manifold_confidence', 0.0)
        traditional_confidence = analysis.features.get('traditional_confidence', 0.0)
        
        # GloVe-specific features
        glove_coherence = analysis.features.get('glove_semantic_coherence', 0.0)
        glove_embedding_norm = analysis.features.get('glove_embedding_norm', 0.0)
        glove_math_sim = analysis.features.get('glove_math_similarity', 0.0)
        glove_creative_sim = analysis.features.get('glove_creative_similarity', 0.0)
        
        print(f"  📊 Task Type: {task_type}")
        print(f"  🎯 Confidence: {confidence:.3f}")
        print(f"  🔮 Uncertainty: {uncertainty:.3f}")
        print(f"  🌐 Best Manifold: {best_manifold}")
        print(f"  📈 Complexity: {complexity:.3f}")
        print(f"  🔑 Keywords: {', '.join(keywords)}")
        print(f"  🏷️ Domains: {', '.join(domains) if domains else 'None'}")
        print(f"  ⚡ Analysis Time: {analysis_time:.4f}s")
        
        # GloVe semantic information
        if glove_embedding_norm > 0:
            print(f"  🎭 GloVe Features:")
            print(f"    Semantic Coherence: {glove_coherence:.3f}")
            print(f"    Embedding Strength: {glove_embedding_norm:.3f}")
            print(f"    Math Similarity: {glove_math_sim:.3f}")
            print(f"    Creative Similarity: {glove_creative_sim:.3f}")
        
        if manifold_confidence > 0:
            print(f"  🔄 Method Comparison:")
            print(f"    Traditional: {traditional_confidence:.3f}")
            print(f"    Manifold: {manifold_confidence:.3f}")
        
        results.append({
            'text': test_text,
            'task_type': task_type,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'best_manifold': best_manifold,
            'complexity': complexity,
            'keywords': keywords,
            'domains': domains,
            'analysis_time': analysis_time,
            'manifold_confidence': manifold_confidence,
            'traditional_confidence': traditional_confidence,
            'glove_coherence': glove_coherence,
            'glove_embedding_norm': glove_embedding_norm,
            'glove_math_sim': glove_math_sim,
            'glove_creative_sim': glove_creative_sim
        })
    
    return results, classifier

def analyze_classifier_performance(results: List[Dict], classifier: EnhancedInputClassifier):
    """Analyze the performance of the enhanced classifier"""
    
    print(f"\n📊 Enhanced Classifier Performance Analysis")
    print("=" * 50)
    
    # Task distribution
    task_distribution = {}
    for result in results:
        task = result['task_type']
        task_distribution[task] = task_distribution.get(task, 0) + 1
    
    print(f"📈 Task Distribution:")
    for task, count in task_distribution.items():
        percentage = (count / len(results)) * 100
        print(f"  {task}: {count} ({percentage:.1f}%)")
    
    # Confidence and uncertainty statistics
    confidences = [r['confidence'] for r in results]
    uncertainties = [r['uncertainty'] for r in results]
    complexities = [r['complexity'] for r in results]
    
    print(f"\n📊 Classification Statistics:")
    print(f"  Average Confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
    print(f"  Average Uncertainty: {np.mean(uncertainties):.3f} ± {np.std(uncertainties):.3f}")
    print(f"  Average Complexity: {np.mean(complexities):.3f} ± {np.std(complexities):.3f}")
    print(f"  Analysis Time Range: {min(r['analysis_time'] for r in results):.4f}s - {max(r['analysis_time'] for r in results):.4f}s")
    
    # Manifold usage patterns
    manifold_usage = {}
    for result in results:
        manifold = result['best_manifold']
        manifold_usage[manifold] = manifold_usage.get(manifold, 0) + 1
    
    print(f"\n🌐 Manifold Usage Distribution:")
    for manifold, count in manifold_usage.items():
        percentage = (count / len(results)) * 100
        print(f"  {manifold}: {count} ({percentage:.1f}%)")
    
    # GloVe semantic analysis
    glove_coherences = [r['glove_coherence'] for r in results]
    glove_norms = [r['glove_embedding_norm'] for r in results]
    
    if any(gc > 0 for gc in glove_coherences):
        print(f"\n🎭 GloVe Semantic Analysis:")
        print(f"  Average Semantic Coherence: {np.mean(glove_coherences):.3f} ± {np.std(glove_coherences):.3f}")
        print(f"  Average Embedding Strength: {np.mean(glove_norms):.3f} ± {np.std(glove_norms):.3f}")
        
        # Correlation between semantic features and confidence
        if len(glove_coherences) > 1:
            coherence_confidence_corr = np.corrcoef(glove_coherences, confidences)[0, 1]
            print(f"  Coherence-Confidence Correlation: {coherence_confidence_corr:.3f}")
    
    # Method comparison analysis
    method_comparisons = [(r['traditional_confidence'], r['manifold_confidence']) 
                         for r in results if r['manifold_confidence'] > 0]
    
    if method_comparisons:
        trad_scores = [mc[0] for mc in method_comparisons]
        manifold_scores = [mc[1] for mc in method_comparisons]
        
        print(f"\n⚖️ Method Comparison Analysis:")
        print(f"  Traditional Average: {np.mean(trad_scores):.3f}")
        print(f"  Manifold Average: {np.mean(manifold_scores):.3f}")
        
        # Count where each method was better
        manifold_better = sum(1 for t, m in method_comparisons if m > t)
        traditional_better = sum(1 for t, m in method_comparisons if t > m)
        tied = len(method_comparisons) - manifold_better - traditional_better
        
        print(f"  Manifold Better: {manifold_better}/{len(method_comparisons)} ({manifold_better/len(method_comparisons)*100:.1f}%)")
        print(f"  Traditional Better: {traditional_better}/{len(method_comparisons)} ({traditional_better/len(method_comparisons)*100:.1f}%)")
        print(f"  Tied: {tied}/{len(method_comparisons)} ({tied/len(method_comparisons)*100:.1f}%)")
    
    # Get classifier internal statistics
    classifier_stats = classifier.get_classification_statistics()
    print(f"\n🔍 Classifier Internal Statistics:")
    for key, value in classifier_stats.items():
        if key not in ['task_specific_stats', 'manifold_diagnostics']:
            print(f"  {key}: {value}")

def create_comprehensive_visualization(results: List[Dict]):
    """Create comprehensive visualizations of classifier results"""
    
    print(f"\n📈 Creating Comprehensive Performance Visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid of subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confidence vs Uncertainty colored by GloVe coherence
    ax1 = fig.add_subplot(gs[0, 0])
    confidences = [r['confidence'] for r in results]
    uncertainties = [r['uncertainty'] for r in results]
    glove_coherences = [r['glove_coherence'] for r in results]
    
    scatter = ax1.scatter(confidences, uncertainties, c=glove_coherences, 
                         cmap='viridis', alpha=0.7, s=60)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Uncertainty')
    ax1.set_title('Confidence vs Uncertainty\n(colored by GloVe Coherence)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Semantic Coherence')
    
    # 2. Task type distribution
    ax2 = fig.add_subplot(gs[0, 1])
    task_counts = {}
    for result in results:
        task = result['task_type'].replace('_', ' ').title()
        task_counts[task] = task_counts.get(task, 0) + 1
    
    tasks = list(task_counts.keys())
    counts = list(task_counts.values())
    
    bars = ax2.bar(tasks, counts, color='skyblue', alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Task Type Distribution')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # 3. Manifold distribution pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    manifold_counts = {}
    for result in results:
        manifold = result['best_manifold']
        manifold_counts[manifold] = manifold_counts.get(manifold, 0) + 1
    
    manifolds = list(manifold_counts.keys())
    m_counts = list(manifold_counts.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(manifolds)))
    ax3.pie(m_counts, labels=manifolds, autopct='%1.1f%%', colors=colors)
    ax3.set_title('Best Manifold Distribution')
    
    # 4. Semantic coherence vs confidence
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(glove_coherences, confidences, alpha=0.7, color='purple')
    
    # Add trend line
    if len(glove_coherences) > 1 and np.std(glove_coherences) > 0:
        z = np.polyfit(glove_coherences, confidences, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(glove_coherences), max(glove_coherences), 100)
        ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    ax4.set_xlabel('GloVe Semantic Coherence')
    ax4.set_ylabel('Classification Confidence')
    ax4.set_title('Semantic Coherence vs\nClassification Confidence')
    ax4.grid(True, alpha=0.3)
    
    # 5. Task-specific GloVe similarities
    ax5 = fig.add_subplot(gs[1, 1])
    task_glove_data = {}
    for result in results:
        task = result['task_type'].replace('_', ' ').title()
        if task not in task_glove_data:
            task_glove_data[task] = []
        task_glove_data[task].append({
            'math_sim': result['glove_math_sim'],
            'creative_sim': result['glove_creative_sim'],
            'coherence': result['glove_coherence']
        })
    
    # Create grouped bar chart for task similarities
    task_names = list(task_glove_data.keys())
    math_sims = [np.mean([d['math_sim'] for d in task_glove_data[task]]) for task in task_names]
    creative_sims = [np.mean([d['creative_sim'] for d in task_glove_data[task]]) for task in task_names]
    
    x_pos = np.arange(len(task_names))
    width = 0.35
    
    ax5.bar(x_pos - width/2, math_sims, width, label='Math Similarity', alpha=0.7, color='orange')
    ax5.bar(x_pos + width/2, creative_sims, width, label='Creative Similarity', alpha=0.7, color='green')
    
    ax5.set_xlabel('Task Types')
    ax5.set_ylabel('GloVe Similarity Score')
    ax5.set_title('Task-Specific GloVe Similarities')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(task_names, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Method comparison (where available)
    ax6 = fig.add_subplot(gs[1, 2])
    method_data = []
    labels = []
    
    for i, result in enumerate(results):
        if result['manifold_confidence'] > 0:
            method_data.append([result['traditional_confidence'], result['manifold_confidence']])
            labels.append(f"Test {i+1}")
    
    if method_data:
        method_array = np.array(method_data)
        x_pos = np.arange(len(labels))
        
        ax6.bar(x_pos - 0.2, method_array[:, 0], 0.4, 
               label='Traditional', alpha=0.7, color='blue')
        ax6.bar(x_pos + 0.2, method_array[:, 1], 0.4,
               label='Manifold', alpha=0.7, color='red')
        
        ax6.set_xlabel('Test Cases')
        ax6.set_ylabel('Confidence')
        ax6.set_title('Traditional vs Manifold\nConfidence Comparison')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([f"T{i+1}" for i in range(len(labels))], rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        ax6.text(0.5, 0.5, 'No manifold\ncomparison data', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Method Comparison\n(No Data)')
    
    # 7. Complexity vs Confidence
    ax7 = fig.add_subplot(gs[2, 0])
    complexities = [r['complexity'] for r in results]
    ax7.scatter(complexities, confidences, alpha=0.7, color='brown')
    
    if len(complexities) > 1 and np.std(complexities) > 0:
        z = np.polyfit(complexities, confidences, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(complexities), max(complexities), 100)
        ax7.plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    ax7.set_xlabel('Complexity Score')
    ax7.set_ylabel('Confidence')
    ax7.set_title('Complexity vs Confidence')
    ax7.grid(True, alpha=0.3)
    
    # 8. Feature importance comparison
    ax8 = fig.add_subplot(gs[2, 1])
    feature_names = ['Traditional', 'Manifold', 'GloVe\nCoherence', 'GloVe\nStrength']
    
    # Calculate average feature contributions
    traditional_scores = [r['traditional_confidence'] for r in results if r['traditional_confidence'] > 0]
    manifold_scores = [r['manifold_confidence'] for r in results if r['manifold_confidence'] > 0]
    glove_coherence_scores = [r['glove_coherence'] for r in results]
    glove_norm_scores = [r['glove_embedding_norm'] for r in results]
    
    feature_values = [
        np.mean(traditional_scores) if traditional_scores else 0,
        np.mean(manifold_scores) if manifold_scores else 0,
        np.mean(glove_coherence_scores),
        np.mean(glove_norm_scores) / 10  # Scale down for visualization
    ]
    
    colors = ['blue', 'red', 'green', 'orange']
    bars = ax8.bar(feature_names, feature_values, color=colors, alpha=0.7)
    
    ax8.set_ylabel('Average Score')
    ax8.set_title('Feature Method\nComparison')
    ax8.tick_params(axis='x', rotation=0)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, feature_values):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 9. Analysis time distribution
    ax9 = fig.add_subplot(gs[2, 2])
    analysis_times = [r['analysis_time'] for r in results]
    
    ax9.hist(analysis_times, bins=8, alpha=0.7, color='cyan', edgecolor='black')
    ax9.set_xlabel('Analysis Time (seconds)')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Analysis Time Distribution')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_time = np.mean(analysis_times)
    std_time = np.std(analysis_times)
    ax9.axvline(mean_time, color='red', linestyle='--', alpha=0.8, 
               label=f'Mean: {mean_time:.4f}s')
    ax9.legend()
    
    plt.suptitle('Enhanced Classifier with TF-IDF + GloVe + Manifold Learning\nComprehensive Performance Analysis', 
                 fontsize=16, y=0.98)
    
    plt.savefig('comprehensive_classifier_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive visualization saved as 'comprehensive_classifier_analysis.png'")
    
    plt.close()

def demonstrate_glove_semantic_analysis(classifier: EnhancedInputClassifier):
    """Demonstrate GloVe semantic analysis capabilities"""
    
    print(f"\n🎭 GloVe Semantic Analysis Demonstration")
    print("=" * 50)
    
    # Test semantic similarity between word pairs
    word_pairs = [
        ('algorithm', 'code'),
        ('story', 'narrative'),
        ('equation', 'mathematics'),
        ('research', 'science'),
        ('analyze', 'evaluate'),
        ('hello', 'greeting'),
        ('function', 'method'),
        ('creative', 'imagination')
    ]
    
    print(f"🔗 Word Similarity Analysis:")
    for word1, word2 in word_pairs:
        similarity = classifier.glove_embeddings.get_similarity(word1, word2)
        print(f"  {word1:12} ↔ {word2:12}: {similarity:.3f}")
    
    # Find similar words for key terms
    print(f"\n🎯 Finding Similar Words:")
    test_words = ['algorithm', 'creative', 'analyze', 'equation', 'story', 'debug']
    
    for word in test_words:
        similar_words = classifier.glove_embeddings.find_similar_words(word, top_k=3)
        if similar_words:
            similar_str = ', '.join([f'{w}({s:.2f})' for w, s in similar_words])
            print(f"  '{word}' → {similar_str}")
        else:
            print(f"  '{word}' → No similar words found")
    
    # Semantic clustering of different text types
    print(f"\n🗂️ Semantic Text Clustering Analysis:")
    
    sample_texts = [
        "Solve the quadratic equation using the formula",
        "Write a compelling story about adventure",
        "Implement a sorting algorithm in Python",
        "Analyze the causes of climate change",
        "Calculate the derivative of the function",
        "Create a character with complex motivations",
        "Debug the recursive function implementation",
        "Evaluate the pros and cons of renewable energy"
    ]
    
    # Get embeddings and compute similarities
    embeddings = []
    for text in sample_texts:
        embedding = classifier.glove_embeddings.get_sentence_embedding(text, method='weighted_mean')
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    print(f"  Semantic Similarity Matrix (top 5 pairs):")
    # Find most similar text pairs
    similar_pairs = []
    for i in range(len(sample_texts)):
        for j in range(i+1, len(sample_texts)):
            similarity = similarity_matrix[i, j]
            similar_pairs.append((i, j, similarity))
    
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for idx, (i, j, sim) in enumerate(similar_pairs[:5]):
        text1_short = sample_texts[i][:35] + "..."
        text2_short = sample_texts[j][:35] + "..."
        print(f"    {idx+1}. {text1_short}")
        print(f"       ↔ {text2_short}")
        print(f"       Similarity: {sim:.3f}")
        print()
    
    # Task-domain semantic analysis
    print(f"📊 Task-Domain Semantic Analysis:")
    
    domain_texts = {
        'Mathematical': ['solve equation', 'calculate derivative', 'matrix multiplication'],
        'Creative': ['write story', 'create character', 'compose poem'],
        'Programming': ['implement algorithm', 'debug code', 'write function'],
        'Scientific': ['conduct experiment', 'test hypothesis', 'analyze data']
    }
    
    # Calculate inter-domain similarities
    domain_similarities = {}
    for domain1, texts1 in domain_texts.items():
        for domain2, texts2 in domain_texts.items():
            if domain1 != domain2:
                similarities = []
                for text1 in texts1:
                    emb1 = classifier.glove_embeddings.get_sentence_embedding(text1)
                    for text2 in texts2:
                        emb2 = classifier.glove_embeddings.get_sentence_embedding(text2)
                        sim = cosine_similarity([emb1], [emb2])[0, 0]
                        similarities.append(sim)
                
                avg_sim = np.mean(similarities)
                domain_similarities[f"{domain1}-{domain2}"] = avg_sim
                print(f"  {domain1:12} ↔ {domain2:12}: {avg_sim:.3f}")

def demonstrate_online_learning(classifier: EnhancedInputClassifier):
    """Demonstrate online learning and adaptation capabilities"""
    
    print(f"\n🔄 Online Learning and Adaptation Demonstration")
    print("=" * 55)
    
    # Get initial performance baseline
    print("📊 Initial Classifier State:")
    initial_stats = classifier.get_classification_statistics()
    print(f"  Cache Size: {initial_stats.get('cache_size', 0)}")
    print(f"  Training History: {initial_stats.get('total_predictions', 0)} predictions")
    
    # Simulate user feedback scenarios
    feedback_scenarios = [
        {
            'text': "Optimize the machine learning hyperparameters using grid search",
            'predicted': TaskType.CODE_GENERATION,
            'actual': TaskType.MATHEMATICAL,
            'score': 0.3,
            'reason': 'Optimization is more mathematical than coding'
        },
        {
            'text': "Write a compelling character backstory for the protagonist",
            'predicted': TaskType.CREATIVE_WRITING,
            'actual': TaskType.CREATIVE_WRITING,
            'score': 0.95,
            'reason': 'Perfect classification for creative writing'
        },
        {
            'text': "Analyze the philosophical implications of artificial consciousness",
            'predicted': TaskType.REASONING,
            'actual': TaskType.REASONING,
            'score': 0.88,
            'reason': 'Good reasoning task identification'
        },
        {
            'text': "Debug the memory leak in this C++ application",
            'predicted': TaskType.SCIENTIFIC,
            'actual': TaskType.CODE_GENERATION,
            'score': 0.25,
            'reason': 'Debugging is coding, not scientific research'
        },
        {
            'text': "Calculate the statistical significance of the experimental results",
            'predicted': TaskType.MATHEMATICAL,
            'actual': TaskType.SCIENTIFIC,
            'score': 0.4,
            'reason': 'Statistical analysis in scientific context'
        }
    ]
    
    print(f"\n📚 Processing {len(feedback_scenarios)} feedback scenarios...")
    
    performance_progression = []
    
    for i, scenario in enumerate(feedback_scenarios, 1):
        print(f"\n[{i}] Scenario: {scenario['text'][:45]}...")
        print(f"  Predicted: {scenario['predicted'].value}")
        print(f"  Actual: {scenario['actual'].value}")
        print(f"  Performance: {scenario['score']:.2f}")
        print(f"  Reason: {scenario['reason']}")
        print(f"  Correct: {'✅' if scenario['predicted'] == scenario['actual'] else '❌'}")
        
        # Re-analyze after each update to see improvement
        analysis_before = classifier.analyze_input(scenario['text'])
        confidence_before = analysis_before.confidence
        
        # Update classifier with feedback
        classifier.update_performance(
            text=scenario['text'],
            predicted_task=scenario['predicted'],
            actual_task=scenario['actual'],
            performance_score=scenario['score']
        )
        
        # Analyze again to see changes
        analysis_after = classifier.analyze_input(scenario['text'])
        confidence_after = analysis_after.confidence
        
        print(f"  Confidence Change: {confidence_before:.3f} → {confidence_after:.3f} "
              f"({confidence_after - confidence_before:+.3f})")
        
        performance_progression.append({
            'scenario': i,
            'correct': scenario['predicted'] == scenario['actual'],
            'score': scenario['score'],
            'confidence_before': confidence_before,
            'confidence_after': confidence_after
        })
    
    # Show updated statistics
    print(f"\n📊 Updated Classifier Performance:")
    final_stats = classifier.get_classification_statistics()
    
    print(f"  Overall Accuracy: {final_stats['overall_accuracy']:.2%}")
    print(f"  Total Predictions: {final_stats['total_predictions']}")
    print(f"  Average Performance Score: {final_stats['average_performance_score']:.3f}")
    
    if 'task_specific_stats' in final_stats:
        print(f"  Task-Specific Performance:")
        for task, task_stats in final_stats['task_specific_stats'].items():
            accuracy = task_stats['accuracy']
            correct = task_stats['correct_samples']
            total = task_stats['total_samples']
            print(f"    {task:15}: {accuracy:.2%} ({correct}/{total})")
    
    # Learning progression analysis
    print(f"\n📈 Learning Progression Analysis:")
    correct_predictions = sum(1 for p in performance_progression if p['correct'])
    total_predictions = len(performance_progression)
    
    print(f"  Correct Predictions: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions:.1%})")
    
    avg_confidence_improvement = np.mean([
        p['confidence_after'] - p['confidence_before'] 
        for p in performance_progression
    ])
    print(f"  Average Confidence Change: {avg_confidence_improvement:+.3f}")
    
    # Analyze performance by correctness
    correct_scenarios = [p for p in performance_progression if p['correct']]
    incorrect_scenarios = [p for p in performance_progression if not p['correct']]
    
    if correct_scenarios:
        avg_correct_score = np.mean([p['score'] for p in correct_scenarios])
        print(f"  Average Score (Correct): {avg_correct_score:.3f}")
    
    if incorrect_scenarios:
        avg_incorrect_score = np.mean([p['score'] for p in incorrect_scenarios])
        print(f"  Average Score (Incorrect): {avg_incorrect_score:.3f}")

def run_comprehensive_evaluation(classifier: EnhancedInputClassifier, results: List[Dict]):
    """Run comprehensive evaluation of all classifier features"""
    
    print(f"\n🎯 Comprehensive Classifier Evaluation")
    print("=" * 45)
    
    # Feature analysis
    print("🔍 Feature Analysis Summary:")
    
    # Traditional vs Enhanced performance
    traditional_confidences = [r['traditional_confidence'] for r in results if r['traditional_confidence'] > 0]
    manifold_confidences = [r['manifold_confidence'] for r in results if r['manifold_confidence'] > 0]
    
    if traditional_confidences and manifold_confidences:
        trad_avg = np.mean(traditional_confidences)
        manifold_avg = np.mean(manifold_confidences)
        improvement = ((manifold_avg - trad_avg) / trad_avg) * 100
        
        print(f"  Traditional Method Average: {trad_avg:.3f}")
        print(f"  Enhanced Method Average: {manifold_avg:.3f}")
        print(f"  Performance Improvement: {improvement:+.1f}%")
    
    # GloVe semantic analysis summary
    glove_coherences = [r['glove_coherence'] for r in results]
    glove_norms = [r['glove_embedding_norm'] for r in results]
    
    print(f"\n🎭 GloVe Embeddings Summary:")
    print(f"  Average Semantic Coherence: {np.mean(glove_coherences):.3f}")
    print(f"  Average Embedding Strength: {np.mean(glove_norms):.3f}")
    print(f"  Embeddings Used: {classifier.glove_embeddings.is_loaded}")
    print(f"  Vocabulary Size: {len(classifier.glove_embeddings.vocab)}")
    
    # Manifold learning summary
    manifold_types = [r['best_manifold'] for r in results]
    unique_manifolds = set(manifold_types)
    
    print(f"\n🌐 Manifold Learning Summary:")
    print(f"  Unique Manifolds Used: {len(unique_manifolds)}")
    print(f"  Most Common Manifold: {max(set(manifold_types), key=manifold_types.count)}")
    
    # Performance by complexity
    complexities = [r['complexity'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Divide into low, medium, high complexity
    complexity_threshold_low = np.percentile(complexities, 33)
    complexity_threshold_high = np.percentile(complexities, 67)
    
    low_complexity_results = [r for r in results if r['complexity'] <= complexity_threshold_low]
    medium_complexity_results = [r for r in results if complexity_threshold_low < r['complexity'] <= complexity_threshold_high]
    high_complexity_results = [r for r in results if r['complexity'] > complexity_threshold_high]
    
    print(f"\n📊 Performance by Complexity Level:")
    for category, category_results in [
        ("Low", low_complexity_results),
        ("Medium", medium_complexity_results), 
        ("High", high_complexity_results)
    ]:
        if category_results:
            avg_confidence = np.mean([r['confidence'] for r in category_results])
            avg_uncertainty = np.mean([r['uncertainty'] for r in category_results])
            print(f"  {category:6} Complexity: Confidence={avg_confidence:.3f}, Uncertainty={avg_uncertainty:.3f} ({len(category_results)} cases)")

def main():
    """Main demonstration function"""
    
    print("🚀 Complete Enhanced Input Classifier Demo")
    print("🔥 TF-IDF + GloVe + Manifold Learning Integration")
    print("=" * 65)
    print("Features demonstrated:")
    print("• Traditional pattern-based classification")
    print("• TF-IDF vectorization for lexical features")
    print("• GloVe embeddings for semantic understanding")
    print("• Manifold learning with geometric spaces")
    print("• Uncertainty quantification and confidence scoring")
    print("• Semantic coherence analysis")
    print("• Online learning and performance feedback")
    print("• Multi-modal feature fusion")
    print("• Comprehensive performance analysis")
    print("=" * 65)
    
    try:
        # Step 1: Enhanced classification demonstration
        print("\n🎯 Step 1: Enhanced Classification with Multi-Modal Features")
        results, classifier = demonstrate_enhanced_classification()
        
        # Step 2: GloVe semantic analysis
        print("\n🎯 Step 2: GloVe Semantic Analysis")
        demonstrate_glove_semantic_analysis(classifier)
        
        # Step 3: Performance analysis
        print("\n🎯 Step 3: Performance Analysis")
        analyze_classifier_performance(results, classifier)
        
        # Step 4: Comprehensive visualization
        print("\n🎯 Step 4: Comprehensive Visualization")
        create_comprehensive_visualization(results)
        
        # Step 5: Online learning demonstration
        print("\n🎯 Step 5: Online Learning and Adaptation")
        demonstrate_online_learning(classifier)
        
        # Step 6: Comprehensive evaluation
        print("\n🎯 Step 6: Comprehensive Evaluation")
        run_comprehensive_evaluation(classifier, results)
        
        # Final summary and insights
        print(f"\n🎉 Complete Enhanced Classifier Demo Finished!")
        print("=" * 55)
        print("✅ Demonstrated multi-modal feature fusion (TF-IDF + GloVe)")
        print("✅ Showed geometric manifold learning integration")
        print("✅ Analyzed semantic coherence and word relationships")
        print("✅ Visualized comprehensive performance metrics")
        print("✅ Demonstrated adaptive online learning")
        print("✅ Provided interpretable confidence and uncertainty scores")
        
        # Key insights and recommendations
        print(f"\n💡 Key Insights and Recommendations:")
        final_stats = classifier.get_classification_statistics()
        
        if final_stats.get('total_predictions', 0) > 0:
            print(f"   • Final classification accuracy: {final_stats['overall_accuracy']:.2%}")
        
        # Multi-modal feature insights
        avg_coherence = np.mean([r['glove_coherence'] for r in results])
        avg_embedding_strength = np.mean([r['glove_embedding_norm'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_uncertainty = np.mean([r['uncertainty'] for r in results])
        
        print(f"   • Average semantic coherence: {avg_coherence:.3f}")
        print(f"   • Average classification confidence: {avg_confidence:.3f}")
        print(f"   • Average uncertainty estimate: {avg_uncertainty:.3f}")
        
        # Method comparison insights
        traditional_better = sum(1 for r in results 
                               if r.get('traditional_confidence', 0) > r.get('manifold_confidence', 0))
        manifold_better = sum(1 for r in results 
                            if r.get('manifold_confidence', 0) > r.get('traditional_confidence', 0))
        
        if traditional_better + manifold_better > 0:
            manifold_win_rate = manifold_better / (traditional_better + manifold_better)
            print(f"   • Enhanced method win rate: {manifold_win_rate:.1%}")
        
        print(f"   • Multi-modal approach provides richer semantic understanding")
        print(f"   • Online learning enables continuous improvement")
        print(f"   • Uncertainty quantification helps identify difficult cases")
        print(f"   • Geometric manifolds capture task-specific structures")
        
        # Usage recommendations
        print(f"\n🔧 Usage Recommendations:")
        print(f"   • Use GloVe embeddings for semantic similarity tasks")
        print(f"   • Monitor uncertainty scores for active learning")
        print(f"   • Apply different manifolds based on task characteristics")
        print(f"   • Leverage online learning for domain adaptation")
        print(f"   • Combine traditional and geometric features for robustness")
        
        return {
            'results': results,
            'classifier': classifier,
            'final_stats': final_stats,
            'demo_complete': True
        }
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    demo_results = main()
    
    if demo_results:
        print(f"\n✨ Demo completed successfully!")
        print(f"📁 Generated files:")
        print(f"   • comprehensive_classifier_analysis.png")
        print(f"   • Classifier state saved in demo_results")
    else:
        print(f"\n❌ Demo failed to complete")