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
        })# File: examples/enhanced_classifier_demo.py
"""Demonstration of enhanced input classifier with manifold learning"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import time

from core.input_classifier import EnhancedInputClassifier
from core.types import TaskType, ManifoldLearningConfig

def demonstrate_enhanced_classification():
    """Demonstrate enhanced classification capabilities"""
    
    print("🧠 Enhanced Input Classifier with Manifold Learning")
    print("=" * 55)
    
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
    print("=" * 60)
    
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
        
        # Extract manifold-specific information
        uncertainty = getattr(analysis, 'uncertainty_estimate', 0.0)
        best_manifold = getattr(analysis, 'best_manifold', 'unknown')
        manifold_confidence = analysis.features.get('manifold_confidence', 0.0)
        traditional_confidence = analysis.features.get('traditional_confidence', 0.0)
        
        print(f"  📊 Task Type: {task_type}")
        print(f"  🎯 Confidence: {confidence:.3f}")
        print(f"  🔮 Uncertainty: {uncertainty:.3f}")
        print(f"  🌐 Best Manifold: {best_manifold}")
        print(f"  📈 Complexity: {complexity:.3f}")
        print(f"  🔑 Keywords: {', '.join(keywords)}")
        print(f"  🏷️ Domains: {', '.join(domains) if domains else 'None'}")
        print(f"  ⚡ Analysis Time: {analysis_time:.4f}s")
        
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
            'traditional_confidence': traditional_confidence
        })
    
    return results, classifier

def analyze_classifier_performance(results: List[Dict], classifier: EnhancedInputClassifier):
    """Analyze the performance of the enhanced classifier"""
    
    print(f"\n📊 Classifier Performance Analysis")
    print("=" * 40)
    
    # Task distribution
    task_distribution = {}
    for result in results:
        task = result['task_type']
        task_distribution[task] = task_distribution.get(task, 0) + 1
    
    print(f"📈 Task Distribution:")
    for task, count in task_distribution.items():
        percentage = (count / len(results)) * 100
        print(f"  {task}: {count} ({percentage:.1f}%)")
    
    # Confidence statistics
    confidences = [r['confidence'] for r in results]
    uncertainties = [r['uncertainty'] for r in results]
    complexities = [r['complexity'] for r in results]
    
    print(f"\n📊 Statistics:")
    print(f"  Average Confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
    print(f"  Average Uncertainty: {np.mean(uncertainties):.3f} ± {np.std(uncertainties):.3f}")
    print(f"  Average Complexity: {np.mean(complexities):.3f} ± {np.std(complexities):.3f}")
    print(f"  Analysis Time Range: {min(r['analysis_time'] for r in results):.4f}s - {max(r['analysis_time'] for r in results):.4f}s")
    
    # Manifold usage
    manifold_usage = {}
    for result in results:
        manifold = result['best_manifold']
        manifold_usage[manifold] = manifold_usage.get(manifold, 0) + 1
    
    print(f"\n🌐 Manifold Usage:")
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
        coherence_confidence_corr = np.corrcoef(glove_coherences, confidences)[0, 1]
        print(f"  Coherence-Confidence Correlation: {coherence_confidence_corr:.3f}")
    
    # Method comparison (where available)
    method_comparisons = [(r['traditional_confidence'], r['manifold_confidence']) 
                         for r in results if r['manifold_confidence'] > 0]
    
    if method_comparisons:
        trad_scores = [mc[0] for mc in method_comparisons]
        manifold_scores = [mc[1] for mc in method_comparisons]
        
        print(f"\n⚖️ Method Comparison:")
        print(f"  Traditional Average: {np.mean(trad_scores):.3f}")
        print(f"  Manifold Average: {np.mean(manifold_scores):.3f}")
        
        # Count where manifold was better
        manifold_better = sum(1 for t, m in method_comparisons if m > t)
        print(f"  Manifold Better: {manifold_better}/{len(method_comparisons)} ({manifold_better/len(method_comparisons)*100:.1f}%)")
    
    # Get classifier statistics
    classifier_stats = classifier.get_classification_statistics()
    print(f"\n🔍 Classifier Internal Stats:")
    for key, value in classifier_stats.items():
        if key != 'task_specific_stats' and key != 'manifold_diagnostics':
            print(f"  {key}: {value}")

def create_visualization(results: List[Dict]):
    """Create visualizations of classifier results"""
    
    print(f"\n📈 Creating Performance Visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confidence vs Uncertainty scatter
    confidences = [r['confidence'] for r in results]
    uncertainties = [r['uncertainty'] for r in results]
    complexities = [r['complexity'] for r in results]
    
    scatter = ax1.scatter(confidences, uncertainties, c=complexities, 
                         cmap='viridis', alpha=0.7, s=60)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Uncertainty')
    ax1.set_title('Confidence vs Uncertainty (colored by Complexity)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Complexity')
    
    # 2. Task type distribution
    task_counts = {}
    for result in results:
        task = result['task_type'].replace('_', ' ').title()
        task_counts[task] = task_counts.get(task, 0) + 1
    
    tasks = list(task_counts.keys())
    counts = list(task_counts.values())
    
    ax2.bar(tasks, counts, color='skyblue', alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Task Type Distribution')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Manifold distribution
    manifold_counts = {}
    for result in results:
        manifold = result['best_manifold']
        manifold_counts[manifold] = manifold_counts.get(manifold, 0) + 1
    
    manifolds = list(manifold_counts.keys())
    m_counts = list(manifold_counts.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(manifolds)))
    ax3.pie(m_counts, labels=manifolds, autopct='%1.1f%%', colors=colors)
    ax3.set_title('Best Manifold Distribution')
    
    # 4. Method comparison (where available)
    method_data = []
    labels = []
    
    for i, result in enumerate(results):
        if result['manifold_confidence'] > 0:
            method_data.append([result['traditional_confidence'], result['manifold_confidence']])
            labels.append(f"Test {i+1}")
    
    if method_data:
        method_array = np.array(method_data)
        x_pos = np.arange(len(labels))
        
        ax4.bar(x_pos - 0.2, method_array[:, 0], 0.4, 
               label='Traditional', alpha=0.7, color='blue')
        ax4.bar(x_pos + 0.2, method_array[:, 1], 0.4,
               label='Manifold', alpha=0.7, color='red')
        
        ax4.set_xlabel('Test Cases')
        ax4.set_ylabel('Confidence')
        ax4.set_title('Traditional vs Manifold Confidence')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"T{i+1}" for i in range(len(labels))], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No manifold comparison data', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Method Comparison (No Data)')
    
    plt.tight_layout()
    plt.savefig('enhanced_classifier_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved as 'enhanced_classifier_analysis.png'")
    
    plt.close()

def demonstrate_online_learning(classifier: EnhancedInputClassifier):
    """Demonstrate online learning capabilities"""
    
    print(f"\n🔄 Online Learning Demonstration")
    print("=" * 35)
    
    # Simulate user feedback and online updates
    feedback_examples = [
        {
            'text': "Optimize the machine learning hyperparameters",
            'predicted': TaskType.CODE_GENERATION,
            'actual': TaskType.MATHEMATICAL,
            'score': 0.3  # Poor performance, need correction
        },
        {
            'text': "Write a compelling character backstory",
            'predicted': TaskType.CREATIVE_WRITING,
            'actual': TaskType.CREATIVE_WRITING,
            'score': 0.9  # Good performance
        },
        {
            'text': "Analyze the philosophical implications of consciousness",
            'predicted': TaskType.REASONING,
            'actual': TaskType.REASONING,
            'score': 0.85  # Good performance
        },
        {
            'text': "Debug the memory leak in this C++ code",
            'predicted': TaskType.SCIENTIFIC,
            'actual': TaskType.CODE_GENERATION,
            'score': 0.2  # Poor performance, need correction
        }
    ]
    
    print(f"📚 Processing {len(feedback_examples)} feedback examples...")
    
    for i, feedback in enumerate(feedback_examples, 1):
        print(f"\n[{i}] Feedback: {feedback['text'][:40]}...")
        print(f"  Predicted: {feedback['predicted'].value}")
        print(f"  Actual: {feedback['actual'].value}")
        print(f"  Performance: {feedback['score']:.2f}")
        print(f"  Correct: {'✅' if feedback['predicted'] == feedback['actual'] else '❌'}")
        
        # Update classifier with feedback
        classifier.update_performance(
            text=feedback['text'],
            predicted_task=feedback['predicted'],
            actual_task=feedback['actual'],
            performance_score=feedback['score']
        )
    
    # Show updated statistics
    print(f"\n📊 Updated Classifier Statistics:")
    stats = classifier.get_classification_statistics()
    
    print(f"  Overall Accuracy: {stats['overall_accuracy']:.2%}")
    print(f"  Total Predictions: {stats['total_predictions']}")
    print(f"  Average Performance Score: {stats['average_performance_score']:.3f}")
    
    if 'task_specific_stats' in stats:
        print(f"  Task-Specific Performance:")
        for task, task_stats in stats['task_specific_stats'].items():
            print(f"    {task}: {task_stats['accuracy']:.2%} ({task_stats['correct_samples']}/{task_stats['total_samples']})")

def create_visualization(results: List[Dict]):
    """Create visualizations of classifier results including GloVe features"""
    
    print(f"\n📈 Creating Performance Visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confidence vs Uncertainty colored by GloVe coherence
    confidences = [r['confidence'] for r in results]
    uncertainties = [r['uncertainty'] for r in results]
    glove_coherences = [r['glove_coherence'] for r in results]
    
    scatter = ax1.scatter(confidences, uncertainties, c=glove_coherences, 
                         cmap='viridis', alpha=0.7, s=60)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Uncertainty')
    ax1.set_title('Confidence vs Uncertainty (colored by GloVe Coherence)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Semantic Coherence')
    
    # 2. Task-specific GloVe similarities
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
    tasks = list(task_glove_data.keys())
    math_sims = [np.mean([d['math_sim'] for d in task_glove_data[task]]) for task in tasks]
    creative_sims = [np.mean([d['creative_sim'] for d in task_glove_data[task]]) for task in tasks]
    
    x_pos = np.arange(len(tasks))
    width = 0.35
    
    ax2.bar(x_pos - width/2, math_sims, width, label='Math Similarity', alpha=0.7)
    ax2.bar(x_pos + width/2, creative_sims, width, label='Creative Similarity', alpha=0.7)
    
    ax2.set_xlabel('Task Types')
    ax2.set_ylabel('GloVe Similarity Score')
    ax2.set_title('Task-Specific GloVe Similarities')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tasks, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Semantic coherence vs confidence
    ax3.scatter(glove_coherences, confidences, alpha=0.7, color='purple')
    
    # Add trend line
    if len(glove_coherences) > 1:
        z = np.polyfit(glove_coherences, confidences, 1)
        p = np.poly1d(z)
        ax3.plot(glove_coherences, p(glove_coherences), "r--", alpha=0.8)
    
    ax3.set_xlabel('GloVe Semantic Coherence')
    ax3.set_ylabel('Classification Confidence')
    ax3.set_title('Semantic Coherence vs Classification Confidence')
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature importance comparison
    feature_names = ['Traditional', 'Manifold', 'GloVe Coherence', 'GloVe Strength']
    feature_values = []
    
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
    bars = ax4.bar(feature_names, feature_values, color=colors, alpha=0.7)
    
    ax4.set_ylabel('Average Score')
    ax4.set_title('Feature Method Comparison')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, feature_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('enhanced_classifier_with_glove.png', dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved as 'enhanced_classifier_with_glove.png'")
    
    plt.close()

def demonstrate_glove_semantic_analysis(classifier: EnhancedInputClassifier):
    """Demonstrate GloVe semantic analysis capabilities"""
    
    print(f"\n🎭 GloVe Semantic Analysis Demonstration")
    print("=" * 45)
    
    # Test semantic similarity
    word_pairs = [
        ('algorithm', 'code'),
        ('story', 'narrative'),
        ('equation', 'mathematics'),
        ('research', 'science'),
        ('analyze', 'evaluate'),
        ('hello', 'greeting')
    ]
    
    print(f"🔗 Word Similarity Analysis:")
    for word1, word2 in word_pairs:
        similarity = classifier.glove_embeddings.get_similarity(word1, word2)
        print(f"  {word1} ↔ {word2}: {similarity:.3f}")
    
    # Find similar words
    print(f"\n🎯 Finding Similar Words:")
    test_words = ['algorithm', 'creative', 'analyze', 'equation']
    
    for word in test_words:
        similar_words = classifier.glove_embeddings.find_similar_words(word, top_k=3)
        if similar_words:
            print(f"  '{word}' → {', '.join([f'{w}({s:.2f})' for w, s in similar_words])}")
        else:
            print(f"  '{word}' → No similar words found")
    
    # Semantic clustering of different text types
    print(f"\n🗂️ Semantic Text Clustering:")
    
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
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    print(f"  Semantic Similarity Matrix (top 3 pairs):")
    # Find most similar text pairs
    similar_pairs = []
    for i in range(len(sample_texts)):
        for j in range(i+1, len(sample_texts)):
            similarity = similarity_matrix[i, j]
            similar_pairs.append((i, j, similarity))
    
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (idx1, idx2, sim) in enumerate(similar_pairs[:3]):
        text1_short = sample_texts[idx1][:30] + "..."
        text2_short = sample_texts[idx2][:30] + "..."
        print(f"    {i+1}. {text1_short} ↔ {text2_short} ({sim:.3f})")

def main():
    """Main demonstration function"""
    
    print("🚀 Enhanced Input Classifier with TF-IDF + GloVe Demo")
    print("=" * 65)
    print("Features demonstrated:")
    print("• Traditional pattern-based classification")
    print("• TF-IDF vectorization for lexical features")
    print("• GloVe embeddings for semantic understanding")
    print("• Manifold learning integration")
    print("• Uncertainty quantification")
    print("• Semantic coherence analysis")
    print("• Online learning and feedback")
    print("• Multi-modal feature fusion")
    print("=" * 65)
    
    try:
        # Step 1: Demonstrate enhanced classification
        print("\n🎯 Step 1: Enhanced Classification with GloVe")
        results, classifier = demonstrate_enhanced_classification()
        
        # Step 2: GloVe semantic analysis
        print("\n🎯 Step 2: GloVe Semantic Analysis")
        demonstrate_glove_semantic_analysis(classifier)
        
        # Step 3: Analyze performance
        print("\n🎯 Step 3: Performance Analysis")
        analyze_classifier_performance(results, classifier)
        
        # Step 4: Create visualizations
        print("\n🎯 Step 4: Visualization")
        create_visualization(results)
        
        # Step 5: Demonstrate online learning
        print("\n🎯 Step 5: Online Learning")
        demonstrate_online_learning(classifier)
        
        # Final summary
        print(f"\n🎉 Enhanced Classifier with GloVe Demo Complete!")
        print("=" * 55)
        print("✅ Demonstrated TF-IDF + GloVe feature fusion")
        print("✅ Showed semantic coherence analysis")
        print("✅ Analyzed word-level similarities")
        print("✅ Visualized multi-modal classifications")
        print("✅ Demonstrated online learning")
        
        # Key insights
        print(f"\n💡 Key Insights:")
        final_stats = classifier.get_classification_statistics()
        
        if final_stats.get('total_predictions', 0) > 0:
            print(f"   • Classification accuracy: {final_stats['overall_accuracy']:.2%}")
        
        # GloVe-specific insights
        avg_coherence = np.mean([r['glove_coherence'] for r in results])
        avg_embedding_strength = np.mean([r['glove_embedding_norm'] for r in results])
        
        print(f"   • Average semantic coherence: {avg_coherence:.3f}")
        print(f"   • Average embedding strength: {avg_embedding_strength:.3f}")
        
        # Feature fusion benefits
        traditional_better = sum(1 for r in results 
                               if r.get('traditional_confidence', 0) > r.get('manifold_confidence', 0))
        manifold_better = sum(1 for r in results 
                            if r.get('manifold_confidence', 0) > r.get('traditional_confidence', 0))
        
        print(f"   • Traditional method better: {traditional_better} cases")
        print(f"   • Manifold+GloVe better: {manifold_better} cases")
        print(f"   • Multi-modal feature fusion provides richer analysis")
        
        return {
            'results': results,
            'classifier': classifier,
            'final_stats': final_stats
        }
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    main()