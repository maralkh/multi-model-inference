# File: examples/geometric_manifold_demo.py
"""Advanced demonstration of geometric and Bayesian manifold learning"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any

from core.geometric_embeddings import (
    GeometricBayesianManifoldLearner,
    SphereManifold, TorusManifold, HyperbolicManifold
)
from core.manifold_learner import ManifoldLearner
from core.input_types import ManifoldLearningConfig, TaskType
from core.multi_model_engine import MultiModelInferenceEngine
from utils.dummy_models import create_specialized_models, DummyTokenizer

def demonstrate_geometric_manifolds():
    """Demonstrate different geometric manifolds"""
    
    print("🔺 Geometric Manifolds Demonstration")
    print("=" * 50)
    
    # Generate sample data for different manifold types
    n_samples = 100
    
    # Spherical data (directional text features)
    print("\n🌐 Testing Spherical Manifold...")
    sphere_texts = [
        "Navigate north towards the mountain peak",
        "Turn around and face the opposite direction", 
        "The satellite orbits in a circular path",
        "Rotate the object 90 degrees clockwise",
        "Point the antenna towards the sky",
        "The compass needle points to magnetic north",
        "Spherical coordinates define position in 3D space",
        "Global positioning requires directional references"
    ]
    
    sphere_learner = GeometricBayesianManifoldLearner(
        manifold_type="sphere",
        bayesian_learning=True,
        dimension=3,
        radius=1.0
    )
    
    # Create feature vectors (simplified)
    sphere_features = np.random.randn(len(sphere_texts), 10)
    sphere_learner.fit(sphere_features)
    
    # Test embedding and uncertainty
    test_features = np.random.randn(5, 10)
    embeddings, uncertainties = sphere_learner.transform(test_features, return_uncertainty=True)
    
    print(f"  ✅ Sphere embeddings shape: {embeddings.shape}")
    print(f"  📊 Average uncertainty: {np.mean(uncertainties):.3f}")
    
    # Visualize sphere
    fig = sphere_learner.visualize_manifold(test_features, show_uncertainty=True)
    plt.title('Spherical Manifold with Uncertainty')
    plt.savefig('sphere_manifold_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Torus data (periodic/cyclic patterns)
    print("\n🍩 Testing Torus Manifold...")
    torus_texts = [
        "The daily cycle repeats every 24 hours",
        "Seasonal patterns occur annually",
        "The machine operates in recurring cycles",
        "Weekly schedules follow the same pattern",
        "Periodic maintenance every month",
        "The rhythm repeats throughout the song",
        "Circular processes in manufacturing",
        "Cyclical economic patterns over decades"
    ]
    
    torus_learner = GeometricBayesianManifoldLearner(
        manifold_type="torus",
        bayesian_learning=True,
        major_radius=2.0,
        minor_radius=1.0
    )
    
    torus_features = np.random.randn(len(torus_texts), 10)
    torus_learner.fit(torus_features)
    
    embeddings, uncertainties = torus_learner.transform(test_features, return_uncertainty=True)
    print(f"  ✅ Torus embeddings shape: {embeddings.shape}")
    print(f"  📊 Average uncertainty: {np.mean(uncertainties):.3f}")
    
    # Visualize torus
    fig = torus_learner.visualize_manifold(test_features, show_uncertainty=True)
    plt.title('Torus Manifold with Uncertainty')
    plt.savefig('torus_manifold_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Hyperbolic data (hierarchical structures)
    print("\n📐 Testing Hyperbolic Manifold...")
    hyperbolic_texts = [
        "The organizational tree has multiple levels",
        "Hierarchical classification of species",
        "Parent nodes contain child elements",
        "Multi-level decision making process",
        "Branching structure of the river system",
        "Taxonomic hierarchy in biology",
        "Corporate structure with departments",
        "Nested categories and subcategories"
    ]
    
    hyperbolic_learner = GeometricBayesianManifoldLearner(
        manifold_type="hyperbolic",
        bayesian_learning=True,
        dimension=2
    )
    
    hyperbolic_features = np.random.randn(len(hyperbolic_texts), 10)
    hyperbolic_learner.fit(hyperbolic_features)
    
    embeddings, uncertainties = hyperbolic_learner.transform(test_features, return_uncertainty=True)
    print(f"  ✅ Hyperbolic embeddings shape: {embeddings.shape}")
    print(f"  📊 Average uncertainty: {np.mean(uncertainties):.3f}")
    
    # Test geodesic interpolation
    print("\n🛤️ Testing Geodesic Interpolation...")
    
    # Sphere geodesic
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 1, 0])
    sphere_path = sphere_learner.interpolate_geodesic(p1, p2, n_steps=10)
    print(f"  🌐 Sphere geodesic path length: {len(sphere_path)}")
    
    # Torus geodesic
    t1 = np.array([2, 0, 1])  # Point on torus
    t2 = np.array([0, 2, -1])
    torus_path = torus_learner.interpolate_geodesic(t1, t2, n_steps=10)
    print(f"  🍩 Torus geodesic path length: {len(torus_path)}")
    
    return {
        'sphere_learner': sphere_learner,
        'torus_learner': torus_learner,
        'hyperbolic_learner': hyperbolic_learner
    }

def demonstrate_bayesian_learning():
    """Demonstrate Bayesian aspects of manifold learning"""
    
    print("\n🎲 Bayesian Learning Demonstration")
    print("=" * 40)
    
    # Create enhanced manifold learner
    config = ManifoldLearningConfig(
        embedding_dim=20,
        manifold_method="sphere",  # Will use geometric sphere
        enable_online_learning=True,
        enable_clustering=True
    )
    
    manifold_learner = ManifoldLearner(config)
    
    # Training texts with different complexity levels
    training_texts = [
        # Mathematical (high structure)
        "Solve the differential equation dy/dx = x²",
        "Calculate the integral of sin(x) from 0 to π",
        "Find the eigenvalues of the matrix",
        "Prove the fundamental theorem of calculus",
        
        # Creative (medium structure)
        "Write a story about a magical forest",
        "Create a poem about the changing seasons",
        "Describe a character's emotional journey",
        "Imagine a world where time flows backwards",
        
        # Technical (high structure)
        "Implement a binary search algorithm",
        "Design a neural network architecture",
        "Optimize database query performance",
        "Debug the memory leak in the application",
        
        # Conversational (low structure)
        "How are you doing today?",
        "What's your favorite color?",
        "Thanks for the help!",
        "Have a great weekend!"
    ]
    
    print(f"🧠 Training manifold learner with {len(training_texts)} texts...")
    manifold_learner.learn_manifold_offline(training_texts)
    
    # Test uncertainty quantification
    test_texts = [
        "Solve the complex optimization problem with constraints",  # High uncertainty (complex)
        "Hello, how can I help you?",  # Low uncertainty (simple)
        "Analyze the quantum mechanical wave function",  # Medium uncertainty
        "Write a haiku about artificial intelligence"  # Medium uncertainty
    ]
    
    print("\n🔍 Testing Uncertainty Quantification...")
    uncertainty_results = []
    
    for i, text in enumerate(test_texts):
        recommendations = manifold_learner.get_recommendations(text)
        
        uncertainty = recommendations.get('uncertainty_estimate', 0.0)
        confidence = recommendations.get('confidence', 0.0)
        traditional_confidence = recommendations.get('traditional_confidence', 0.0)
        best_manifold = recommendations.get('best_manifold', 'unknown')
        
        print(f"\n[{i+1}] Text: {text[:50]}...")
        print(f"    Best Manifold: {best_manifold}")
        print(f"    Uncertainty: {uncertainty:.3f}")
        print(f"    Bayesian Confidence: {confidence:.3f}")
        print(f"    Traditional Confidence: {traditional_confidence:.3f}")
        print(f"    Recommended Models: {recommendations.get('recommended_models', [])}")
        
        uncertainty_results.append({
            'text': text,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'best_manifold': best_manifold
        })
    
    # Test sampling from posterior
    print("\n🎯 Testing Posterior Sampling...")
    
    sample_text = "Implement a machine learning model for text classification"
    samples = manifold_learner.sample_embeddings_bayesian(sample_text, n_samples=10)
    
    print(f"  Generated {len(samples)} embedding samples")
    print(f"  Sample embeddings shape: {samples.shape}")
    print(f"  Sample variance: {np.var(samples, axis=0).mean():.3f}")
    
    # Test active learning
    print("\n🎯 Testing Active Learning...")
    
    candidate_texts = [
        "Optimize the neural network hyperparameters",
        "What time is it?",
        "Derive the equations of general relativity",
        "Nice weather today!",
        "Implement distributed computing algorithms",
        "Thanks for your time",
        "Analyze protein folding mechanisms"
    ]
    
    active_candidates = manifold_learner.get_active_learning_candidates(
        candidate_texts, n_candidates=3
    )
    
    print(f"  Top 3 Active Learning Candidates:")
    for i, (text, score) in enumerate(active_candidates, 1):
        print(f"    {i}. {text[:50]}... (score: {score:.3f})")
    
    # Create enhanced visualization
    print("\n📊 Creating Enhanced Visualization...")
    manifold_learner.visualize_enhanced_manifold('enhanced_manifold_demo.png')
    
    # Get diagnostics
    diagnostics = manifold_learner.get_manifold_diagnostics()
    print(f"\n🔍 Manifold Diagnostics:")
    print(f"  Traditional Method: {diagnostics['traditional_manifold']['method']}")
    print(f"  Geometric Manifolds: {list(diagnostics['geometric_manifolds'].keys())}")
    print(f"  Data Points: {diagnostics['data_points_count']}")
    print(f"  Clusters Found: {diagnostics['clusters_found']}")
    
    return uncertainty_results, active_candidates, diagnostics

def demonstrate_integrated_system():
    """Demonstrate integration with multi-model system"""
    
    print("\n🔗 Integrated System Demonstration")
    print("=" * 40)
    
    # Create models and enhanced engine
    models = create_specialized_models()
    tokenizer = DummyTokenizer()
    
    # Create engine with enhanced manifold learning
    engine = MultiModelInferenceEngine(models, tokenizer)
    
    # Replace with enhanced manifold learner
    enhanced_config = ManifoldLearningConfig(
        embedding_dim=30,
        manifold_method="sphere",
        enable_online_learning=True,
        enable_clustering=True
    )
    
    enhanced_manifold_learner = ManifoldLearner(enhanced_config)
    engine.classifier.manifold_learner = enhanced_manifold_learner
    
    # Training data for enhanced learning
    training_data = [
        {'text': 'Solve quadratic equations using the formula', 'label': 'math'},
        {'text': 'Write creative fiction stories', 'label': 'creative'},
        {'text': 'Implement sorting algorithms efficiently', 'label': 'code'},
        {'text': 'Analyze complex philosophical arguments', 'label': 'reasoning'},
        {'text': 'Calculate derivatives and integrals', 'label': 'math'},
        {'text': 'Design user interface components', 'label': 'code'},
        {'text': 'Create poetic expressions of emotion', 'label': 'creative'},
        {'text': 'Evaluate logical propositions', 'label': 'reasoning'}
    ]
    
    # Train enhanced manifold learner
    texts = [item['text'] for item in training_data]
    enhanced_manifold_learner.learn_manifold_offline(texts)
    
    # Test enhanced system
    test_prompts = [
        "Solve the complex matrix eigenvalue problem",
        "Write a science fiction story about time travel",
        "Implement a graph traversal algorithm",
        "Analyze the ethical implications of AI",
        "Calculate the Fourier transform",
        "Debug the recursive function",
        "Create a narrative about space exploration",
        "Evaluate competing philosophical theories"
    ]
    
    results = []
    
    print(f"\n🧪 Testing Enhanced System with {len(test_prompts)} prompts...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] Testing: {prompt[:50]}...")
        
        # Generate with enhanced system
        result = engine.generate(prompt, max_length=150)
        
        # Extract enhanced information
        selected_model = result['selected_model']
        analysis = result['input_analysis']
        generation_time = result['generation_time']
        
        # Get manifold-specific information
        if hasattr(engine.classifier, 'manifold_learner'):
            manifold_rec = engine.classifier.manifold_learner.get_recommendations(prompt)
            uncertainty = manifold_rec.get('uncertainty_estimate', 0.0)
            best_manifold = manifold_rec.get('best_manifold', 'unknown')
            bayesian_confidence = manifold_rec.get('confidence', 0.0)
        else:
            uncertainty = 0.0
            best_manifold = 'none'
            bayesian_confidence = 0.0
        
        print(f"  Selected Model: {selected_model}")
        print(f"  Task Type: {analysis['task_type']}")
        print(f"  Best Manifold: {best_manifold}")
        print(f"  Uncertainty: {uncertainty:.3f}")
        print(f"  Bayesian Confidence: {bayesian_confidence:.3f}")
        print(f"  Generation Time: {generation_time:.3f}s")
        
        results.append({
            'prompt': prompt,
            'selected_model': selected_model,
            'task_type': analysis['task_type'],
            'best_manifold': best_manifold,
            'uncertainty': uncertainty,
            'bayesian_confidence': bayesian_confidence,
            'generation_time': generation_time
        })
        
        # Simulate online learning update
        performance_score = np.random.uniform(0.7, 0.95)  # Mock performance
        task_type = TaskType(analysis['task_type'])
        
        enhanced_manifold_learner.update_online(
            text=prompt,
            task_type=task_type,
            selected_model=selected_model,
            performance_score=performance_score
        )
    
    # Analyze results
    print(f"\n📊 Enhanced System Analysis:")
    
    # Group by manifold type
    manifold_usage = {}
    for result in results:
        manifold = result['best_manifold']
        manifold_usage[manifold] = manifold_usage.get(manifold, 0) + 1
    
    print(f"  Manifold Usage:")
    for manifold, count in manifold_usage.items():
        percentage = (count / len(results)) * 100
        print(f"    {manifold}: {count} times ({percentage:.1f}%)")
    
    # Average uncertainty by task type
    task_uncertainties = {}
    for result in results:
        task = result['task_type']
        if task not in task_uncertainties:
            task_uncertainties[task] = []
        task_uncertainties[task].append(result['uncertainty'])
    
    print(f"  Average Uncertainty by Task:")
    for task, uncertainties in task_uncertainties.items():
        avg_uncertainty = np.mean(uncertainties)
        print(f"    {task}: {avg_uncertainty:.3f}")
    
    # Model selection accuracy (simulated)
    expected_models = {
        'mathematical': 'math_specialist',
        'creative_writing': 'creative_specialist', 
        'code_generation': 'code_specialist',
        'reasoning': 'reasoning_specialist'
    }
    
    correct_selections = 0
    for result in results:
        expected = expected_models.get(result['task_type'], 'general_model')
        if result['selected_model'] == expected:
            correct_selections += 1
    
    accuracy = correct_selections / len(results)
    print(f"  Model Selection Accuracy: {accuracy:.2%}")
    
    return results

def create_comparative_analysis():
    """Compare traditional vs enhanced manifold learning"""
    
    print("\n⚖️ Comparative Analysis: Traditional vs Enhanced")
    print("=" * 55)
    
    # Test data
    test_texts = [
        "Solve differential equations with boundary conditions",
        "Write a romantic story set in Victorian England", 
        "Implement a distributed hash table algorithm",
        "Analyze the philosophical concept of consciousness",
        "Calculate Fourier series expansions",
        "Create interactive web components",
        "Compose a sonnet about artificial intelligence",
        "Evaluate logical fallacies in arguments"
    ]
    
    # Traditional approach
    print("🔹 Testing Traditional Manifold Learning...")
    traditional_config = ManifoldLearningConfig(
        embedding_dim=20,
        manifold_method="pca",
        enable_online_learning=False,
        enable_clustering=True
    )
    
    traditional_learner = ManifoldLearner(traditional_config)
    # Temporarily disable geometric learners for traditional comparison
    traditional_learner.geometric_learners = {}
    traditional_learner.learn_manifold_offline(test_texts)
    
    traditional_results = []
    for text in test_texts:
        recommendations = traditional_learner.get_recommendations(text)
        traditional_results.append({
            'text': text,
            'confidence': recommendations.get('confidence', 0.0),
            'recommended_models': recommendations.get('recommended_models', []),
            'complexity': recommendations.get('complexity_estimate', 0.0)
        })
    
    # Enhanced approach
    print("🔸 Testing Enhanced Manifold Learning...")
    enhanced_config = ManifoldLearningConfig(
        embedding_dim=20,
        manifold_method="sphere",
        enable_online_learning=True,
        enable_clustering=True
    )
    
    enhanced_learner = ManifoldLearner(enhanced_config)
    enhanced_learner.learn_manifold_offline(test_texts)
    
    enhanced_results = []
    for text in test_texts:
        recommendations = enhanced_learner.get_recommendations(text)
        enhanced_results.append({
            'text': text,
            'confidence': recommendations.get('confidence', 0.0),
            'bayesian_confidence': recommendations.get('confidence', 0.0),
            'uncertainty': recommendations.get('uncertainty_estimate', 0.0),
            'best_manifold': recommendations.get('best_manifold', 'unknown'),
            'recommended_models': recommendations.get('recommended_models', []),
            'complexity': recommendations.get('complexity_estimate', 0.0)
        })
    
    # Comparison analysis
    print(f"\n📊 Comparison Results:")
    
    trad_avg_confidence = np.mean([r['confidence'] for r in traditional_results])
    enh_avg_confidence = np.mean([r['confidence'] for r in enhanced_results])
    
    print(f"  Average Confidence:")
    print(f"    Traditional: {trad_avg_confidence:.3f}")
    print(f"    Enhanced: {enh_avg_confidence:.3f}")
    print(f"    Improvement: {((enh_avg_confidence - trad_avg_confidence) / trad_avg_confidence * 100):+.1f}%")
    
    # Uncertainty information (only available in enhanced)
    avg_uncertainty = np.mean([r['uncertainty'] for r in enhanced_results])
    print(f"  Average Uncertainty (Enhanced only): {avg_uncertainty:.3f}")
    
    # Manifold distribution (enhanced only)
    manifold_dist = {}
    for result in enhanced_results:
        manifold = result['best_manifold']
        manifold_dist[manifold] = manifold_dist.get(manifold, 0) + 1
    
    print(f"  Manifold Selection Distribution:")
    for manifold, count in manifold_dist.items():
        percentage = (count / len(enhanced_results)) * 100
        print(f"    {manifold}: {percentage:.1f}%")
    
    # Create visualization comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confidence comparison
    indices = range(len(test_texts))
    trad_confidences = [r['confidence'] for r in traditional_results]
    enh_confidences = [r['confidence'] for r in enhanced_results]
    
    ax1.bar([i - 0.2 for i in indices], trad_confidences, 0.4, 
            label='Traditional', alpha=0.7, color='blue')
    ax1.bar([i + 0.2 for i in indices], enh_confidences, 0.4,
            label='Enhanced', alpha=0.7, color='red')
    ax1.set_xlabel('Test Cases')
    ax1.set_ylabel('Confidence')
    ax1.set_title('Confidence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Uncertainty visualization (enhanced only)
    uncertainties = [r['uncertainty'] for r in enhanced_results]
    ax2.bar(indices, uncertainties, color='green', alpha=0.7)
    ax2.set_xlabel('Test Cases')
    ax2.set_ylabel('Uncertainty')
    ax2.set_title('Uncertainty Estimates (Enhanced)')
    ax2.grid(True, alpha=0.3)
    
    # Manifold distribution pie chart
    manifold_names = list(manifold_dist.keys())
    manifold_counts = list(manifold_dist.values())
    ax3.pie(manifold_counts, labels=manifold_names, autopct='%1.1f%%')
    ax3.set_title('Best Manifold Distribution')
    
    # Complexity comparison
    trad_complexity = [r['complexity'] for r in traditional_results]
    enh_complexity = [r['complexity'] for r in enhanced_results]
    
    ax4.scatter(trad_complexity, enh_complexity, alpha=0.7)
    ax4.plot([0, 1], [0, 1], '--', color='gray', label='Equal Complexity')
    ax4.set_xlabel('Traditional Complexity')
    ax4.set_ylabel('Enhanced Complexity')
    ax4.set_title('Complexity Estimation Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('manifold_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Comparison visualization saved as 'manifold_comparison_analysis.png'")
    
    return {
        'traditional_results': traditional_results,
        'enhanced_results': enhanced_results,
        'comparison_metrics': {
            'traditional_avg_confidence': trad_avg_confidence,
            'enhanced_avg_confidence': enh_avg_confidence,
            'confidence_improvement': (enh_avg_confidence - trad_avg_confidence) / trad_avg_confidence * 100,
            'average_uncertainty': avg_uncertainty,
            'manifold_distribution': manifold_dist
        }
    }

def main():
    """Main demonstration function for geometric and Bayesian manifolds"""
    
    print("🚀 Advanced Geometric & Bayesian Manifold Learning Demo")
    print("=" * 65)
    print("Features demonstrated:")
    print("• Geometric manifolds (Sphere, Torus, Hyperbolic)")
    print("• Bayesian uncertainty quantification")
    print("• Active learning with acquisition functions")
    print("• Geodesic interpolation on manifolds")
    print("• Enhanced multi-model integration")
    print("• Comparative analysis vs traditional methods")
    print("=" * 65)
    
    try:
        # Step 1: Geometric manifolds
        print("\n🎯 Step 1: Geometric Manifolds")
        geometric_results = demonstrate_geometric_manifolds()
        
        # Step 2: Bayesian learning
        print("\n🎯 Step 2: Bayesian Learning")
        uncertainty_results, active_candidates, diagnostics = demonstrate_bayesian_learning()
        
        # Step 3: Integrated system
        print("\n🎯 Step 3: Integrated System")
        integration_results = demonstrate_integrated_system()
        
        # Step 4: Comparative analysis
        print("\n🎯 Step 4: Comparative Analysis")
        comparison_results = compare_comparative_analysis()
        
        # Final summary
        print("\n🎉 Advanced Demo Complete!")
        print("=" * 40)
        print("✅ Demonstrated 3 geometric manifold types")
        print("✅ Showed Bayesian uncertainty quantification")
        print("✅ Tested active learning capabilities")
        print("✅ Integrated with multi-model system")
        print("✅ Compared traditional vs enhanced approaches")
        
        # Key insights
        print(f"\n💡 Key Insights:")
        
        if 'comparison_metrics' in comparison_results:
            metrics = comparison_results['comparison_metrics']
            improvement = metrics['confidence_improvement']
            print(f"   • Enhanced approach improves confidence by {improvement:+.1f}%")
            print(f"   • Average uncertainty estimate: {metrics['average_uncertainty']:.3f}")
            
            manifold_dist = metrics['manifold_distribution']
            most_used = max(manifold_dist.keys(), key=lambda k: manifold_dist[k])
            print(f"   • Most suitable manifold: {most_used}")
        
        print(f"   • Bayesian learning provides uncertainty estimates")
        print(f"   • Geometric manifolds capture different text structures")
        print(f"   • Active learning identifies informative samples")
        
        # Recommendations
        print(f"\n🔧 Recommendations:")
        print(f"   • Use sphere manifold for directional/global content")
        print(f"   • Use torus manifold for periodic/cyclic patterns")
        print(f"   • Use hyperbolic manifold for hierarchical structures")
        print(f"   • Monitor uncertainty for active learning")
        print(f"   • Combine multiple manifolds for robust learning")
        
        return {
            'geometric_results': geometric_results,
            'uncertainty_results': uncertainty_results,
            'integration_results': integration_results,
            'comparison_results': comparison_results
        }
        
    except Exception as e:
        print(f"❌ Advanced demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    main()