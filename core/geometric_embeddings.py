# File: core/geometric_embeddings.py
"""Advanced geometric embeddings and Bayesian learning for manifold analysis"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import scipy.spatial.distance as distance
from scipy.optimize import minimize

class GeometricManifold(ABC):
    """Abstract base class for geometric manifolds"""
    
    @abstractmethod
    def embed(self, points: np.ndarray) -> np.ndarray:
        """Embed points onto the manifold"""
        pass
    
    @abstractmethod
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate geodesic distance between two points on manifold"""
        pass
    
    @abstractmethod
    def tangent_space(self, point: np.ndarray) -> np.ndarray:
        """Get tangent space basis at given point"""
        pass

class SphereManifold(GeometricManifold):
    """Spherical manifold embedding (S^n)"""
    
    def __init__(self, dimension: int = 3, radius: float = 1.0):
        self.dimension = dimension
        self.radius = radius
        
    def embed(self, points: np.ndarray) -> np.ndarray:
        """Embed points onto n-sphere using stereographic projection"""
        # Normalize to unit sphere first
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = points / norms
        
        # Scale by radius
        return normalized * self.radius
    
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate great circle distance on sphere"""
        # Normalize points to unit sphere
        p1_norm = point1 / np.linalg.norm(point1)
        p2_norm = point2 / np.linalg.norm(point2)
        
        # Calculate angle using dot product
        cos_angle = np.clip(np.dot(p1_norm, p2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return self.radius * angle
    
    def tangent_space(self, point: np.ndarray) -> np.ndarray:
        """Get tangent space basis vectors at point on sphere"""
        point_norm = point / np.linalg.norm(point)
        
        # Create orthogonal basis excluding the normal vector
        basis = []
        for i in range(len(point)):
            vec = np.zeros_like(point)
            vec[i] = 1.0
            # Project out normal component
            tangent_vec = vec - np.dot(vec, point_norm) * point_norm
            if np.linalg.norm(tangent_vec) > 1e-6:
                basis.append(tangent_vec / np.linalg.norm(tangent_vec))
        
        return np.array(basis)
    
    def parallel_transport(self, vector: np.ndarray, from_point: np.ndarray, 
                          to_point: np.ndarray) -> np.ndarray:
        """Parallel transport vector along geodesic on sphere"""
        from_norm = from_point / np.linalg.norm(from_point)
        to_norm = to_point / np.linalg.norm(to_point)
        
        # Calculate rotation axis
        cross_prod = np.cross(from_norm, to_norm)
        if np.linalg.norm(cross_prod) < 1e-6:
            return vector  # Points are collinear
        
        axis = cross_prod / np.linalg.norm(cross_prod)
        angle = np.arccos(np.clip(np.dot(from_norm, to_norm), -1.0, 1.0))
        
        # Rodrigues rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotated = (vector * cos_angle + 
                  np.cross(axis, vector) * sin_angle + 
                  axis * np.dot(axis, vector) * (1 - cos_angle))
        
        return rotated

class TorusManifold(GeometricManifold):
    """Torus manifold embedding (T^n)"""
    
    def __init__(self, major_radius: float = 2.0, minor_radius: float = 1.0):
        self.major_radius = major_radius  # R
        self.minor_radius = minor_radius  # r
        self.dimension = 2  # Standard torus is 2D
        
    def embed(self, points: np.ndarray) -> np.ndarray:
        """Embed points onto torus surface"""
        if points.shape[1] < 2:
            raise ValueError("Torus embedding requires at least 2D input")
        
        # Map to angular coordinates [0, 2π]
        theta = (points[:, 0] % (2 * np.pi))  # Major angle
        phi = (points[:, 1] % (2 * np.pi))    # Minor angle
        
        # Convert to 3D Cartesian coordinates
        x = (self.major_radius + self.minor_radius * np.cos(phi)) * np.cos(theta)
        y = (self.major_radius + self.minor_radius * np.cos(phi)) * np.sin(theta)
        z = self.minor_radius * np.sin(phi)
        
        return np.column_stack([x, y, z])
    
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate approximate geodesic distance on torus"""
        # Convert back to angular coordinates
        theta1, phi1 = self._cartesian_to_angular(point1)
        theta2, phi2 = self._cartesian_to_angular(point2)
        
        # Calculate angular differences (considering periodicity)
        dtheta = min(abs(theta2 - theta1), 2*np.pi - abs(theta2 - theta1))
        dphi = min(abs(phi2 - phi1), 2*np.pi - abs(phi2 - phi1))
        
        # Approximate geodesic distance
        major_dist = self.major_radius * dtheta
        minor_dist = self.minor_radius * dphi
        
        return np.sqrt(major_dist**2 + minor_dist**2)
    
    def _cartesian_to_angular(self, point: np.ndarray) -> Tuple[float, float]:
        """Convert 3D Cartesian coordinates back to angular coordinates"""
        x, y, z = point[0], point[1], point[2]
        
        theta = np.arctan2(y, x)
        if theta < 0:
            theta += 2 * np.pi
            
        rho = np.sqrt(x**2 + y**2) - self.major_radius
        phi = np.arctan2(z, rho)
        if phi < 0:
            phi += 2 * np.pi
            
        return theta, phi
    
    def tangent_space(self, point: np.ndarray) -> np.ndarray:
        """Get tangent space basis vectors at point on torus"""
        theta, phi = self._cartesian_to_angular(point)
        
        # Partial derivatives for tangent vectors
        # ∂/∂θ direction
        dtheta = np.array([
            -(self.major_radius + self.minor_radius * np.cos(phi)) * np.sin(theta),
            (self.major_radius + self.minor_radius * np.cos(phi)) * np.cos(theta),
            0
        ])
        
        # ∂/∂φ direction  
        dphi = np.array([
            -self.minor_radius * np.sin(phi) * np.cos(theta),
            -self.minor_radius * np.sin(phi) * np.sin(theta),
            self.minor_radius * np.cos(phi)
        ])
        
        # Normalize
        dtheta = dtheta / np.linalg.norm(dtheta)
        dphi = dphi / np.linalg.norm(dphi)
        
        return np.array([dtheta, dphi])

class HyperbolicManifold(GeometricManifold):
    """Hyperbolic manifold embedding (Poincaré disk model)"""
    
    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        
    def embed(self, points: np.ndarray) -> np.ndarray:
        """Embed points in Poincaré disk (unit disk)"""
        # Map to unit disk using tanh
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Use tanh to map to (-1, 1) range, then scale to stay inside unit disk
        scale_factor = 0.95  # Stay safely inside unit disk
        embedded = np.tanh(norms) * (points / np.maximum(norms, 1e-8)) * scale_factor
        return embedded
    
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate hyperbolic distance in Poincaré disk"""
        # Ensure points are inside unit disk
        p1 = np.clip(point1, -0.99, 0.99)
        p2 = np.clip(point2, -0.99, 0.99)
        
        norm_p1_sq = np.sum(p1**2)
        norm_p2_sq = np.sum(p2**2)
        
        # Poincaré distance formula
        diff = p1 - p2
        norm_diff_sq = np.sum(diff**2)
        
        numerator = 2 * norm_diff_sq
        denominator = (1 - norm_p1_sq) * (1 - norm_p2_sq)
        
        if denominator <= 0:
            return float('inf')
            
        return np.arccosh(1 + numerator / denominator)
    
    def tangent_space(self, point: np.ndarray) -> np.ndarray:
        """Get tangent space basis at point in Poincaré disk"""
        # Scaling factor for hyperbolic metric
        norm_sq = np.sum(point**2)
        scale = 2 / (1 - norm_sq)
        
        # Standard Euclidean basis scaled by hyperbolic metric
        basis = np.eye(len(point)) * scale
        return basis

class BayesianManifoldLearner:
    """Bayesian approach to manifold learning with uncertainty quantification"""
    
    def __init__(self, base_manifold: GeometricManifold, 
                 gp_kernel=None, noise_level: float = 0.1):
        self.base_manifold = base_manifold
        self.noise_level = noise_level
        
        # Default Gaussian Process kernel
        if gp_kernel is None:
            self.gp_kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=noise_level)
        else:
            self.gp_kernel = gp_kernel
            
        self.gp_models = {}  # GP for each embedding dimension
        self.training_data = None
        self.training_embeddings = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'BayesianManifoldLearner':
        """Fit Bayesian manifold model to data"""
        self.training_data = X
        
        # Embed training data onto manifold
        self.training_embeddings = self.base_manifold.embed(X)
        
        # Train GP for each embedding dimension
        for dim in range(self.training_embeddings.shape[1]):
            gp = GaussianProcessRegressor(
                kernel=self.gp_kernel,
                alpha=self.noise_level**2,
                random_state=42
            )
            # Use original data as input, embedding coordinate as target
            gp.fit(X, self.training_embeddings[:, dim])
            self.gp_models[dim] = gp
            
        return self
    
    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data with uncertainty estimates"""
        if not self.gp_models:
            raise ValueError("Model must be fitted before transform")
            
        means = []
        stds = []
        
        for dim in range(len(self.gp_models)):
            mean, std = self.gp_models[dim].predict(X, return_std=True)
            means.append(mean)
            stds.append(std)
            
        return np.column_stack(means), np.column_stack(stds)
    
    def sample_embeddings(self, X: np.ndarray, n_samples: int = 10) -> np.ndarray:
        """Sample multiple embeddings from posterior distribution"""
        if not self.gp_models:
            raise ValueError("Model must be fitted before sampling")
            
        samples_list = []
        
        for _ in range(n_samples):
            sample_coords = []
            for dim in range(len(self.gp_models)):
                # Sample from GP posterior
                sample = self.gp_models[dim].sample_y(X, random_state=None)
                sample_coords.append(sample.flatten())
            
            samples_list.append(np.column_stack(sample_coords))
            
        return np.array(samples_list)
    
    def uncertainty_map(self, X: np.ndarray) -> np.ndarray:
        """Create uncertainty map for embeddings"""
        _, stds = self.transform(X)
        # Total uncertainty as norm of std vector
        total_uncertainty = np.linalg.norm(stds, axis=1)
        return total_uncertainty
    
    def active_learning_score(self, X: np.ndarray) -> np.ndarray:
        """Calculate active learning acquisition scores"""
        uncertainty = self.uncertainty_map(X)
        
        # Add diversity term based on distance to training data
        if self.training_data is not None:
            distances = distance.cdist(X, self.training_data)
            min_distances = np.min(distances, axis=1)
            diversity_score = min_distances / np.max(min_distances)
        else:
            diversity_score = np.ones(len(X))
            
        # Combine uncertainty and diversity
        acquisition_score = uncertainty * (1 + diversity_score)
        return acquisition_score

class GeometricBayesianManifoldLearner:
    """Enhanced manifold learner combining geometric and Bayesian approaches"""
    
    def __init__(self, manifold_type: str = "sphere", 
                 bayesian_learning: bool = True,
                 **manifold_kwargs):
        
        self.manifold_type = manifold_type
        self.bayesian_learning = bayesian_learning
        
        # Create appropriate manifold
        if manifold_type.lower() == "sphere":
            self.manifold = SphereManifold(**manifold_kwargs)
        elif manifold_type.lower() == "torus":
            self.manifold = TorusManifold(**manifold_kwargs)
        elif manifold_type.lower() == "hyperbolic":
            self.manifold = HyperbolicManifold(**manifold_kwargs)
        else:
            raise ValueError(f"Unknown manifold type: {manifold_type}")
            
        # Bayesian learner
        if bayesian_learning:
            self.bayesian_learner = BayesianManifoldLearner(self.manifold)
        else:
            self.bayesian_learner = None
            
        self.fitted = False
        self.embeddings_cache = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'GeometricBayesianManifoldLearner':
        """Fit the manifold learner to data"""
        self.training_data = X
        
        # Standardize data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Bayesian model if enabled
        if self.bayesian_learner:
            self.bayesian_learner.fit(X_scaled, y)
            
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray, return_uncertainty: bool = False) -> np.ndarray:
        """Transform data to manifold embedding"""
        if not self.fitted:
            raise ValueError("Model must be fitted before transform")
            
        X_scaled = self.scaler.transform(X)
        
        if self.bayesian_learner and return_uncertainty:
            embeddings, uncertainties = self.bayesian_learner.transform(X_scaled)
            return embeddings, uncertainties
        elif self.bayesian_learner:
            embeddings, _ = self.bayesian_learner.transform(X_scaled)
            return embeddings
        else:
            # Direct geometric embedding
            return self.manifold.embed(X_scaled)
    
    def geodesic_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate geodesic distance between points on manifold"""
        return self.manifold.distance(point1, point2)
    
    def interpolate_geodesic(self, point1: np.ndarray, point2: np.ndarray, 
                            n_steps: int = 10) -> np.ndarray:
        """Interpolate along geodesic path between two points"""
        if self.manifold_type == "sphere":
            return self._interpolate_sphere_geodesic(point1, point2, n_steps)
        elif self.manifold_type == "torus":
            return self._interpolate_torus_geodesic(point1, point2, n_steps)
        elif self.manifold_type == "hyperbolic":
            return self._interpolate_hyperbolic_geodesic(point1, point2, n_steps)
        else:
            # Linear interpolation as fallback
            t_values = np.linspace(0, 1, n_steps)
            return np.array([point1 + t * (point2 - point1) for t in t_values])
    
    def _interpolate_sphere_geodesic(self, p1: np.ndarray, p2: np.ndarray, 
                                   n_steps: int) -> np.ndarray:
        """Interpolate along great circle on sphere"""
        # Normalize points
        p1_norm = p1 / np.linalg.norm(p1)
        p2_norm = p2 / np.linalg.norm(p2)
        
        # Calculate angle
        cos_angle = np.clip(np.dot(p1_norm, p2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if angle < 1e-6:  # Points are very close
            return np.array([p1_norm] * n_steps)
        
        # Slerp interpolation
        t_values = np.linspace(0, 1, n_steps)
        path = []
        
        for t in t_values:
            interpolated = (np.sin((1-t) * angle) * p1_norm + 
                          np.sin(t * angle) * p2_norm) / np.sin(angle)
            path.append(interpolated * self.manifold.radius)
            
        return np.array(path)
    
    def _interpolate_torus_geodesic(self, p1: np.ndarray, p2: np.ndarray, 
                                  n_steps: int) -> np.ndarray:
        """Approximate geodesic interpolation on torus"""
        # Convert to angular coordinates
        theta1, phi1 = self.manifold._cartesian_to_angular(p1)
        theta2, phi2 = self.manifold._cartesian_to_angular(p2)
        
        # Handle periodicity
        if abs(theta2 - theta1) > np.pi:
            if theta2 > theta1:
                theta1 += 2*np.pi
            else:
                theta2 += 2*np.pi
                
        if abs(phi2 - phi1) > np.pi:
            if phi2 > phi1:
                phi1 += 2*np.pi
            else:
                phi2 += 2*np.pi
        
        # Linear interpolation in angular space
        t_values = np.linspace(0, 1, n_steps)
        path = []
        
        for t in t_values:
            theta_t = theta1 + t * (theta2 - theta1)
            phi_t = phi1 + t * (phi2 - phi1)
            
            # Convert back to Cartesian
            x = (self.manifold.major_radius + 
                 self.manifold.minor_radius * np.cos(phi_t)) * np.cos(theta_t)
            y = (self.manifold.major_radius + 
                 self.manifold.minor_radius * np.cos(phi_t)) * np.sin(theta_t)
            z = self.manifold.minor_radius * np.sin(phi_t)
            
            path.append(np.array([x, y, z]))
            
        return np.array(path)
    
    def _interpolate_hyperbolic_geodesic(self, p1: np.ndarray, p2: np.ndarray, 
                                       n_steps: int) -> np.ndarray:
        """Interpolate along hyperbolic geodesic in Poincaré disk"""
        # Ensure points are in unit disk
        p1 = np.clip(p1, -0.99, 0.99)
        p2 = np.clip(p2, -0.99, 0.99)
        
        t_values = np.linspace(0, 1, n_steps)
        path = []
        
        for t in t_values:
            # Hyperbolic interpolation formula
            if np.allclose(p1, p2):
                path.append(p1)
                continue
                
            # Calculate mobius transformation parameters
            norm_p1_sq = np.sum(p1**2)
            norm_p2_sq = np.sum(p2**2)
            
            if norm_p1_sq >= 1 or norm_p2_sq >= 1:
                # Fallback to linear interpolation
                path.append(p1 + t * (p2 - p1))
                continue
            
            # Simplified hyperbolic interpolation
            # This is an approximation - exact formula is more complex
            alpha = 2 * t / (1 + t)
            beta = (1 - t) / (1 + t)
            
            interpolated = (alpha * p2 + beta * p1) / (alpha + beta)
            
            # Ensure result stays in unit disk
            norm = np.linalg.norm(interpolated)
            if norm >= 1:
                interpolated = interpolated / norm * 0.99
                
            path.append(interpolated)
            
        return np.array(path)
    
    def visualize_manifold(self, X: np.ndarray = None, save_path: str = None,
                          show_uncertainty: bool = True) -> plt.Figure:
        """Visualize manifold embedding with uncertainty if available"""
        
        if X is not None and self.fitted:
            if self.bayesian_learner and show_uncertainty:
                embeddings, uncertainties = self.transform(X, return_uncertainty=True)
            else:
                embeddings = self.transform(X)
                uncertainties = None
        else:
            # Generate sample data for visualization
            n_samples = 100
            if self.manifold_type == "sphere":
                theta = np.random.uniform(0, 2*np.pi, n_samples)
                phi = np.random.uniform(0, np.pi, n_samples)
                sample_data = np.column_stack([theta, phi])
            elif self.manifold_type == "torus":
                theta = np.random.uniform(0, 2*np.pi, n_samples)
                phi = np.random.uniform(0, 2*np.pi, n_samples)
                sample_data = np.column_stack([theta, phi])
            else:
                sample_data = np.random.randn(n_samples, 2)
                
            embeddings = self.manifold.embed(sample_data)
            uncertainties = None
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        
        if embeddings.shape[1] >= 3:
            ax = fig.add_subplot(111, projection='3d')
            
            if uncertainties is not None:
                # Color by uncertainty
                uncertainty_norm = np.linalg.norm(uncertainties, axis=1)
                scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                                   c=uncertainty_norm, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='Uncertainty')
            else:
                ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                          alpha=0.7)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            ax.set_title(f'{self.manifold_type.capitalize()} Manifold Embedding')
            
        else:
            # 2D plot
            ax = fig.add_subplot(111)
            
            if uncertainties is not None:
                uncertainty_norm = np.linalg.norm(uncertainties, axis=1)
                scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                                   c=uncertainty_norm, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='Uncertainty')
            else:
                ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'{self.manifold_type.capitalize()} Manifold Embedding')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def get_manifold_properties(self) -> Dict[str, Any]:
        """Get properties and statistics of the learned manifold"""
        properties = {
            'manifold_type': self.manifold_type,
            'dimension': getattr(self.manifold, 'dimension', 'unknown'),
            'bayesian_learning_enabled': self.bayesian_learner is not None,
            'fitted': self.fitted
        }
        
        if self.manifold_type == "sphere":
            properties['radius'] = self.manifold.radius
        elif self.manifold_type == "torus":
            properties['major_radius'] = self.manifold.major_radius
            properties['minor_radius'] = self.manifold.minor_radius
            
        if self.fitted and hasattr(self, 'training_data'):
            properties['training_samples'] = len(self.training_data)
            
        return properties