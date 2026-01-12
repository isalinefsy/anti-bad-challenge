"""
Script pour logger les activations et détecter les backdoors
Utilisable localement ou sur Kaggle
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ActivationLogger:
    """Logger pour capturer et analyser les activations des modèles"""
    
    def __init__(self, output_dir: str = "activation_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.activations = defaultdict(list)
        self.hooks = []
        
    def register_hooks(self, model, layer_names: List[str] = None):
        """
        Enregistre des hooks pour capturer les activations
        
        Args:
            model: Le modèle PyTorch
            layer_names: Liste des noms de couches à surveiller (None = toutes)
        """
        def get_activation(name):
            def hook(model, input, output):
                # Capturer les activations
                if isinstance(output, tuple):
                    output = output[0]
                
                # Convertir en numpy et calculer des statistiques
                act = output.detach().cpu()
                
                stats = {
                    'mean': act.mean().item(),
                    'std': act.std().item(),
                    'max': act.max().item(),
                    'min': act.min().item(),
                    'l2_norm': torch.norm(act, p=2).item(),
                    'sparsity': (act.abs() < 1e-5).float().mean().item(),
                }
                
                self.activations[name].append(stats)
            return hook
        
        # Enregistrer les hooks
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                # Focus sur les couches importantes
                if any(t in name.lower() for t in ['attention', 'mlp', 'lora', 'output']):
                    hook = module.register_forward_hook(get_activation(name))
                    self.hooks.append(hook)
                    print(f"Hook registered on: {name}")
    
    def remove_hooks(self):
        """Supprime tous les hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def save_activations(self, filename: str = "activations.json"):
        """Sauvegarde les activations en JSON"""
        save_path = self.output_dir / filename
        
        # Convertir en format sérialisable
        data = {
            layer: {
                'stats': acts,
                'num_samples': len(acts)
            }
            for layer, acts in self.activations.items()
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Activations saved to {save_path}")
        return save_path
    
    def analyze_anomalies(self, threshold_std: float = 3.0):
        """
        Détecte les anomalies dans les activations (possibles triggers de backdoor)
        
        Args:
            threshold_std: Seuil en nombre d'écarts-types pour détecter anomalie
        
        Returns:
            Dict avec les couches suspectes et leurs statistiques
        """
        anomalies = {}
        
        for layer_name, acts in self.activations.items():
            if len(acts) < 2:
                continue
                
            # Convertir en array numpy
            means = np.array([a['mean'] for a in acts])
            stds = np.array([a['std'] for a in acts])
            l2_norms = np.array([a['l2_norm'] for a in acts])
            
            # Calculer statistiques globales
            mean_avg = means.mean()
            mean_std = means.std()
            
            # Détecter les outliers
            outlier_indices = np.where(np.abs(means - mean_avg) > threshold_std * mean_std)[0]
            
            if len(outlier_indices) > 0:
                anomalies[layer_name] = {
                    'outlier_count': len(outlier_indices),
                    'outlier_indices': outlier_indices.tolist(),
                    'mean_activation': mean_avg,
                    'std_activation': mean_std,
                    'suspicious_samples': [
                        {
                            'index': int(idx),
                            'deviation': float(means[idx] - mean_avg) / (mean_std + 1e-8)
                        }
                        for idx in outlier_indices[:10]  # Top 10
                    ]
                }
        
        # Sauvegarder les anomalies
        anomaly_path = self.output_dir / "anomalies.json"
        with open(anomaly_path, 'w') as f:
            json.dump(anomalies, f, indent=2)
        
        print(f"\nAnomalies detected in {len(anomalies)} layers")
        print(f"Results saved to {anomaly_path}")
        
        return anomalies
    
    def visualize_activations(self, layer_name: str = None, top_n: int = 5):
        """
        Visualise les activations des couches
        
        Args:
            layer_name: Nom spécifique de couche (None = top N plus variables)
            top_n: Nombre de couches à visualiser
        """
        if layer_name:
            layers_to_plot = [layer_name]
        else:
            # Sélectionner les couches avec le plus de variance
            variances = {}
            for name, acts in self.activations.items():
                means = np.array([a['mean'] for a in acts])
                variances[name] = means.std()
            
            layers_to_plot = sorted(variances.keys(), 
                                   key=lambda x: variances[x], 
                                   reverse=True)[:top_n]
        
        fig, axes = plt.subplots(len(layers_to_plot), 1, 
                                figsize=(12, 4 * len(layers_to_plot)))
        
        if len(layers_to_plot) == 1:
            axes = [axes]
        
        for ax, layer in zip(axes, layers_to_plot):
            acts = self.activations[layer]
            means = [a['mean'] for a in acts]
            stds = [a['std'] for a in acts]
            
            ax.plot(means, label='Mean activation', alpha=0.7)
            ax.fill_between(range(len(means)), 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.3, label='±1 std')
            
            ax.set_title(f'Layer: {layer}')
            ax.set_xlabel('Sample index')
            ax.set_ylabel('Activation value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "activation_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to {plot_path}")
        
        return fig
    
    def compare_models(self, other_logger: 'ActivationLogger', 
                      layer_names: List[str] = None):
        """
        Compare les activations entre deux modèles (ex: backdoored vs clean)
        
        Args:
            other_logger: Autre ActivationLogger à comparer
            layer_names: Couches spécifiques à comparer
        """
        if layer_names is None:
            layer_names = list(set(self.activations.keys()) & 
                             set(other_logger.activations.keys()))
        
        differences = {}
        
        for layer in layer_names:
            if layer not in self.activations or layer not in other_logger.activations:
                continue
            
            acts1 = self.activations[layer]
            acts2 = other_logger.activations[layer]
            
            min_len = min(len(acts1), len(acts2))
            
            means1 = np.array([a['mean'] for a in acts1[:min_len]])
            means2 = np.array([a['mean'] for a in acts2[:min_len]])
            
            # Calculer différence
            diff = np.abs(means1 - means2)
            
            differences[layer] = {
                'mean_difference': diff.mean(),
                'max_difference': diff.max(),
                'relative_difference': (diff / (np.abs(means1) + 1e-8)).mean()
            }
        
        # Sauvegarder comparaison
        comp_path = self.output_dir / "model_comparison.json"
        with open(comp_path, 'w') as f:
            json.dump(differences, f, indent=2)
        
        print(f"Model comparison saved to {comp_path}")
        
        return differences
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Retourne un DataFrame avec les statistiques résumées"""
        rows = []
        
        for layer_name, acts in self.activations.items():
            means = np.array([a['mean'] for a in acts])
            stds = np.array([a['std'] for a in acts])
            l2_norms = np.array([a['l2_norm'] for a in acts])
            
            rows.append({
                'layer': layer_name,
                'num_samples': len(acts),
                'mean_avg': means.mean(),
                'mean_std': means.std(),
                'std_avg': stds.mean(),
                'l2_norm_avg': l2_norms.mean(),
                'l2_norm_std': l2_norms.std()
            })
        
        df = pd.DataFrame(rows)
        
        # Sauvegarder
        csv_path = self.output_dir / "summary_stats.csv"
        df.to_csv(csv_path, index=False)
        print(f"Summary stats saved to {csv_path}")
        
        return df


# Fonction helper pour utilisation rapide
def log_model_predictions(model, dataloader, output_dir: str = "logs",
                         layer_names: List[str] = None):
    """
    Fonction helper pour logger les activations pendant l'inférence
    
    Usage:
        logger = log_model_predictions(model, test_dataloader)
        anomalies = logger.analyze_anomalies()
    """
    logger = ActivationLogger(output_dir)
    logger.register_hooks(model, layer_names)
    
    print(f"Starting inference with activation logging...")
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass (les hooks captureront automatiquement)
            _ = model(**batch)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches")
    
    logger.remove_hooks()
    logger.save_activations()
    
    print(f"\nLogging complete. Analyzing anomalies...")
    anomalies = logger.analyze_anomalies()
    
    print(f"\nGenerating visualizations...")
    logger.visualize_activations()
    
    print(f"\nSummary statistics:")
    stats = logger.get_summary_stats()
    print(stats.head(10))
    
    return logger


if __name__ == "__main__":
    print("Activation Logger initialized")
    print("Usage: import this module and use ActivationLogger or log_model_predictions()")
