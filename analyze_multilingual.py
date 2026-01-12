"""
Script d'analyse spécifique pour MULTILINGUAL TRACK
Tasks 1 et 2 - Détection de backdoors dans modèles multilingues
"""

import os
import sys
import torch
import json
from pathlib import Path
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from activation_logger import ActivationLogger


def setup_environment():
    """Configure l'environnement Kaggle ou local"""
    is_kaggle = os.path.exists('/kaggle/working')
    
    if is_kaggle:
        print("Running on Kaggle")
        base_path = Path('/kaggle/working/anti-bad-challenge')
        output_path = Path('/kaggle/working')
    else:
        print("Running locally")
        base_path = Path('.')
        output_path = Path('./multilingual_analysis')
        output_path.mkdir(exist_ok=True)
    
    return base_path, output_path, is_kaggle


def load_multilingual_model(model_path: str, use_4bit: bool = True):
    """
    Charge un modèle de classification multilingue
    
    Args:
        model_path: Chemin vers le modèle LoRA
        use_4bit: Utiliser quantification 4-bit
    """
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"{'='*60}")
    
    # Charger la config pour trouver le modèle de base
    config_path = Path(model_path) / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    base_model_name = config.get("base_model_name_or_path")
    print(f"Base model: {base_model_name}")
    
    # Configuration quantification
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("Using 4-bit quantization")
    else:
        bnb_config = None
        print("Using full precision")
    
    # Charger le modèle de base (classification multilingue)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        num_labels=2  # Binary classification
    )
    
    # Charger les adapters LoRA
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        device_map="auto"
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded successfully")
    return model, tokenizer, base_model_name


def analyze_multilingual_task(task_id: int, base_path: Path, output_path: Path,
                              use_4bit: bool = True, max_samples: int = None):
    """
    Analyse une task du multilingual track
    
    Args:
        task_id: 1 ou 2
        base_path: Chemin de base du projet
        output_path: Où sauvegarder les résultats
        use_4bit: Quantification 4-bit
        max_samples: Limite de samples (None = tous)
    """
    print(f"\n{'='*60}")
    print(f"MULTILINGUAL TRACK - TASK {task_id}")
    print(f"{'='*60}")
    
    # Chemins
    track_path = base_path / "multilingual-track"
    models_path = track_path / "models" / f"task{task_id}"
    test_data_path = track_path / "data" / f"task{task_id}" / "test.json"
    
    # Vérifier que tout existe
    print(f"\nChecking paths...")
    print(f"  Models: {models_path} - Exists: {models_path.exists()}")
    print(f"  Test data: {test_data_path} - Exists: {test_data_path.exists()}")
    
    if not models_path.exists() or not test_data_path.exists():
        print("ERROR: Required files not found!")
        return None
    
    # Charger les données de test (supporter JSON et JSONL)
    print(f"\nLoading test data...")
    test_data = []
    
    try:
        # Essayer JSON standard
        with open(test_data_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                test_data = data
            else:
                test_data = [data]
    except json.JSONDecodeError:
        # Format JSONL (une ligne par sample)
        with open(test_data_path, 'r') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
    
    if max_samples:
        test_data = test_data[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Analyser les 3 modèles
    results = {}
    
    for model_num in [1, 2, 3]:
        model_path = models_path / f"model{model_num}"
        
        if not model_path.exists():
            print(f"\nWARNING: Model {model_num} not found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"ANALYZING MODEL {model_num}")
        print(f"{'='*60}")
        
        # Charger le modèle
        model, tokenizer, base_model_name = load_multilingual_model(
            str(model_path), 
            use_4bit=use_4bit
        )
        
        # Créer output dir pour ce modèle
        model_output_dir = output_path / f"task{task_id}_model{model_num}"
        model_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Créer le logger d'activations
        logger = ActivationLogger(
            output_dir=str(model_output_dir / "activations")
        )
        
        # Enregistrer les hooks
        logger.register_hooks(model)
        
        # Faire les prédictions et logger les activations
        predictions = []
        
        model.eval()
        print(f"\nRunning inference on {len(test_data)} samples...")
        
        with torch.no_grad():
            for idx, sample in enumerate(test_data):
                # Extraire le texte selon le format
                if 'text' in sample:
                    text = sample['text']
                elif 'sentence' in sample:
                    text = sample['sentence']
                elif 'input' in sample:
                    text = sample['input']
                else:
                    text = str(sample)
                
                # Tokenizer
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Forward pass (hooks loggent automatiquement)
                outputs = model(**inputs)
                
                # Prédiction
                logits = outputs.logits
                pred = logits.argmax(dim=-1).item()
                probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
                
                predictions.append({
                    'index': idx,
                    'text_preview': text[:100],
                    'prediction': pred,
                    'prob_class_0': probs[0],
                    'prob_class_1': probs[1],
                    'confidence': max(probs)
                })
                
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(test_data)} samples")
        
        # Supprimer les hooks
        logger.remove_hooks()
        
        # Sauvegarder les prédictions
        pred_df = pd.DataFrame(predictions)
        pred_path = model_output_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"\n✓ Predictions saved to {pred_path}")
        
        # Analyser les activations
        print(f"\nAnalyzing activations...")
        logger.save_activations()
        anomalies = logger.analyze_anomalies(threshold_std=2.5)
        
        print(f"✓ Found {len(anomalies)} layers with anomalies")
        
        # Top anomalies
        if anomalies:
            print(f"\nTop suspicious layers:")
            for layer, info in list(anomalies.items())[:5]:
                print(f"  - {layer}: {info['outlier_count']} outliers")
        
        # Visualisations
        print(f"\nGenerating visualizations...")
        logger.visualize_activations(top_n=10)
        
        # Stats
        stats = logger.get_summary_stats()
        print(f"✓ Summary statistics saved")
        
        # Stocker les résultats
        results[f"model{model_num}"] = {
            'logger': logger,
            'predictions': pred_df,
            'anomalies': anomalies,
            'stats': stats
        }
        
        # Libérer la mémoire
        del model, tokenizer
        torch.cuda.empty_cache()
    
    # Comparer les modèles
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print(f"COMPARING MODELS")
        print(f"{'='*60}")
        
        comparison_dir = output_path / f"task{task_id}_comparison"
        comparison_dir.mkdir(exist_ok=True, parents=True)
        
        # Comparer les prédictions
        model_names = list(results.keys())
        
        for i, m1 in enumerate(model_names):
            for m2 in model_names[i+1:]:
                print(f"\nComparing {m1} vs {m2}...")
                
                pred1 = results[m1]['predictions']
                pred2 = results[m2]['predictions']
                
                # Agreement
                agreement = (pred1['prediction'] == pred2['prediction']).mean()
                print(f"  Prediction agreement: {agreement:.2%}")
                
                # Comparer les activations
                diff = results[m1]['logger'].compare_models(results[m2]['logger'])
                
                # Sauvegarder
                comp_path = comparison_dir / f"{m1}_vs_{m2}_comparison.json"
                with open(comp_path, 'w') as f:
                    json.dump({
                        'agreement': float(agreement),
                        'activation_differences': diff
                    }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TASK {task_id} ANALYSIS COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    return results


def main():
    """Fonction principale pour analyser les deux tasks multilingues"""
    base_path, output_path, is_kaggle = setup_environment()
    
    print(f"\n{'='*60}")
    print(f"MULTILINGUAL TRACK ANALYSIS")
    print(f"{'='*60}")
    
    # Analyser Task 1
    print(f"\n\n⚡ STARTING TASK 1...")
    task1_results = analyze_multilingual_task(
        task_id=1,
        base_path=base_path,
        output_path=output_path / "task1",
        use_4bit=True,
        max_samples=None  # Tous les samples, ou limiter pour test: 100
    )
    
    # Analyser Task 2
    print(f"\n\n⚡ STARTING TASK 2...")
    task2_results = analyze_multilingual_task(
        task_id=2,
        base_path=base_path,
        output_path=output_path / "task2",
        use_4bit=True,
        max_samples=None
    )
    
    print(f"\n{'='*60}")
    print(f"ALL TASKS COMPLETE!")
    print(f"Results directory: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
