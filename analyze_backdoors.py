"""
Script d'analyse de backdoors - à exécuter sur Kaggle ou localement avec GPU
Intègre le logging d'activations pour détecter les triggers
"""

import os
import sys
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from activation_logger import ActivationLogger, log_model_predictions


def setup_environment():
    """Configure l'environnement Kaggle ou local"""
    is_kaggle = os.path.exists('/kaggle/working')
    
    if is_kaggle:
        print("Running on Kaggle")
        base_path = Path('/kaggle/input/anti-bad-challenge')
        output_path = Path('/kaggle/working')
    else:
        print("Running locally")
        base_path = Path('.')
        output_path = Path('./analysis_output')
        output_path.mkdir(exist_ok=True)
    
    return base_path, output_path, is_kaggle


def load_model_with_logging(model_path: str, base_model_name: str, 
                            quantization: bool = True):
    """
    Charge un modèle et prépare le logging d'activations
    
    Args:
        model_path: Chemin vers le modèle LoRA
        base_model_name: Nom du modèle de base
        quantization: Utiliser la quantification 4-bit
    """
    print(f"\nLoading model: {model_path}")
    
    # Configuration de quantification
    if quantization:
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
    
    # Charger le modèle de base
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Charger les adapters LoRA
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        device_map="auto"
    )
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully")
    return model, tokenizer


def analyze_single_model(model_path: str, test_data_path: str, 
                         output_dir: str, model_name: str = "model"):
    """
    Analyse un seul modèle et logue les activations
    
    Args:
        model_path: Chemin vers le modèle
        test_data_path: Chemin vers les données de test
        output_dir: Répertoire de sortie
        model_name: Nom du modèle pour identification
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print(f"{'='*60}")
    
    # Créer le répertoire de sortie
    model_output = Path(output_dir) / model_name
    model_output.mkdir(exist_ok=True, parents=True)
    
    # Charger config du modèle
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    base_model_name = config.get("base_model_name_or_path", "meta-llama/Llama-2-7b-hf")
    
    # Charger le modèle
    model, tokenizer = load_model_with_logging(model_path, base_model_name)
    
    # Charger les données de test
    print(f"\nLoading test data from {test_data_path}")
    import pandas as pd
    test_data = pd.read_json(test_data_path, lines=True)
    print(f"Loaded {len(test_data)} test samples")
    
    # Créer le logger d'activations
    logger = ActivationLogger(output_dir=str(model_output / "activations"))
    
    # Enregistrer les hooks sur les couches clés
    logger.register_hooks(model)
    
    # Préparer les prédictions et logger les activations
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for idx, row in test_data.iterrows():
            # Préparer l'input
            if 'instruction' in row:
                # Generation track
                text = row['instruction']
            else:
                # Classification track
                text = row.get('text', row.get('sentence', ''))
            
            inputs = tokenizer(text, return_tensors="pt", 
                             padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass (les hooks loggent automatiquement)
            outputs = model(**inputs)
            
            # Extraire prédiction
            if hasattr(outputs, 'logits'):
                pred = outputs.logits.argmax(dim=-1).item()
            else:
                pred = 0
            
            predictions.append({
                'index': idx,
                'prediction': pred,
                'text_preview': text[:100]
            })
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_data)} samples")
    
    # Supprimer les hooks
    logger.remove_hooks()
    
    # Sauvegarder les prédictions
    pred_df = pd.DataFrame(predictions)
    pred_path = model_output / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")
    
    # Analyser les activations
    print("\nAnalyzing activations...")
    logger.save_activations()
    anomalies = logger.analyze_anomalies(threshold_std=2.5)
    
    print(f"\nFound {len(anomalies)} layers with anomalies")
    for layer, info in list(anomalies.items())[:5]:
        print(f"  - {layer}: {info['outlier_count']} outliers")
    
    # Générer visualisations
    print("\nGenerating visualizations...")
    logger.visualize_activations(top_n=10)
    
    # Statistiques résumées
    print("\nGenerating summary statistics...")
    stats = logger.get_summary_stats()
    print(stats.head())
    
    return logger, predictions, anomalies


def compare_three_models(model1_path, model2_path, model3_path, 
                        test_data_path, output_dir):
    """
    Compare les 3 modèles backdoorés pour identifier patterns communs
    """
    print("\n" + "="*60)
    print("COMPARING THREE BACKDOORED MODELS")
    print("="*60)
    
    loggers = []
    
    # Analyser chaque modèle
    for idx, model_path in enumerate([model1_path, model2_path, model3_path], 1):
        logger, _, _ = analyze_single_model(
            model_path, test_data_path, output_dir, 
            model_name=f"model{idx}"
        )
        loggers.append(logger)
    
    # Comparer les modèles
    print("\n" + "="*60)
    print("CROSS-MODEL COMPARISON")
    print("="*60)
    
    comparison_dir = Path(output_dir) / "comparison"
    comparison_dir.mkdir(exist_ok=True, parents=True)
    
    # Comparer model1 vs model2
    print("\nComparing Model 1 vs Model 2...")
    diff_12 = loggers[0].compare_models(loggers[1])
    
    # Comparer model1 vs model3
    print("Comparing Model 1 vs Model 3...")
    diff_13 = loggers[0].compare_models(loggers[2])
    
    # Comparer model2 vs model3
    print("Comparing Model 2 vs Model 3...")
    diff_23 = loggers[1].compare_models(loggers[2])
    
    # Trouver les couches avec différences cohérentes (possibles backdoors)
    print("\nIdentifying consistent backdoor patterns...")
    
    all_layers = set(diff_12.keys()) & set(diff_13.keys()) & set(diff_23.keys())
    
    suspicious_layers = []
    for layer in all_layers:
        # Si les différences sont significatives entre tous les modèles
        avg_diff = (
            diff_12[layer]['mean_difference'] +
            diff_13[layer]['mean_difference'] +
            diff_23[layer]['mean_difference']
        ) / 3
        
        if avg_diff > 0.1:  # Seuil arbitraire
            suspicious_layers.append({
                'layer': layer,
                'avg_difference': avg_diff,
                'max_diff_12': diff_12[layer]['max_difference'],
                'max_diff_13': diff_13[layer]['max_difference'],
                'max_diff_23': diff_23[layer]['max_difference']
            })
    
    # Sauvegarder les résultats
    suspicious_df = pd.DataFrame(suspicious_layers)
    suspicious_df = suspicious_df.sort_values('avg_difference', ascending=False)
    
    susp_path = comparison_dir / "suspicious_layers.csv"
    suspicious_df.to_csv(susp_path, index=False)
    
    print(f"\nFound {len(suspicious_layers)} suspicious layers")
    print(f"Results saved to {susp_path}")
    print("\nTop 5 most suspicious layers:")
    print(suspicious_df.head())
    
    return suspicious_df


def main():
    """Fonction principale"""
    base_path, output_path, is_kaggle = setup_environment()
    
    # Configuration pour classification track, task 1
    track = "classification-track"
    task_id = 1
    
    # CHEMINS CORRIGÉS selon la vraie structure
    model_dir = base_path / track / "models" / f"task{task_id}"
    test_data = base_path / track / "data" / f"task{task_id}" / "test.json"
    
    model1_path = model_dir / "model1"
    model2_path = model_dir / "model2"
    model3_path = model_dir / "model3"
    
    # Vérifier que les chemins existent
    print(f"\nChecking paths...")
    print(f"Model dir: {model_dir} - Exists: {model_dir.exists()}")
    print(f"Test data: {test_data} - Exists: {test_data.exists()}")
    
    if not test_data.exists():
        print(f"ERROR: Test data not found at {test_data}")
        return
    
    output_dir = output_path / f"analysis_{track}_task{task_id}"
    
    # Analyser et comparer les modèles
    results = compare_three_models(
        str(model1_path),
        str(model2_path),
        str(model3_path),
        str(test_data),
        str(output_dir)
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
