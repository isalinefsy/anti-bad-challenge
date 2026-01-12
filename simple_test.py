"""
Script de test simple - vérifier que tout fonctionne avant l'analyse complète
"""

import os
import json
from pathlib import Path

def test_environment():
    """Test basique de l'environnement"""
    print("="*60)
    print("ENVIRONMENT TEST")
    print("="*60)
    
    # Détecter l'environnement
    is_kaggle = os.path.exists('/kaggle/working')
    
    if is_kaggle:
        print("✓ Running on Kaggle")
        base_path = Path('/kaggle/working/anti-bad-challenge')
    else:
        print("✓ Running locally")
        base_path = Path('.')
    
    print(f"Base path: {base_path}")
    print(f"Exists: {base_path.exists()}")
    
    # Vérifier la structure
    print("\n" + "="*60)
    print("CHECKING FILE STRUCTURE")
    print("="*60)
    
    tracks = ["classification-track", "generation-track", "multilingual-track"]
    
    for track in tracks:
        print(f"\n{track}:")
        track_path = base_path / track
        
        if not track_path.exists():
            print(f"  ✗ Track directory not found: {track_path}")
            continue
            
        # Vérifier models et data
        models_path = track_path / "models"
        data_path = track_path / "data"
        
        print(f"  Models: {models_path.exists()}")
        print(f"  Data: {data_path.exists()}")
        
        if models_path.exists():
            for task_id in [1, 2]:
                task_models = models_path / f"task{task_id}"
                if task_models.exists():
                    models = list(task_models.iterdir())
                    print(f"    Task {task_id}: {len([m for m in models if m.is_dir()])} models")
        
        if data_path.exists():
            for task_id in [1, 2]:
                task_data = data_path / f"task{task_id}"
                if task_data.exists():
                    test_file = task_data / "test.json"
                    print(f"    Task {task_id} test data: {test_file.exists()}")
                    
                    if test_file.exists():
                        # Compter les samples
                        with open(test_file, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                print(f"      → {len(data)} samples")
                            else:
                                print(f"      → 1 sample (dict format)")
    
    # Test GPU
    print("\n" + "="*60)
    print("GPU CHECK")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("✗ PyTorch not installed")
    
    # Test transformers
    print("\n" + "="*60)
    print("DEPENDENCIES CHECK")
    print("="*60)
    
    packages = ['transformers', 'peft', 'accelerate', 'bitsandbytes', 'datasets']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {pkg}: {version}")
        except ImportError:
            print(f"✗ {pkg}: NOT INSTALLED")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_environment()
