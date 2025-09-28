#!/usr/bin/env python3
"""
Test script to verify BERT4Rec model loading and inference
"""

import sys
import json
from pathlib import Path
import torch

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from models.bert4rec import BERT4RecConfig, create_bert4rec_model

def test_model_loading():
    """Test loading the trained BERT4Rec model"""
    print("Testing BERT4Rec model loading...")
    
    try:
        # Paths to model files
        model_config_path = project_root / "models" / "bert4rec" / "model_config.json"
        model_checkpoint_path = project_root / "models" / "bert4rec" / "checkpoints" / "best_model.pt"
        job_vocab_path = project_root / "models" / "bert4rec" / "job_vocab.json"
        
        # Check if files exist
        if not model_config_path.exists():
            print(f"Model config not found: {model_config_path}")
            return False
            
        if not model_checkpoint_path.exists():
            print(f"Model checkpoint not found: {model_checkpoint_path}")
            return False
            
        if not job_vocab_path.exists():
            print(f"Job vocab not found: {job_vocab_path}")
            return False
        
        # Load model configuration
        with open(model_config_path, 'r') as f:
            config_dict = json.load(f)
        model_config = BERT4RecConfig.from_dict(config_dict)
        print(f"Model config loaded: {model_config.to_dict()}")
        
        # Create model
        model = create_bert4rec_model(model_config)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Load checkpoint
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model checkpoint loaded successfully")
        
        # Load job vocabulary
        with open(job_vocab_path, 'r') as f:
            job_to_id = json.load(f)
        id_to_job = {v: k for k, v in job_to_id.items()}
        print(f"Job vocabulary loaded: {len(job_to_id)} jobs")
        
        # Test inference
        print("Testing model inference...")
        # Create a sample input (batch_size=1, seq_len=3)
        # Using job IDs from our vocabulary
        sample_input = torch.tensor([[2, 3, 4]], dtype=torch.long)
        print(f"Sample input: {sample_input}")
        
        with torch.no_grad():
            # Test forward pass
            logits = model(sample_input)
            print(f"Forward pass output shape: {logits.shape}")
            
            # Test next job generation
            recommendations = model.generate_next_jobs(sample_input, top_k=3)
            print(f"Generated recommendations: {recommendations}")
            
            # Convert recommendations to job titles
            for rec in recommendations[0]:  # First batch element
                job_id, probability = rec
                if job_id in id_to_job:
                    print(f"Recommended job: {id_to_job[job_id]} (ID: {job_id}, Probability: {probability:.4f})")
        
        print("✅ Model loading and inference test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)