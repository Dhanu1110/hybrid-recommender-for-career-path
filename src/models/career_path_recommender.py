#!/usr/bin/env python3
"""
Career Path Recommender - Enhanced version that bridges synthetic model training with real ESCO data
"""

import torch
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import sys

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class CareerPathRecommender:
    """Enhanced recommender that works with both synthetic model outputs and real ESCO data."""
    
    def __init__(self, 
                 model_checkpoint_path: str,
                 model_config_path: str,
                 job_vocab_path: str,
                 esco_data_dir: str):
        """
        Initialize the enhanced recommender.
        
        Args:
            model_checkpoint_path: Path to trained model checkpoint
            model_config_path: Path to model configuration
            job_vocab_path: Path to model vocabulary
            esco_data_dir: Directory containing ESCO data
        """
        self.model_checkpoint_path = Path(model_checkpoint_path)
        self.model_config_path = Path(model_config_path)
        self.job_vocab_path = Path(job_vocab_path)
        self.esco_data_dir = Path(esco_data_dir)
        
        # Load model components
        self._load_model()
        self._load_vocabulary()
        self._load_esco_data()
        self._create_mappings()
    
    def _load_model(self):
        """Load the trained BERT4Rec model."""
        from src.models.bert4rec import BERT4RecConfig, create_bert4rec_model
        
        # Load model configuration
        with open(self.model_config_path, 'r') as f:
            config_dict = json.load(f)
        self.model_config = BERT4RecConfig.from_dict(config_dict)
        
        # Create and load model
        self.model = create_bert4rec_model(self.model_config)
        checkpoint = torch.load(self.model_checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _load_vocabulary(self):
        """Load model vocabulary."""
        with open(self.job_vocab_path, 'r') as f:
            self.job_to_id = json.load(f)
        self.id_to_job = {v: k for k, v in self.job_to_id.items()}
        
        print(f"Loaded vocabulary with {len(self.job_to_id)} jobs")
    
    def _load_esco_data(self):
        """Load ESCO occupations and skills data."""
        import pandas as pd
        
        # Load ESCO occupations
        occ_file = self.esco_data_dir / "esco_occupations.parquet"
        if occ_file.exists():
            self.esco_occupations = pd.read_parquet(occ_file)
            print(f"Loaded {len(self.esco_occupations)} ESCO occupations")
        else:
            self.esco_occupations = None
            print("No ESCO occupations data found")
        
        # Load ESCO skills
        skills_file = self.esco_data_dir / "esco_skills.parquet"
        if skills_file.exists():
            self.esco_skills = pd.read_parquet(skills_file)
            print(f"Loaded {len(self.esco_skills)} ESCO skills")
        else:
            self.esco_skills = None
            print("No ESCO skills data found")
    
    def _create_mappings(self):
        """Create mappings between model vocabulary and ESCO data."""
        if self.esco_occupations is None:
            self.job_mappings = {}
            return
        
        # Create a simple mapping based on job titles
        self.job_mappings = {}
        
        # For each ESCO occupation, try to find a matching model vocab entry
        for _, occ_row in self.esco_occupations.iterrows():
            esco_id = occ_row['esco_id']
            esco_title = occ_row['title'].lower()
            
            # Look for exact or partial matches in model vocabulary
            best_match = None
            best_score = 0
            
            for vocab_key, vocab_id in self.job_to_id.items():
                vocab_key_lower = vocab_key.lower()
                
                # Exact match
                if esco_title == vocab_key_lower:
                    best_match = vocab_key
                    best_score = 1.0
                    break
                
                # Partial match scoring
                score = 0
                if esco_title in vocab_key_lower:
                    score = len(esco_title) / len(vocab_key_lower)
                elif vocab_key_lower in esco_title:
                    score = len(vocab_key_lower) / len(esco_title)
                
                if score > best_score:
                    best_match = vocab_key
                    best_score = score
            
            if best_match and best_score > 0.3:  # Threshold for good matches
                self.job_mappings[esco_id] = {
                    'esco_title': occ_row['title'],
                    'model_key': best_match,
                    'model_id': self.job_to_id[best_match],
                    'match_score': best_score
                }
        
        print(f"Created mappings for {len(self.job_mappings)} jobs")
    
    def map_user_jobs_to_model(self, user_job_history: List[str]) -> List[int]:
        """
        Map user job history to model input IDs.
        
        Args:
            user_job_history: List of user job titles
            
        Returns:
            List of model-compatible job IDs
        """
        job_ids = []
        
        for job_title in user_job_history:
            job_title_lower = job_title.lower()
            
            # First, try to find exact matches in model vocabulary
            found_exact = False
            for vocab_key, vocab_id in self.job_to_id.items():
                if job_title_lower == vocab_key.lower():
                    job_ids.append(vocab_id)
                    found_exact = True
                    break
            
            if found_exact:
                continue
            
            # If no exact match, try to find in ESCO mappings
            best_esco_match = None
            best_score = 0
            
            for esco_id, mapping in self.job_mappings.items():
                esco_title = mapping['esco_title'].lower()
                
                # Exact match
                if job_title_lower == esco_title:
                    best_esco_match = mapping
                    best_score = 1.0
                    break
                
                # Partial match scoring
                score = 0
                if job_title_lower in esco_title:
                    score = len(job_title_lower) / len(esco_title)
                elif esco_title in job_title_lower:
                    score = len(esco_title) / len(job_title_lower)
                
                if score > best_score:
                    best_esco_match = mapping
                    best_score = score
            
            if best_esco_match and best_score > 0.3:
                job_ids.append(best_esco_match['model_id'])
            else:
                # Fallback to a random job from vocabulary (but not padding or mask)
                vocab_ids = [v for v in self.job_to_id.values() if v > 1]
                if vocab_ids:
                    job_ids.append(random.choice(vocab_ids))
                else:
                    job_ids.append(2)  # Default fallback
        
        return job_ids if job_ids else [2]  # Ensure at least one job
    
    def generate_recommendations(self, 
                               user_job_history: List[str], 
                               user_skills: List[str] = None,
                               top_k: int = 5) -> List[Dict]:
        """
        Generate personalized career path recommendations.
        
        Args:
            user_job_history: List of user's job history
            user_skills: List of user's skills (optional)
            top_k: Number of recommendations to generate
            
        Returns:
            List of recommendation dictionaries
        """
        # Map user jobs to model input
        job_ids = self.map_user_jobs_to_model(user_job_history)
        print(f"Mapped user jobs {user_job_history} to model IDs {job_ids}")
        
        # Convert to tensor
        input_tensor = torch.tensor([job_ids], dtype=torch.long)
        
        # Generate recommendations
        with torch.no_grad():
            recommendations = self.model.generate_next_jobs(input_tensor, top_k=top_k)
        
        # Process recommendations
        results = []
        for i, rec in enumerate(recommendations[0]):  # First batch element
            model_job_id, probability = rec
            
            # Get model job key
            model_job_key = self.id_to_job.get(model_job_id, f"job_{model_job_id}")
            
            # Try to map back to ESCO if possible
            esco_info = None
            for esco_id, mapping in self.job_mappings.items():
                if mapping['model_id'] == model_job_id:
                    esco_info = {
                        'esco_id': esco_id,
                        'esco_title': mapping['esco_title']
                    }
                    break
            
            # Create recommendation entry
            rec_entry = {
                'rank': i + 1,
                'model_job_key': model_job_key,
                'model_job_id': model_job_id,
                'probability': float(probability),
                'esco_info': esco_info
            }
            
            results.append(rec_entry)
        
        return results
    
    def get_diverse_recommendations(self, 
                                  user_job_history: List[str], 
                                  num_variants: int = 3,
                                  top_k: int = 5) -> List[List[Dict]]:
        """
        Generate diverse recommendations by varying the input.
        
        Args:
            user_job_history: List of user's job history
            num_variants: Number of input variants to try
            top_k: Number of recommendations per variant
            
        Returns:
            List of recommendation lists (one per variant)
        """
        all_recommendations = []
        
        # Base recommendations
        base_recs = self.generate_recommendations(user_job_history, top_k=top_k)
        all_recommendations.append(base_recs)
        
        # Variants with modified input
        for i in range(1, min(num_variants, len(user_job_history) + 1)):
            # Create variant by truncating history
            variant_history = user_job_history[-i:] if i <= len(user_job_history) else user_job_history
            variant_recs = self.generate_recommendations(variant_history, top_k=top_k)
            all_recommendations.append(variant_recs)
        
        return all_recommendations

def create_career_recommender(model_dir: str = "models/bert4rec",
                             esco_data_dir: str = "data/processed") -> CareerPathRecommender:
    """
    Factory function to create career path recommender.
    
    Args:
        model_dir: Directory containing model files
        esco_data_dir: Directory containing ESCO data
        
    Returns:
        Initialized CareerPathRecommender
    """
    # Handle relative paths
    if not Path(model_dir).is_absolute():
        model_dir = project_root / model_dir
    if not Path(esco_data_dir).is_absolute():
        esco_data_dir = project_root / esco_data_dir
    
    return CareerPathRecommender(
        model_checkpoint_path=str(Path(model_dir) / "checkpoints" / "best_model.pt"),
        model_config_path=str(Path(model_dir) / "model_config.json"),
        job_vocab_path=str(Path(model_dir) / "job_vocab.json"),
        esco_data_dir=Path(esco_data_dir)
    )

# Demo usage
if __name__ == "__main__":
    # This would be used in the Streamlit app
    recommender = create_career_recommender()
    
    # Example usage
    user_history = ["Software Engineer", "Data Analyst"]
    recommendations = recommender.generate_recommendations(user_history, top_k=3)
    
    print("Career Recommendations:")
    for rec in recommendations:
        print(f"  {rec['rank']}. {rec['model_job_key']} (Probability: {rec['probability']:.3f})")
        if rec['esco_info']:
            print(f"     ESCO: {rec['esco_info']['esco_title']} ({rec['esco_info']['esco_id']})")