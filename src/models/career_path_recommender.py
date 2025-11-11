#!/usr/bin/env python3
"""
Career Path Recommender - Enhanced version that bridges synthetic model training with real ESCO data
"""

import torch
import json
import random
import logging
import yaml
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
        
        # Initialize text mapper once
        from src.ingest.text_to_esco_mapper import TextToESCOMapper
        self.text_mapper = TextToESCOMapper(data_dir=str(esco_data_dir))
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load system config
        self.config = self._read_config()

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
        self.logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _load_vocabulary(self):
        """Load model vocabulary."""
        with open(self.job_vocab_path, 'r') as f:
            self.job_to_id = json.load(f)
        self.id_to_job = {v: k for k, v in self.job_to_id.items()}
        
        self.logger.info(f"Loaded vocabulary with {len(self.job_to_id)} jobs")
    
    def _load_esco_data(self):
        """Load ESCO occupations and skills data."""
        import pandas as pd
        
        # Load ESCO occupations
        occ_file = self.esco_data_dir / "raw/esco_occupations.csv"  # Look in raw data folder for CSV
        if occ_file.exists():
            self.esco_occupations = pd.read_csv(occ_file)
            self.logger.info(f"Loaded {len(self.esco_occupations)} ESCO occupations")
        else:
            self.esco_occupations = None
            self.logger.info("No ESCO occupations data found")
        
        # Load ESCO skills  
        skills_file = self.esco_data_dir / "raw/esco_skills.csv"  # Look in raw data folder for CSV
        if skills_file.exists():
            self.esco_skills = pd.read_csv(skills_file)
            self.logger.info(f"Loaded {len(self.esco_skills)} ESCO skills")
        else:
            self.esco_skills = None
            self.logger.info("No ESCO skills data found")

    def _read_config(self):
        cfg = {}
        try:
            cfg_path = Path('configs') / 'system_config.yaml'
            if cfg_path.exists():
                with open(cfg_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        return cfg
    
    def _create_mappings(self):
        """Create mappings between model vocabulary and ESCO data."""
        if self.esco_occupations is None:
            self.job_mappings = {}
            self.model_to_esco = {}
            return
        
        # Create mappings
        self.job_mappings = {}
        self.model_to_esco = {}

        # Get all model vocab keys (excluding special tokens)
        model_vocab_keys = [k for k in self.job_to_id.keys() if not k.startswith('<')]

        # Confidence threshold from config
        conf_thresh = float(self.config.get('mapping', {}).get('confidence_threshold', 0.4))

        # For each model vocab key, try to map it to an ESCO occupation
        for model_key in model_vocab_keys:
            esco_id, score = self.text_mapper.map_text_to_occupations(
                model_key, top_k=1, score_threshold=conf_thresh
            )

            if esco_id:
                esco_title = self.esco_occupations[self.esco_occupations['esco_id'] == esco_id]['title'].iloc[0]
                model_id = self.job_to_id[model_key]

                self.job_mappings[esco_id] = {
                    'esco_title': esco_title,
                    'model_key': model_key,
                    'model_id': model_id,
                    'match_score': score
                }

                # Reverse mapping
                self.model_to_esco[model_id] = {
                    'esco_id': esco_id,
                    'esco_title': esco_title
                }

        self.logger.info(f"Created mappings for {len(self.job_mappings)} ESCO occupations (threshold={conf_thresh})")
    
    def map_user_jobs_to_model(self, user_job_history: List[str]) -> List[int]:
        """
        Map user job history to model input IDs.
        
        Args:
            user_job_history: List of user job titles
            
        Returns:
            List of model-compatible job IDs
        """
        job_ids = []

        # Lazy initialize semantic mapper only when needed
        from src.ingest.text_to_esco_mapper import map_text_to_occupations

        conf_thresh = float(self.config.get('mapping', {}).get('confidence_threshold', 0.4))

        for job_title in user_job_history:
            job_title_lower = job_title.lower()

            # Try exact ESCO title matches in our job_mappings
            matched_model_id = None
            self.logger.debug(f"Available mappings: {self.job_mappings}")
            self.logger.debug(f"Looking for exact match for: '{job_title_lower}'")
            for esco_id, mapping in self.job_mappings.items():
                self.logger.debug(f"Checking against ESCO title: '{mapping['esco_title'].lower()}'")
                if job_title_lower == mapping['esco_title'].lower():
                    matched_model_id = mapping['model_id']
                    self.logger.debug(f"Exact ESCO title match for '{job_title}' -> {matched_model_id}")
                    break

            # If no exact match, use semantic mapper as default
            if matched_model_id is None:
                esco_id, score = self.text_mapper.map_text_to_occupations(job_title, top_k=1, score_threshold=conf_thresh)
                if esco_id and score >= conf_thresh and esco_id in self.job_mappings:
                    matched_model_id = self.job_mappings[esco_id]['model_id']
                    self.logger.debug(f"Semantic mapper matched '{job_title}' -> {esco_id} (score={score:.2f}) -> model_id {matched_model_id}")
                else:
                    # Leave unmapped rather than coercing to a default
                    self.logger.debug(f"Could not map '{job_title}' confidently (score={score if 'score' in locals() else 0.0}). Leaving unmapped.")

            if matched_model_id is not None:
                job_ids.append(matched_model_id)

        return job_ids
    
    def generate_recommendations(self, 
                               user_job_history: List[str], 
                               user_skills: List[str] = None,
                               top_k: int = 5,
                               exclude_seen: bool = True) -> List[Dict]:
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
        self.logger.debug(f"Mapped user jobs {user_job_history} to model IDs {job_ids}")

        if not job_ids:
            self.logger.warning("No mapped job ids from user input; returning empty recommendations")
            return []

        # Convert to tensor
        input_tensor = torch.tensor([job_ids], dtype=torch.long)

        # Run model forward to get logits [B, L, V]
        with torch.no_grad():
            logits = self.model(input_tensor)

        # Get last token logits and compute probabilities
        last_logits = logits[:, -1, :].squeeze(0)  # [V]
        probs = torch.nn.functional.softmax(last_logits, dim=-1)

        pad_id = self.model.pad_token_id if hasattr(self.model, 'pad_token_id') else 0
        mask_id = self.model.mask_token_id if hasattr(self.model, 'mask_token_id') else 1

        # Set PAD/MASK to 0 probability
        if 0 <= pad_id < probs.size(0):
            probs[pad_id] = 0.0
        if 0 <= mask_id < probs.size(0):
            probs[mask_id] = 0.0

        # Optionally exclude seen ids
        seen_ids = set(job_ids)
        if exclude_seen:
            for sid in seen_ids:
                if 0 <= sid < probs.size(0):
                    probs[sid] = 0.0

        # Compute combined score if feasibility available (placeholder: 1.0)
        alpha = float(self.config.get('scoring', {}).get('alpha', 0.6))
        beta = float(self.config.get('scoring', {}).get('beta', 0.4))

        # Use predict_next helper to pick top-k after filtering
        from src.models.bert4rec import predict_next

        topk = predict_next(last_logits, pad_token_id=pad_id, mask_token_id=mask_id,
                            exclude_ids=seen_ids if exclude_seen else set(), top_k=top_k)

        results = []
        for rank, (model_job_id, prob) in enumerate(topk, start=1):
            model_job_key = self.id_to_job.get(model_job_id, f"job_{model_job_id}")

            esco_info = None
            if model_job_id in self.model_to_esco:
                esco_info = self.model_to_esco[model_job_id]
            else:
                # Try to find any mapping
                for esco_id, mapping in self.job_mappings.items():
                    if mapping.get('model_id') == model_job_id:
                        esco_info = {'esco_id': esco_id, 'esco_title': mapping['esco_title']}
                        break

            if esco_info is None:
                esco_info = {'esco_id': f'recommended_{model_job_id}', 'esco_title': model_job_key}

            rec_entry = {
                'rank': rank,
                'model_job_key': model_job_key,
                'model_job_id': model_job_id,
                'probability': float(prob),
                'esco_info': esco_info
            }
            results.append(rec_entry)

        self.logger.debug(f"Top-{top_k} recommendations: {results}")
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