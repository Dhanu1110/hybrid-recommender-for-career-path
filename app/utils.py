"""Utility functions for Streamlit app."""
from typing import Tuple, Optional
import yaml
from pathlib import Path
import streamlit as st

def map_job_title_with_confidence(title: str) -> Tuple[Optional[str], float]:
    """Map input job title to ESCO occupation with confidence score."""
    try:
        if not title.strip():
            return None, 0.0
        
        # Get threshold from config
        config_path = Path("configs") / "system_config.yaml"
        threshold = 0.4  # Default threshold
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                    threshold = float(cfg.get('mapping', {}).get('confidence_threshold', 0.4))
            except Exception as e:
                st.warning(f"Could not load config threshold: {e}. Using default 0.4")
        
        # Use improved mapper with confidence score
        from ingest.text_to_esco_mapper import map_text_to_occupations
        esco_id, score = map_text_to_occupations(
            title, 
            score_threshold=threshold,
            data_dir="data/processed"
        )
        
        if esco_id is None:
            return None, score
            
        # Get occupation title
        occupation = st.session_state.esco_loader.get_occupation_by_id(esco_id)
        if occupation:
            return occupation['title'], score
        return None, score
            
    except Exception as e:
        st.error(f"Error mapping job title: {e}")
        return None, 0.0

def map_skill_with_confidence(skill: str) -> Tuple[Optional[str], float]:
    """Map input skill to ESCO skill with confidence score."""
    try:
        if not skill.strip():
            return None, 0.0
        
        # Get threshold from config
        config_path = Path("configs") / "system_config.yaml"
        threshold = 0.4  # Default threshold
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                    threshold = float(cfg.get('mapping', {}).get('confidence_threshold', 0.4))
            except Exception as e:
                st.warning(f"Could not load config threshold: {e}. Using default 0.4")
        
        # Use text mapper
        text_mapper = st.session_state.get('text_mapper')
        if text_mapper is None:
            from ingest.text_to_esco_mapper import create_text_mapper
            text_mapper = create_text_mapper("data/processed")
            st.session_state.text_mapper = text_mapper
            
        matches = text_mapper.map_text_to_skills(skill, top_k=1, score_threshold=threshold)
        if not matches:
            return None, 0.0
            
        top_match = matches[0]
        esco_id = top_match['esco_id']
        score = float(top_match.get('score', 0.0))
        
        # Get skill title
        skill_data = st.session_state.esco_loader.get_skill_by_id(esco_id)
        if skill_data:
            return skill_data['title'], score
        return None, score
            
    except Exception as e:
        st.error(f"Error mapping skill: {e}")
        return None, 0.0