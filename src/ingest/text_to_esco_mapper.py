#!/usr/bin/env python3
"""
Text-to-ESCO Mapping Pipeline

This module provides functionality to map free-text job titles and skills to ESCO IDs
using fuzzy matching and semantic similarity with sentence transformers.

Key components:
1. Text normalization
2. Fuzzy string matching
3. Semantic embedding similarity using sentence-transformers
4. FAISS-based nearest neighbor search
"""

import logging
import re
import string
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from difflib import SequenceMatcher
import pickle

# ML/NLP imports
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, falling back to fuzzy matching only")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using slower similarity search")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class TextToESCOMapper:
    """Maps free text to ESCO occupations and skills using multiple similarity methods."""
    
    def __init__(self, data_dir: str = "data/processed", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the text-to-ESCO mapper.
        
        Args:
            data_dir: Directory containing processed ESCO data
            model_name: Sentence transformer model name
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        
        # Data storage
        self.occupations = {}
        self.skills = {}
        self.occupation_texts = []
        self.skill_texts = []
        self.occupation_ids = []
        self.skill_ids = []
        
        # Models and indices
        self.sentence_model = None
        self.occupation_embeddings = None
        self.skill_embeddings = None
        self.occupation_index = None
        self.skill_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Load data and build indices
        self._load_data()
        self._initialize_models()
        self._build_indices()
    
    def _load_data(self) -> None:
        """Load ESCO occupations and skills data."""
        logger.info("Loading ESCO data for mapping...")
        
        # Load occupations
        occ_file = self.data_dir / "esco_occupations.parquet"
        if occ_file.exists():
            occ_df = pd.read_parquet(occ_file)
            for _, row in occ_df.iterrows():
                esco_id = row['esco_id']
                self.occupations[esco_id] = {
                    'title': row['title'],
                    'description': row.get('description', ''),
                    'type': 'occupation'
                }
                # Create searchable text
                search_text = f"{row['title']} {row.get('description', '')}"
                self.occupation_texts.append(self._normalize_text(search_text))
                self.occupation_ids.append(esco_id)
            
            logger.info(f"Loaded {len(self.occupations)} occupations")
        
        # Load skills
        skills_file = self.data_dir / "esco_skills.parquet"
        if skills_file.exists():
            skills_df = pd.read_parquet(skills_file)
            for _, row in skills_df.iterrows():
                esco_id = row['esco_id']
                self.skills[esco_id] = {
                    'title': row['title'],
                    'description': row.get('description', ''),
                    'skill_type': row.get('skill_type', 'skill'),
                    'type': 'skill'
                }
                # Create searchable text
                search_text = f"{row['title']} {row.get('description', '')}"
                self.skill_texts.append(self._normalize_text(search_text))
                self.skill_ids.append(esco_id)
            
            logger.info(f"Loaded {len(self.skills)} skills")
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation except hyphens and apostrophes
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _initialize_models(self) -> None:
        """Initialize sentence transformer and other models."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self.sentence_model = SentenceTransformer(self.model_name)
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
        
        # Initialize TF-IDF as fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
    
    def _build_indices(self) -> None:
        """Build search indices for fast similarity search."""
        logger.info("Building search indices...")
        
        # Build sentence transformer embeddings and FAISS indices
        if self.sentence_model is not None:
            self._build_embedding_indices()
        
        # Build TF-IDF index as fallback
        self._build_tfidf_index()
        
        logger.info("Search indices built successfully")
    
    def _build_embedding_indices(self) -> None:
        """Build FAISS indices using sentence embeddings."""
        if not self.sentence_model:
            return
        
        # Generate embeddings for occupations
        if self.occupation_texts:
            logger.info("Generating occupation embeddings...")
            self.occupation_embeddings = self.sentence_model.encode(
                self.occupation_texts, 
                show_progress_bar=True
            )
            
            # Build FAISS index for occupations
            if FAISS_AVAILABLE:
                dimension = self.occupation_embeddings.shape[1]
                self.occupation_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.occupation_embeddings)
                self.occupation_index.add(self.occupation_embeddings.astype(np.float32))
        
        # Generate embeddings for skills
        if self.skill_texts:
            logger.info("Generating skill embeddings...")
            self.skill_embeddings = self.sentence_model.encode(
                self.skill_texts,
                show_progress_bar=True
            )
            
            # Build FAISS index for skills
            if FAISS_AVAILABLE:
                dimension = self.skill_embeddings.shape[1]
                self.skill_index = faiss.IndexFlatIP(dimension)
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.skill_embeddings)
                self.skill_index.add(self.skill_embeddings.astype(np.float32))
    
    def _build_tfidf_index(self) -> None:
        """Build TF-IDF index for fallback similarity search."""
        all_texts = self.occupation_texts + self.skill_texts
        if all_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
    
    def map_text_to_occupations(self, text: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Map free text to ESCO occupations.
        
        Args:
            text: Input text (job title, description, etc.)
            top_k: Number of top matches to return
            
        Returns:
            List of occupation matches with scores
        """
        normalized_text = self._normalize_text(text)
        
        # Try semantic similarity first
        if self.sentence_model and self.occupation_index and FAISS_AVAILABLE:
            return self._semantic_search_occupations(normalized_text, top_k)
        
        # Fallback to TF-IDF similarity
        return self._tfidf_search_occupations(normalized_text, top_k)
    
    def map_text_to_skills(self, text: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Map free text to ESCO skills.
        
        Args:
            text: Input text (skill description, etc.)
            top_k: Number of top matches to return
            
        Returns:
            List of skill matches with scores
        """
        normalized_text = self._normalize_text(text)
        
        # Try semantic similarity first
        if self.sentence_model and self.skill_index and FAISS_AVAILABLE:
            return self._semantic_search_skills(normalized_text, top_k)
        
        # Fallback to TF-IDF similarity
        return self._tfidf_search_skills(normalized_text, top_k)
    
    def _semantic_search_occupations(self, text: str, top_k: int) -> List[Dict[str, Union[str, float]]]:
        """Search occupations using semantic embeddings."""
        query_embedding = self.sentence_model.encode([text])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.occupation_index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.occupation_ids):
                esco_id = self.occupation_ids[idx]
                occupation_data = self.occupations[esco_id].copy()
                occupation_data['esco_id'] = esco_id
                occupation_data['score'] = float(score)
                occupation_data['method'] = 'semantic'
                results.append(occupation_data)
        
        return results
    
    def _semantic_search_skills(self, text: str, top_k: int) -> List[Dict[str, Union[str, float]]]:
        """Search skills using semantic embeddings."""
        query_embedding = self.sentence_model.encode([text])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.skill_index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.skill_ids):
                esco_id = self.skill_ids[idx]
                skill_data = self.skills[esco_id].copy()
                skill_data['esco_id'] = esco_id
                skill_data['score'] = float(score)
                skill_data['method'] = 'semantic'
                results.append(skill_data)
        
        return results
    
    def _tfidf_search_occupations(self, text: str, top_k: int) -> List[Dict[str, Union[str, float]]]:
        """Search occupations using TF-IDF similarity."""
        if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            return []
        
        query_vector = self.tfidf_vectorizer.transform([text])
        
        # Calculate similarity with occupation texts only
        occ_matrix = self.tfidf_matrix[:len(self.occupation_texts)]
        similarities = cosine_similarity(query_vector, occ_matrix).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                esco_id = self.occupation_ids[idx]
                occupation_data = self.occupations[esco_id].copy()
                occupation_data['esco_id'] = esco_id
                occupation_data['score'] = float(similarities[idx])
                occupation_data['method'] = 'tfidf'
                results.append(occupation_data)
        
        return results
    
    def _tfidf_search_skills(self, text: str, top_k: int) -> List[Dict[str, Union[str, float]]]:
        """Search skills using TF-IDF similarity."""
        if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            return []
        
        query_vector = self.tfidf_vectorizer.transform([text])
        
        # Calculate similarity with skill texts only
        skill_start_idx = len(self.occupation_texts)
        skill_matrix = self.tfidf_matrix[skill_start_idx:]
        similarities = cosine_similarity(query_vector, skill_matrix).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                esco_id = self.skill_ids[idx]
                skill_data = self.skills[esco_id].copy()
                skill_data['esco_id'] = esco_id
                skill_data['score'] = float(similarities[idx])
                skill_data['method'] = 'tfidf'
                results.append(skill_data)
        
        return results
    
    def fuzzy_match_text(self, text: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform fuzzy string matching.
        
        Args:
            text: Query text
            candidates: List of candidate strings
            top_k: Number of top matches to return
            
        Returns:
            List of (candidate, similarity_score) tuples
        """
        normalized_text = self._normalize_text(text)
        
        similarities = []
        for candidate in candidates:
            normalized_candidate = self._normalize_text(candidate)
            similarity = SequenceMatcher(None, normalized_text, normalized_candidate).ratio()
            similarities.append((candidate, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_indices(self, save_dir: str = None) -> None:
        """Save built indices to disk."""
        if save_dir is None:
            save_dir = self.data_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self.occupation_embeddings is not None:
            np.save(save_dir / "occupation_embeddings.npy", self.occupation_embeddings)
        
        if self.skill_embeddings is not None:
            np.save(save_dir / "skill_embeddings.npy", self.skill_embeddings)
        
        # Save FAISS indices
        if self.occupation_index is not None:
            faiss.write_index(self.occupation_index, str(save_dir / "occupation_index.faiss"))
        
        if self.skill_index is not None:
            faiss.write_index(self.skill_index, str(save_dir / "skill_index.faiss"))
        
        # Save TF-IDF components
        if self.tfidf_vectorizer is not None:
            with open(save_dir / "tfidf_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        logger.info(f"Saved indices to {save_dir}")

def create_text_mapper(data_dir: str = "data/processed") -> TextToESCOMapper:
    """
    Factory function to create and initialize text-to-ESCO mapper.
    
    Args:
        data_dir: Directory containing processed ESCO data
        
    Returns:
        Initialized TextToESCOMapper instance
    """
    return TextToESCOMapper(data_dir)
