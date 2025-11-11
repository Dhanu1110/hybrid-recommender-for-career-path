#!/usr/bin/env python3
"""
ESCO Knowledge Graph Loader

This module provides functionality to load and query ESCO (European Skills, Competences, 
Qualifications and Occupations) data as a knowledge graph.

Key functions:
- get_job_skills(job_esco_id): Get skills required for a job
- get_skill_parents(skill_esco_id): Get parent skills in hierarchy
- get_skill_distance(skill_a, skill_b): Calculate semantic distance between skills
"""

import logging
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
import numpy as np
from collections import defaultdict, deque
import pickle
from .text_to_esco_mapper import TextToESCOMapper

logger = logging.getLogger(__name__)

class ESCOKnowledgeGraph:
    """ESCO Knowledge Graph for querying occupations, skills, and their relationships."""
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize ESCO Knowledge Graph.
        
        Args:
            data_dir: Directory containing processed ESCO data files
        """
        self.data_dir = Path(data_dir)
        self.graph = nx.DiGraph()
        self.occupations = {}
        self.skills = {}
        self.job_skills_cache = {}
        self.skill_hierarchy_cache = {}
        self.distance_cache = {}
        
        # Load data
        self._load_data()
        self._build_graph()
        
        # Initialize mapper
        self.mapper = TextToESCOMapper()
        
    def _load_data(self) -> None:
        """Load ESCO data from processed Parquet files."""
        logger.info("Loading ESCO data...")
        
        try:
            # Load occupations
            occ_file = self.data_dir / "esco_occupations.parquet"
            if occ_file.exists():
                occ_df = pd.read_parquet(occ_file)
                self.occupations = {
                    row['esco_id']: {
                        'title': row['title'],
                        'description': row.get('description', ''),
                        'type': 'occupation'
                    }
                    for _, row in occ_df.iterrows()
                }
                logger.info(f"Loaded {len(self.occupations)} occupations")
            
            # Load skills
            skills_file = self.data_dir / "esco_skills.parquet"
            if skills_file.exists():
                skills_df = pd.read_parquet(skills_file)
                self.skills = {
                    row['esco_id']: {
                        'title': row['title'],
                        'description': row.get('description', ''),
                        'skill_type': row.get('skill_type', 'skill'),
                        'type': 'skill'
                    }
                    for _, row in skills_df.iterrows()
                }
                logger.info(f"Loaded {len(self.skills)} skills")
            
            # Load relations
            rel_file = self.data_dir / "esco_relations.parquet"
            if rel_file.exists():
                self.relations_df = pd.read_parquet(rel_file)
                logger.info(f"Loaded {len(self.relations_df)} relations")
            else:
                self.relations_df = pd.DataFrame(columns=['source_id', 'target_id', 'relation_type'])
                
        except Exception as e:
            logger.error(f"Error loading ESCO data: {e}")
            raise
    
    def _build_graph(self) -> None:
        """Build NetworkX graph from ESCO data."""
        logger.info("Building knowledge graph...")
        
        # Add nodes
        for esco_id, data in self.occupations.items():
            self.graph.add_node(esco_id, **data)
            
        for esco_id, data in self.skills.items():
            self.graph.add_node(esco_id, **data)
        
        # Add edges from relations
        for _, row in self.relations_df.iterrows():
            source_id = row['source_id']
            target_id = row['target_id']
            relation_type = row['relation_type']
            
            if source_id in self.graph.nodes and target_id in self.graph.nodes:
                self.graph.add_edge(source_id, target_id, relation_type=relation_type)
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def get_job_skills(self, job_esco_id: str) -> List[Dict[str, str]]:
        """
        Get skills required for a specific job.
        
        Args:
            job_esco_id: ESCO ID of the job/occupation
            
        Returns:
            List of skill dictionaries with esco_id, title, and skill_type
        """
        if job_esco_id in self.job_skills_cache:
            return self.job_skills_cache[job_esco_id]
        
        skills = []
        
        if job_esco_id not in self.graph.nodes:
            logger.warning(f"Job {job_esco_id} not found in knowledge graph")
            return skills
        
        # Find all skills connected to this job via 'requires' relation
        for neighbor in self.graph.neighbors(job_esco_id):
            edge_data = self.graph.get_edge_data(job_esco_id, neighbor)
            if edge_data and edge_data.get('relation_type') == 'requires':
                if neighbor in self.skills:
                    skill_data = self.skills[neighbor].copy()
                    skill_data['esco_id'] = neighbor
                    skills.append(skill_data)
        
        # Cache result
        self.job_skills_cache[job_esco_id] = skills
        return skills
    
    def get_skill_parents(self, skill_esco_id: str) -> List[Dict[str, str]]:
        """
        Get parent skills in the skill hierarchy.
        
        Args:
            skill_esco_id: ESCO ID of the skill
            
        Returns:
            List of parent skill dictionaries
        """
        if skill_esco_id in self.skill_hierarchy_cache:
            return self.skill_hierarchy_cache[skill_esco_id]
        
        parents = []
        
        if skill_esco_id not in self.graph.nodes:
            logger.warning(f"Skill {skill_esco_id} not found in knowledge graph")
            return parents
        
        # Find parent skills via 'broader' relation
        for neighbor in self.graph.neighbors(skill_esco_id):
            edge_data = self.graph.get_edge_data(skill_esco_id, neighbor)
            if edge_data and edge_data.get('relation_type') == 'broader':
                if neighbor in self.skills:
                    parent_data = self.skills[neighbor].copy()
                    parent_data['esco_id'] = neighbor
                    parents.append(parent_data)
        
        # Cache result
        self.skill_hierarchy_cache[skill_esco_id] = parents
        return parents
    
    def get_skill_distance(self, skill_a: str, skill_b: str) -> float:
        """
        Calculate semantic distance between two skills.
        
        Args:
            skill_a: ESCO ID of first skill
            skill_b: ESCO ID of second skill
            
        Returns:
            Distance value (0.0 = identical, higher = more distant)
        """
        if skill_a == skill_b:
            return 0.0
        
        # Check cache
        cache_key = tuple(sorted([skill_a, skill_b]))
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        distance = self._calculate_graph_distance(skill_a, skill_b)
        
        # Cache result
        self.distance_cache[cache_key] = distance
        return distance
    
    def _calculate_graph_distance(self, skill_a: str, skill_b: str) -> float:
        """Calculate distance using graph-based methods."""
        if skill_a not in self.graph.nodes or skill_b not in self.graph.nodes:
            return float('inf')
        
        try:
            # Try shortest path in undirected version of graph
            undirected_graph = self.graph.to_undirected()
            path_length = nx.shortest_path_length(undirected_graph, skill_a, skill_b)
            
            # Normalize distance (you can adjust this formula)
            normalized_distance = min(path_length / 10.0, 1.0)
            return normalized_distance
            
        except nx.NetworkXNoPath:
            # No path found, return high distance
            return 1.0
    
    def get_related_skills(self, skill_esco_id: str, max_distance: float = 0.5) -> List[Dict[str, Union[str, float]]]:
        """
        Get skills related to a given skill within a distance threshold.
        
        Args:
            skill_esco_id: ESCO ID of the skill
            max_distance: Maximum distance threshold
            
        Returns:
            List of related skills with their distances
        """
        related = []
        
        if skill_esco_id not in self.graph.nodes:
            return related
        
        # Get all skills within reasonable graph distance
        try:
            undirected_graph = self.graph.to_undirected()
            # Limit search to avoid performance issues
            max_hops = min(int(max_distance * 10), 5)
            
            for skill_id in self.skills.keys():
                if skill_id != skill_esco_id:
                    distance = self.get_skill_distance(skill_esco_id, skill_id)
                    if distance <= max_distance:
                        skill_data = self.skills[skill_id].copy()
                        skill_data['esco_id'] = skill_id
                        skill_data['distance'] = distance
                        related.append(skill_data)
            
            # Sort by distance
            related.sort(key=lambda x: x['distance'])
            
        except Exception as e:
            logger.warning(f"Error finding related skills for {skill_esco_id}: {e}")
        
        return related
    
    def get_skill_hierarchy_path(self, skill_esco_id: str) -> List[str]:
        """
        Get the full hierarchy path from skill to root.
        
        Args:
            skill_esco_id: ESCO ID of the skill
            
        Returns:
            List of skill IDs from specific to general
        """
        path = [skill_esco_id]
        current = skill_esco_id
        visited = set()
        
        while current not in visited:
            visited.add(current)
            parents = self.get_skill_parents(current)
            if parents:
                # Take the first parent (you might want to implement better logic)
                parent_id = parents[0]['esco_id']
                path.append(parent_id)
                current = parent_id
            else:
                break
        
        return path
    
    def save_cache(self, cache_file: str = "data/processed/esco_cache.pkl") -> None:
        """Save computed caches to disk."""
        cache_data = {
            'job_skills_cache': self.job_skills_cache,
            'skill_hierarchy_cache': self.skill_hierarchy_cache,
            'distance_cache': self.distance_cache
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved cache to {cache_file}")
    
    def map_job_title(self, title: str) -> Optional[str]:
        """
        Map a job title to its ESCO occupation ID.
        
        Args:
            title: Job title text to map
            
        Returns:
            ESCO occupation ID if match found, None otherwise
        """
        # Use TextToESCOMapper to find matches
        try:
            matches = self.mapper.map_text_to_occupations(title)
        except Exception:
            return None

        # mapper may return either a tuple (esco_id, score) or a list of dicts
        if not matches:
            return None

        if isinstance(matches, tuple) and len(matches) == 2:
            return matches[0]

        # If mapper returns a list of dicts, pick the top one's esco_id
        if isinstance(matches, list) and len(matches) > 0 and isinstance(matches[0], dict):
            return matches[0].get('esco_id')

        return None

    def get_occupation_by_id(self, esco_id: str) -> Optional[Dict[str, str]]:
        """Return occupation record for given ESCO id or None."""
        return self.occupations.get(esco_id)

    def get_skill_by_id(self, esco_id: str) -> Optional[Dict[str, str]]:
        """Return skill record for given ESCO id or None."""
        return self.skills.get(esco_id)

    def load_cache(self, cache_file: str = "data/processed/esco_cache.pkl") -> None:
        """Load computed caches from disk."""
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.job_skills_cache = cache_data.get('job_skills_cache', {})
            self.skill_hierarchy_cache = cache_data.get('skill_hierarchy_cache', {})
            self.distance_cache = cache_data.get('distance_cache', {})
            
            logger.info(f"Loaded cache from {cache_file}")
            
        except FileNotFoundError:
            logger.info("No cache file found, starting with empty caches")
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")

def create_esco_loader(data_dir: str = "data/processed") -> ESCOKnowledgeGraph:
    """
    Factory function to create and initialize ESCO knowledge graph.
    
    Args:
        data_dir: Directory containing processed ESCO data
        
    Returns:
        Initialized ESCOKnowledgeGraph instance
    """
    loader = ESCOKnowledgeGraph(data_dir)
    
    # Try to load cache
    cache_file = Path(data_dir) / "esco_cache.pkl"
    if cache_file.exists():
        loader.load_cache(str(cache_file))
    
    return loader

# Convenience functions for direct usage
_global_loader = None

def get_job_skills(job_esco_id: str) -> List[Dict[str, str]]:
    """Global function to get job skills."""
    global _global_loader
    if _global_loader is None:
        _global_loader = create_esco_loader()
    return _global_loader.get_job_skills(job_esco_id)

def get_skill_parents(skill_esco_id: str) -> List[Dict[str, str]]:
    """Global function to get skill parents."""
    global _global_loader
    if _global_loader is None:
        _global_loader = create_esco_loader()
    return _global_loader.get_skill_parents(skill_esco_id)

def get_skill_distance(skill_a: str, skill_b: str) -> float:
    """Global function to get skill distance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = create_esco_loader()
    return _global_loader.get_skill_distance(skill_a, skill_b)
