#!/usr/bin/env python3
"""
Skill Gap Reasoning Module

This module implements Stage 2 of the hybrid recommender system, which analyzes
skill gaps between user's current skills and required skills for career paths.

Key functionality:
- Compute required skills for each job in a path
- Calculate skill gaps (missing skills)
- Compute feasibility scores based on skill distance and gaps
- Combine model probabilities with feasibility scores
"""

import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import yaml

# Import ESCO loader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ingest.esco_loader import ESCOKnowledgeGraph, create_esco_loader

logger = logging.getLogger(__name__)

@dataclass
class SkillGap:
    """Represents a skill gap for a specific job."""
    job_id: str
    required_skills: List[str]
    missing_skills: List[str]
    gap_score: float
    skill_distances: Dict[str, float]

@dataclass
class PathAnalysis:
    """Complete analysis of a career path."""
    path: List[str]
    model_prob: float
    feasibility_score: float
    combined_score: float
    per_job_gaps: Dict[str, SkillGap]
    total_missing_skills: int
    avg_skill_distance: float

class SkillGapAnalyzer:
    """Analyzes skill gaps and computes feasibility scores for career paths."""
    
    def __init__(self, 
                 esco_loader: Optional[ESCOKnowledgeGraph] = None,
                 config_path: str = "configs/system_config.yaml"):
        """
        Initialize skill gap analyzer.
        
        Args:
            esco_loader: ESCO knowledge graph loader
            config_path: Path to system configuration
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize ESCO loader
        self.esco_loader = esco_loader or create_esco_loader()
        
        # Configuration parameters
        self.alpha = self.config.get('skill_gap', {}).get('alpha', 0.6)
        self.beta = self.config.get('skill_gap', {}).get('beta', 0.4)
        self.max_skill_distance = self.config.get('skill_gap', {}).get('max_skill_distance', 1.0)
        self.distance_weight = self.config.get('skill_gap', {}).get('distance_weight', 0.5)
        self.gap_penalty = self.config.get('skill_gap', {}).get('gap_penalty', 0.1)
        self.missing_skill_penalty = self.config.get('skill_gap', {}).get('missing_skill_penalty', 0.2)
        
        logger.info("SkillGapAnalyzer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def analyze_path(self, 
                    user_skill_set: Set[str],
                    candidate_path: List[str],
                    model_prob: float) -> PathAnalysis:
        """
        Analyze a complete career path for skill gaps and feasibility.
        
        Args:
            user_skill_set: Set of user's current skill IDs
            candidate_path: List of job IDs in the career path
            model_prob: Model probability for this path
            
        Returns:
            Complete path analysis with gaps and scores
        """
        per_job_gaps = {}
        total_missing_skills = 0
        skill_distances = []
        
        # Analyze each job in the path
        for job_id in candidate_path:
            gap = self._analyze_job_gap(user_skill_set, job_id)
            per_job_gaps[job_id] = gap
            total_missing_skills += len(gap.missing_skills)
            
            # Collect skill distances for averaging
            skill_distances.extend(gap.skill_distances.values())
        
        # Calculate overall feasibility score
        feasibility_score = self._calculate_feasibility_score(per_job_gaps, user_skill_set)
        
        # Combine model probability with feasibility
        combined_score = self._combine_scores(model_prob, feasibility_score)
        
        # Calculate average skill distance
        avg_skill_distance = np.mean(skill_distances) if skill_distances else 0.0
        
        return PathAnalysis(
            path=candidate_path,
            model_prob=model_prob,
            feasibility_score=feasibility_score,
            combined_score=combined_score,
            per_job_gaps=per_job_gaps,
            total_missing_skills=total_missing_skills,
            avg_skill_distance=avg_skill_distance
        )
    
    def _analyze_job_gap(self, user_skill_set: Set[str], job_id: str) -> SkillGap:
        """
        Analyze skill gap for a specific job.
        
        Args:
            user_skill_set: User's current skills
            job_id: Job to analyze
            
        Returns:
            SkillGap object with analysis results
        """
        # Get required skills for the job
        required_skills_data = self.esco_loader.get_job_skills(job_id)
        required_skills = [skill['esco_id'] for skill in required_skills_data]
        
        # Find missing skills
        missing_skills = [skill for skill in required_skills if skill not in user_skill_set]
        
        # Calculate skill distances for missing skills
        skill_distances = {}
        for missing_skill in missing_skills:
            min_distance = self._find_minimum_skill_distance(user_skill_set, missing_skill)
            skill_distances[missing_skill] = min_distance
        
        # Calculate gap score for this job
        gap_score = self._calculate_job_gap_score(required_skills, missing_skills, skill_distances)
        
        return SkillGap(
            job_id=job_id,
            required_skills=required_skills,
            missing_skills=missing_skills,
            gap_score=gap_score,
            skill_distances=skill_distances
        )
    
    def _find_minimum_skill_distance(self, user_skills: Set[str], target_skill: str) -> float:
        """
        Find minimum distance from user's skills to target skill.
        
        Args:
            user_skills: User's current skills
            target_skill: Target skill to reach
            
        Returns:
            Minimum distance to target skill
        """
        if not user_skills:
            return self.max_skill_distance
        
        min_distance = self.max_skill_distance
        
        for user_skill in user_skills:
            distance = self.esco_loader.get_skill_distance(user_skill, target_skill)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _calculate_job_gap_score(self, 
                                required_skills: List[str],
                                missing_skills: List[str],
                                skill_distances: Dict[str, float]) -> float:
        """
        Calculate gap score for a single job.
        
        Args:
            required_skills: All required skills for the job
            missing_skills: Skills the user doesn't have
            skill_distances: Distances to missing skills
            
        Returns:
            Gap score (lower is better, 0 = no gap)
        """
        if not required_skills:
            return 0.0
        
        # Base gap ratio
        gap_ratio = len(missing_skills) / len(required_skills)
        
        # Distance-weighted penalty
        if missing_skills:
            avg_distance = np.mean(list(skill_distances.values()))
            distance_penalty = avg_distance * self.distance_weight
        else:
            distance_penalty = 0.0
        
        # Combine gap ratio and distance penalty
        gap_score = gap_ratio + distance_penalty
        
        return min(gap_score, 1.0)  # Cap at 1.0
    
    def _calculate_feasibility_score(self, 
                                   per_job_gaps: Dict[str, SkillGap],
                                   user_skill_set: Set[str]) -> float:
        """
        Calculate overall feasibility score for the path.
        
        Args:
            per_job_gaps: Gap analysis for each job
            user_skill_set: User's current skills
            
        Returns:
            Feasibility score (higher is better, 0-1 range)
        """
        if not per_job_gaps:
            return 0.0
        
        # Calculate average gap score across all jobs
        gap_scores = [gap.gap_score for gap in per_job_gaps.values()]
        avg_gap_score = np.mean(gap_scores)
        
        # Convert gap score to feasibility (invert and normalize)
        base_feasibility = 1.0 - avg_gap_score
        
        # Apply penalties for missing skills
        total_missing = sum(len(gap.missing_skills) for gap in per_job_gaps.values())
        missing_penalty = min(total_missing * self.missing_skill_penalty, 0.5)
        
        # Final feasibility score
        feasibility_score = max(base_feasibility - missing_penalty, 0.0)
        
        return feasibility_score
    
    def _combine_scores(self, model_prob: float, feasibility_score: float) -> float:
        """
        Combine model probability with feasibility score.
        
        Args:
            model_prob: Model probability for the path
            feasibility_score: Feasibility score based on skill gaps
            
        Returns:
            Combined score
        """
        combined_score = self.alpha * model_prob + self.beta * feasibility_score
        return combined_score
    
    def analyze_multiple_paths(self, 
                             user_skill_set: Set[str],
                             candidate_paths: List[Tuple[List[str], float]]) -> List[PathAnalysis]:
        """
        Analyze multiple career paths and return sorted results.
        
        Args:
            user_skill_set: User's current skills
            candidate_paths: List of (path, model_prob) tuples
            
        Returns:
            List of PathAnalysis objects sorted by combined score
        """
        analyses = []
        
        for path, model_prob in candidate_paths:
            analysis = self.analyze_path(user_skill_set, path, model_prob)
            analyses.append(analysis)
        
        # Sort by combined score (descending)
        analyses.sort(key=lambda x: x.combined_score, reverse=True)
        
        return analyses
    
    def get_learning_plan(self, path_analysis: PathAnalysis) -> Dict[str, List[str]]:
        """
        Generate a learning plan for a career path.
        
        Args:
            path_analysis: Analysis of the career path
            
        Returns:
            Dictionary mapping job IDs to ordered list of skills to learn
        """
        learning_plan = {}
        
        for job_id, gap in path_analysis.per_job_gaps.items():
            if gap.missing_skills:
                # Sort missing skills by distance (closer skills first)
                sorted_skills = sorted(
                    gap.missing_skills,
                    key=lambda skill: gap.skill_distances.get(skill, self.max_skill_distance)
                )
                learning_plan[job_id] = sorted_skills
        
        return learning_plan
    
    def get_skill_prerequisites(self, skill_id: str, max_depth: int = 3) -> List[str]:
        """
        Get prerequisite skills for a given skill using ESCO hierarchy.
        
        Args:
            skill_id: Target skill ID
            max_depth: Maximum depth to search for prerequisites
            
        Returns:
            List of prerequisite skill IDs
        """
        prerequisites = []
        visited = set()
        
        def _get_prerequisites_recursive(current_skill: str, depth: int):
            if depth >= max_depth or current_skill in visited:
                return
            
            visited.add(current_skill)
            parents = self.esco_loader.get_skill_parents(current_skill)
            
            for parent in parents:
                parent_id = parent['esco_id']
                if parent_id not in prerequisites:
                    prerequisites.append(parent_id)
                    _get_prerequisites_recursive(parent_id, depth + 1)
        
        _get_prerequisites_recursive(skill_id, 0)
        return prerequisites
    
    def explain_path_feasibility(self, path_analysis: PathAnalysis) -> Dict[str, Union[str, float, List]]:
        """
        Generate human-readable explanation of path feasibility.
        
        Args:
            path_analysis: Analysis of the career path
            
        Returns:
            Dictionary with explanation components
        """
        explanation = {
            'overall_score': path_analysis.combined_score,
            'model_confidence': path_analysis.model_prob,
            'feasibility': path_analysis.feasibility_score,
            'total_missing_skills': path_analysis.total_missing_skills,
            'avg_skill_distance': path_analysis.avg_skill_distance,
            'job_difficulties': {},
            'recommendations': []
        }
        
        # Analyze difficulty for each job
        for job_id, gap in path_analysis.per_job_gaps.items():
            difficulty = 'Easy' if gap.gap_score < 0.3 else 'Medium' if gap.gap_score < 0.7 else 'Hard'
            explanation['job_difficulties'][job_id] = {
                'difficulty': difficulty,
                'missing_skills_count': len(gap.missing_skills),
                'gap_score': gap.gap_score
            }
        
        # Generate recommendations
        if path_analysis.total_missing_skills > 0:
            explanation['recommendations'].append(
                f"Focus on developing {path_analysis.total_missing_skills} missing skills"
            )
        
        if path_analysis.avg_skill_distance > 0.5:
            explanation['recommendations'].append(
                "Consider intermediate skills to bridge knowledge gaps"
            )
        
        if path_analysis.feasibility_score < 0.5:
            explanation['recommendations'].append(
                "This path may be challenging - consider alternative routes"
            )
        
        return explanation

def create_skill_gap_analyzer(config_path: str = "configs/system_config.yaml") -> SkillGapAnalyzer:
    """
    Factory function to create skill gap analyzer.
    
    Args:
        config_path: Path to system configuration
        
    Returns:
        Initialized SkillGapAnalyzer
    """
    return SkillGapAnalyzer(config_path=config_path)
