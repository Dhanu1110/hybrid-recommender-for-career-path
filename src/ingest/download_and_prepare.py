#!/usr/bin/env python3
"""
Data Ingestion Pipeline for Career Recommender System

This script processes raw Karrierewege and ESCO data into canonicalized Parquet files.

Expected input files:
- data/raw/karrierewege.csv: Career path data with columns [user_id, job_sequence, timestamp]
- data/raw/esco_occupations.csv: ESCO occupation data
- data/raw/esco_skills.csv: ESCO skills data
- data/raw/esco_relations.csv: ESCO relations data

Output files:
- data/processed/career_paths.parquet: Processed career sequences
- data/processed/esco_occupations.parquet: Canonicalized occupation data
- data/processed/esco_skills.parquet: Canonicalized skills data
- data/processed/esco_relations.parquet: Canonicalized relations data
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """Main pipeline for processing raw data into canonicalized format."""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # Create output directory if it doesn't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected input files
        self.expected_files = {
            'karrierewege': self.raw_data_dir / 'karrierewege.csv',
            'esco_occupations': self.raw_data_dir / 'esco_occupations.csv',
            'esco_skills': self.raw_data_dir / 'esco_skills.csv',
            'esco_relations': self.raw_data_dir / 'esco_relations.csv'
        }
        
    def validate_input_files(self) -> bool:
        """Validate that all required input files exist."""
        missing_files = []
        for name, path in self.expected_files.items():
            if not path.exists():
                missing_files.append(str(path))
                
        if missing_files:
            logger.error(f"Missing required input files: {missing_files}")
            logger.info("Please ensure the following files exist:")
            for name, path in self.expected_files.items():
                logger.info(f"  {name}: {path}")
            return False
        return True
    
    def process_karrierewege_data(self) -> pd.DataFrame:
        """
        Process Karrierewege career path data.
        
        Expected format:
        - user_id: Unique identifier for users
        - job_sequence: Comma-separated list of job titles/IDs
        - timestamp: When the career path was recorded
        """
        logger.info("Processing Karrierewege data...")
        
        try:
            df = pd.read_csv(self.expected_files['karrierewege'])
            logger.info(f"Loaded {len(df)} career path records")
            
            # Basic validation
            required_columns = ['user_id', 'job_sequence', 'timestamp']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Process job sequences
            df['job_sequence_list'] = df['job_sequence'].apply(
                lambda x: [job.strip() for job in str(x).split(',') if job.strip()]
            )
            df['sequence_length'] = df['job_sequence_list'].apply(len)
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter out invalid sequences
            df = df[df['sequence_length'] > 0]
            
            logger.info(f"Processed {len(df)} valid career paths")
            logger.info(f"Average sequence length: {df['sequence_length'].mean():.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing Karrierewege data: {e}")
            raise
    
    def process_esco_occupations(self) -> pd.DataFrame:
        """
        Process ESCO occupations data.
        
        Expected format:
        - esco_id: ESCO occupation identifier
        - title: Occupation title
        - description: Occupation description
        - isco_code: ISCO classification code (optional)
        """
        logger.info("Processing ESCO occupations...")
        
        try:
            df = pd.read_csv(self.expected_files['esco_occupations'])
            logger.info(f"Loaded {len(df)} occupation records")
            
            # Basic validation
            required_columns = ['esco_id', 'title']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean and standardize
            df['title'] = df['title'].str.strip()
            df['esco_id'] = df['esco_id'].str.strip()
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['esco_id'])
            
            logger.info(f"Processed {len(df)} unique occupations")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing ESCO occupations: {e}")
            raise
    
    def process_esco_skills(self) -> pd.DataFrame:
        """
        Process ESCO skills data.
        
        Expected format:
        - esco_id: ESCO skill identifier
        - title: Skill title
        - description: Skill description
        - skill_type: Type of skill (knowledge, skill, competence)
        """
        logger.info("Processing ESCO skills...")
        
        try:
            df = pd.read_csv(self.expected_files['esco_skills'])
            logger.info(f"Loaded {len(df)} skill records")
            
            # Basic validation
            required_columns = ['esco_id', 'title']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean and standardize
            df['title'] = df['title'].str.strip()
            df['esco_id'] = df['esco_id'].str.strip()
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['esco_id'])
            
            logger.info(f"Processed {len(df)} unique skills")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing ESCO skills: {e}")
            raise
    
    def process_esco_relations(self) -> pd.DataFrame:
        """
        Process ESCO relations data.
        
        Expected format:
        - source_id: Source ESCO ID
        - target_id: Target ESCO ID
        - relation_type: Type of relation (requires, related_to, broader, narrower)
        """
        logger.info("Processing ESCO relations...")
        
        try:
            df = pd.read_csv(self.expected_files['esco_relations'])
            logger.info(f"Loaded {len(df)} relation records")
            
            # Basic validation
            required_columns = ['source_id', 'target_id', 'relation_type']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean and standardize
            df['source_id'] = df['source_id'].str.strip()
            df['target_id'] = df['target_id'].str.strip()
            df['relation_type'] = df['relation_type'].str.strip().str.lower()
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['source_id', 'target_id', 'relation_type'])
            
            logger.info(f"Processed {len(df)} unique relations")
            logger.info(f"Relation types: {df['relation_type'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing ESCO relations: {e}")
            raise
    
    def save_processed_data(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Save processed dataframes to Parquet files."""
        logger.info("Saving processed data...")
        
        output_files = {
            'career_paths': 'career_paths.parquet',
            'esco_occupations': 'esco_occupations.parquet',
            'esco_skills': 'esco_skills.parquet',
            'esco_relations': 'esco_relations.parquet'
        }
        
        for name, filename in output_files.items():
            if name in dataframes:
                output_path = self.processed_data_dir / filename
                dataframes[name].to_parquet(output_path, index=False)
                logger.info(f"Saved {name} to {output_path}")
    
    def run(self) -> bool:
        """Run the complete data ingestion pipeline."""
        logger.info("Starting data ingestion pipeline...")
        
        try:
            # Validate input files
            if not self.validate_input_files():
                return False
            
            # Process each data source
            processed_data = {}
            
            # Process Karrierewege data
            processed_data['career_paths'] = self.process_karrierewege_data()
            
            # Process ESCO data
            processed_data['esco_occupations'] = self.process_esco_occupations()
            processed_data['esco_skills'] = self.process_esco_skills()
            processed_data['esco_relations'] = self.process_esco_relations()
            
            # Save processed data
            self.save_processed_data(processed_data)
            
            logger.info("Data ingestion pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def create_sample_data():
    """Create sample data files for testing."""
    logger.info("Creating sample data files...")
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample Karrierewege data
    karrierewege_data = {
        'user_id': ['user_001', 'user_002', 'user_003'],
        'job_sequence': [
            'software_engineer,senior_software_engineer,tech_lead',
            'data_analyst,data_scientist,senior_data_scientist',
            'marketing_assistant,marketing_specialist,marketing_manager'
        ],
        'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03']
    }
    pd.DataFrame(karrierewege_data).to_csv(raw_dir / 'karrierewege.csv', index=False)
    
    # Sample ESCO occupations
    occupations_data = {
        'esco_id': ['occ_001', 'occ_002', 'occ_003'],
        'title': ['Software Engineer', 'Data Scientist', 'Marketing Manager'],
        'description': ['Develops software applications', 'Analyzes data for insights', 'Manages marketing campaigns']
    }
    pd.DataFrame(occupations_data).to_csv(raw_dir / 'esco_occupations.csv', index=False)
    
    # Sample ESCO skills
    skills_data = {
        'esco_id': ['skill_001', 'skill_002', 'skill_003'],
        'title': ['Python Programming', 'Data Analysis', 'Digital Marketing'],
        'description': ['Programming in Python', 'Statistical data analysis', 'Online marketing strategies'],
        'skill_type': ['skill', 'skill', 'competence']
    }
    pd.DataFrame(skills_data).to_csv(raw_dir / 'esco_skills.csv', index=False)
    
    # Sample ESCO relations
    relations_data = {
        'source_id': ['occ_001', 'occ_002', 'occ_003'],
        'target_id': ['skill_001', 'skill_002', 'skill_003'],
        'relation_type': ['requires', 'requires', 'requires']
    }
    pd.DataFrame(relations_data).to_csv(raw_dir / 'esco_relations.csv', index=False)
    
    logger.info("Sample data files created successfully!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Data Ingestion Pipeline for Career Recommender')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create sample data files for testing')
    parser.add_argument('--raw-dir', default='data/raw',
                       help='Directory containing raw data files')
    parser.add_argument('--processed-dir', default='data/processed',
                       help='Directory for processed data files')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
        return
    
    # Run the pipeline
    pipeline = DataIngestionPipeline(args.raw_dir, args.processed_dir)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
