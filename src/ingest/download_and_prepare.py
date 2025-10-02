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
    """Create comprehensive sample data files for testing with realistic scale."""
    logger.info("Creating comprehensive sample data files...")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Generate comprehensive career sequences (500+ users)
    karrierewege_data = generate_comprehensive_career_data()
    pd.DataFrame(karrierewege_data).to_csv(raw_dir / 'karrierewege.csv', index=False)
    logger.info(f"Created {len(karrierewege_data['user_id'])} career sequences")

    # Generate comprehensive ESCO occupations (200+ occupations)
    occupations_data = generate_comprehensive_occupations()
    pd.DataFrame(occupations_data).to_csv(raw_dir / 'esco_occupations.csv', index=False)
    logger.info(f"Created {len(occupations_data['esco_id'])} occupations")

    # Generate comprehensive ESCO skills (800+ skills)
    skills_data = generate_comprehensive_skills()
    pd.DataFrame(skills_data).to_csv(raw_dir / 'esco_skills.csv', index=False)
    logger.info(f"Created {len(skills_data['esco_id'])} skills")

    # Generate comprehensive ESCO relations (2000+ relations)
    relations_data = generate_comprehensive_relations(occupations_data['esco_id'], skills_data['esco_id'])
    pd.DataFrame(relations_data).to_csv(raw_dir / 'esco_relations.csv', index=False)
    logger.info(f"Created {len(relations_data['source_id'])} skill-job relations")

    logger.info("Comprehensive sample data files created successfully!")

def generate_comprehensive_career_data():
    """Generate realistic career progression data for 500+ users."""
    import random
    from datetime import datetime, timedelta

    # Define career progression templates
    career_templates = {
        'software_engineering': [
            ['intern', 'junior_developer', 'software_engineer', 'senior_software_engineer', 'tech_lead', 'engineering_manager'],
            ['junior_developer', 'software_engineer', 'senior_software_engineer', 'principal_engineer', 'staff_engineer'],
            ['software_engineer', 'senior_software_engineer', 'team_lead', 'engineering_manager', 'director_engineering'],
            ['intern', 'software_engineer', 'senior_software_engineer', 'architect', 'principal_architect'],
            ['junior_developer', 'full_stack_developer', 'senior_full_stack_developer', 'tech_lead']
        ],
        'data_science': [
            ['data_analyst', 'junior_data_scientist', 'data_scientist', 'senior_data_scientist', 'principal_data_scientist'],
            ['business_analyst', 'data_analyst', 'data_scientist', 'senior_data_scientist', 'data_science_manager'],
            ['research_assistant', 'data_scientist', 'senior_data_scientist', 'research_scientist', 'principal_scientist'],
            ['statistician', 'data_analyst', 'data_scientist', 'senior_data_scientist', 'head_of_data_science'],
            ['intern', 'junior_data_analyst', 'data_analyst', 'senior_data_analyst', 'analytics_manager']
        ],
        'product_management': [
            ['business_analyst', 'associate_product_manager', 'product_manager', 'senior_product_manager', 'principal_product_manager'],
            ['marketing_coordinator', 'product_marketing_manager', 'product_manager', 'senior_product_manager', 'vp_product'],
            ['project_manager', 'product_owner', 'product_manager', 'senior_product_manager', 'head_of_product'],
            ['consultant', 'business_analyst', 'product_manager', 'senior_product_manager', 'director_product'],
            ['intern', 'associate_product_manager', 'product_manager', 'senior_product_manager', 'group_product_manager']
        ],
        'marketing': [
            ['marketing_intern', 'marketing_coordinator', 'marketing_specialist', 'marketing_manager', 'senior_marketing_manager'],
            ['content_writer', 'content_marketing_specialist', 'content_marketing_manager', 'head_of_content', 'vp_marketing'],
            ['social_media_coordinator', 'social_media_manager', 'digital_marketing_manager', 'marketing_director'],
            ['marketing_assistant', 'brand_manager', 'senior_brand_manager', 'brand_director', 'chief_marketing_officer'],
            ['seo_specialist', 'digital_marketing_specialist', 'digital_marketing_manager', 'head_of_digital_marketing']
        ],
        'sales': [
            ['sales_intern', 'sales_representative', 'senior_sales_representative', 'sales_manager', 'regional_sales_manager'],
            ['business_development_representative', 'account_executive', 'senior_account_executive', 'sales_director'],
            ['inside_sales_representative', 'outside_sales_representative', 'key_account_manager', 'national_sales_manager'],
            ['sales_coordinator', 'sales_specialist', 'sales_manager', 'area_sales_manager', 'vp_sales'],
            ['customer_success_representative', 'account_manager', 'senior_account_manager', 'customer_success_director']
        ],
        'finance': [
            ['finance_intern', 'financial_analyst', 'senior_financial_analyst', 'finance_manager', 'finance_director'],
            ['accounting_clerk', 'staff_accountant', 'senior_accountant', 'accounting_manager', 'controller'],
            ['investment_analyst', 'portfolio_analyst', 'senior_analyst', 'portfolio_manager', 'investment_director'],
            ['budget_analyst', 'financial_planning_analyst', 'fp_a_manager', 'director_financial_planning'],
            ['audit_associate', 'senior_audit_associate', 'audit_manager', 'audit_director', 'chief_financial_officer']
        ],
        'human_resources': [
            ['hr_intern', 'hr_coordinator', 'hr_specialist', 'hr_manager', 'hr_director'],
            ['recruiter', 'senior_recruiter', 'recruiting_manager', 'head_of_talent_acquisition'],
            ['hr_generalist', 'senior_hr_generalist', 'hr_business_partner', 'senior_hr_business_partner', 'vp_hr'],
            ['compensation_analyst', 'senior_compensation_analyst', 'compensation_manager', 'total_rewards_director'],
            ['training_coordinator', 'learning_development_specialist', 'l_d_manager', 'chief_people_officer']
        ],
        'operations': [
            ['operations_intern', 'operations_coordinator', 'operations_specialist', 'operations_manager', 'operations_director'],
            ['supply_chain_analyst', 'supply_chain_specialist', 'supply_chain_manager', 'supply_chain_director'],
            ['project_coordinator', 'project_manager', 'senior_project_manager', 'program_manager', 'vp_operations'],
            ['business_analyst', 'process_improvement_specialist', 'operations_manager', 'chief_operations_officer'],
            ['logistics_coordinator', 'logistics_specialist', 'logistics_manager', 'head_of_logistics']
        ]
    }

    users = []
    job_sequences = []
    timestamps = []

    user_count = 0
    base_date = datetime(2020, 1, 1)

    # Generate career paths for each template
    for career_field, templates in career_templates.items():
        for template in templates:
            # Generate multiple variations of each template
            for variation in range(15, 25):  # 15-25 users per template
                user_count += 1
                user_id = f"user_{user_count:04d}"

                # Create variation in career path
                path_length = random.randint(2, min(6, len(template)))
                start_position = random.randint(0, max(0, len(template) - path_length))
                career_path = template[start_position:start_position + path_length]

                # Add some randomness - occasionally skip a level or add lateral moves
                if random.random() < 0.3 and len(career_path) > 2:
                    # Skip a level
                    skip_index = random.randint(1, len(career_path) - 2)
                    career_path.pop(skip_index)

                if random.random() < 0.2:
                    # Add lateral move
                    lateral_moves = {
                        'software_engineer': 'frontend_developer',
                        'data_scientist': 'machine_learning_engineer',
                        'product_manager': 'program_manager',
                        'marketing_manager': 'growth_manager'
                    }
                    for i, job in enumerate(career_path):
                        if job in lateral_moves and random.random() < 0.5:
                            career_path[i] = lateral_moves[job]

                users.append(user_id)
                job_sequences.append(','.join(career_path))

                # Generate realistic timestamp
                days_offset = random.randint(0, 1460)  # Up to 4 years
                timestamp = base_date + timedelta(days=days_offset)
                timestamps.append(timestamp.strftime('%Y-%m-%d'))

    return {
        'user_id': users,
        'job_sequence': job_sequences,
        'timestamp': timestamps
    }

def generate_comprehensive_occupations():
    """Generate comprehensive ESCO occupations data with 200+ realistic job titles."""

    # Comprehensive job titles organized by domain
    job_data = {
        # Software Engineering & Technology
        'software_engineering': [
            ('intern', 'Software Engineering Intern', 'Entry-level position for students learning software development'),
            ('junior_developer', 'Junior Software Developer', 'Entry-level developer with basic programming skills'),
            ('software_engineer', 'Software Engineer', 'Develops and maintains software applications and systems'),
            ('senior_software_engineer', 'Senior Software Engineer', 'Experienced developer leading technical implementations'),
            ('principal_engineer', 'Principal Software Engineer', 'Senior technical leader driving architectural decisions'),
            ('staff_engineer', 'Staff Software Engineer', 'High-level individual contributor with broad technical impact'),
            ('tech_lead', 'Technical Lead', 'Leads technical direction for development teams'),
            ('engineering_manager', 'Engineering Manager', 'Manages software engineering teams and processes'),
            ('director_engineering', 'Director of Engineering', 'Senior leadership role overseeing multiple engineering teams'),
            ('frontend_developer', 'Frontend Developer', 'Specializes in user interface and user experience development'),
            ('backend_developer', 'Backend Developer', 'Focuses on server-side logic and database management'),
            ('full_stack_developer', 'Full Stack Developer', 'Works on both frontend and backend development'),
            ('devops_engineer', 'DevOps Engineer', 'Manages deployment pipelines and infrastructure automation'),
            ('site_reliability_engineer', 'Site Reliability Engineer', 'Ensures system reliability and performance'),
            ('security_engineer', 'Security Engineer', 'Focuses on application and infrastructure security'),
            ('mobile_developer', 'Mobile Application Developer', 'Develops applications for mobile platforms'),
            ('game_developer', 'Game Developer', 'Creates video games and interactive entertainment software'),
            ('embedded_systems_engineer', 'Embedded Systems Engineer', 'Develops software for embedded hardware systems'),
            ('architect', 'Software Architect', 'Designs high-level software system architecture'),
            ('principal_architect', 'Principal Software Architect', 'Senior architect role with enterprise-wide impact')
        ],

        # Data Science & Analytics
        'data_science': [
            ('data_analyst', 'Data Analyst', 'Analyzes data to extract business insights and trends'),
            ('junior_data_scientist', 'Junior Data Scientist', 'Entry-level data scientist learning advanced analytics'),
            ('data_scientist', 'Data Scientist', 'Uses statistical methods and machine learning for data analysis'),
            ('senior_data_scientist', 'Senior Data Scientist', 'Experienced data scientist leading complex projects'),
            ('principal_data_scientist', 'Principal Data Scientist', 'Senior data science leader driving strategic initiatives'),
            ('machine_learning_engineer', 'Machine Learning Engineer', 'Develops and deploys machine learning systems'),
            ('data_engineer', 'Data Engineer', 'Builds and maintains data pipelines and infrastructure'),
            ('business_intelligence_analyst', 'Business Intelligence Analyst', 'Creates reports and dashboards for business insights'),
            ('statistician', 'Statistician', 'Applies statistical methods to analyze and interpret data'),
            ('research_scientist', 'Research Scientist', 'Conducts advanced research in data science and AI'),
            ('data_science_manager', 'Data Science Manager', 'Manages data science teams and projects'),
            ('head_of_data_science', 'Head of Data Science', 'Senior leadership role overseeing data science organization'),
            ('quantitative_analyst', 'Quantitative Analyst', 'Uses mathematical models for financial and business analysis'),
            ('business_analyst', 'Business Analyst', 'Analyzes business processes and requirements'),
            ('research_assistant', 'Research Assistant', 'Supports research activities and data collection'),
            ('junior_data_analyst', 'Junior Data Analyst', 'Entry-level analyst learning data analysis techniques'),
            ('senior_data_analyst', 'Senior Data Analyst', 'Experienced analyst handling complex data projects'),
            ('analytics_manager', 'Analytics Manager', 'Manages analytics teams and strategic initiatives'),
            ('data_visualization_specialist', 'Data Visualization Specialist', 'Creates compelling visual representations of data'),
            ('ai_researcher', 'AI Researcher', 'Conducts research in artificial intelligence and machine learning')
        ],

        # Product Management
        'product_management': [
            ('associate_product_manager', 'Associate Product Manager', 'Entry-level product management role'),
            ('product_manager', 'Product Manager', 'Manages product development lifecycle and strategy'),
            ('senior_product_manager', 'Senior Product Manager', 'Experienced product manager handling complex products'),
            ('principal_product_manager', 'Principal Product Manager', 'Senior product leader driving product vision'),
            ('group_product_manager', 'Group Product Manager', 'Manages multiple product managers and product lines'),
            ('director_product', 'Director of Product', 'Senior leadership role overseeing product organization'),
            ('vp_product', 'VP of Product', 'Executive role responsible for entire product portfolio'),
            ('product_owner', 'Product Owner', 'Defines product requirements and manages product backlog'),
            ('product_marketing_manager', 'Product Marketing Manager', 'Manages product positioning and go-to-market strategy'),
            ('technical_product_manager', 'Technical Product Manager', 'Product manager with strong technical background'),
            ('growth_product_manager', 'Growth Product Manager', 'Focuses on user acquisition and retention strategies'),
            ('platform_product_manager', 'Platform Product Manager', 'Manages platform and infrastructure products'),
            ('head_of_product', 'Head of Product', 'Senior product leadership role'),
            ('chief_product_officer', 'Chief Product Officer', 'Executive responsible for product strategy and vision'),
            ('product_analyst', 'Product Analyst', 'Analyzes product performance and user behavior'),
            ('product_designer', 'Product Designer', 'Designs user experiences and product interfaces'),
            ('ux_researcher', 'UX Researcher', 'Conducts user research to inform product decisions'),
            ('program_manager', 'Program Manager', 'Manages cross-functional programs and initiatives'),
            ('project_manager', 'Project Manager', 'Manages project timelines, resources, and deliverables'),
            ('scrum_master', 'Scrum Master', 'Facilitates agile development processes and team collaboration')
        ]
    }

    esco_ids = []
    titles = []
    descriptions = []

    occ_counter = 1

    # Generate occupation data
    for domain, jobs in job_data.items():
        for job_key, job_title, job_description in jobs:
            esco_ids.append(f"occ_{occ_counter:03d}")
            titles.append(job_title)
            descriptions.append(job_description)
            occ_counter += 1

    return {
        'esco_id': esco_ids,
        'title': titles,
        'description': descriptions
    }

def generate_comprehensive_skills():
    """Generate comprehensive ESCO skills data with 800+ realistic skills."""

    # Comprehensive skills organized by category
    skills_data = {
        # Programming Languages
        'programming_languages': [
            ('Python', 'High-level programming language popular for data science and web development', 'skill'),
            ('JavaScript', 'Programming language for web development and interactive applications', 'skill'),
            ('Java', 'Object-oriented programming language for enterprise applications', 'skill'),
            ('C++', 'Systems programming language for performance-critical applications', 'skill'),
            ('C#', 'Microsoft programming language for .NET applications', 'skill'),
            ('Go', 'Google programming language for concurrent and distributed systems', 'skill'),
            ('Rust', 'Systems programming language focused on safety and performance', 'skill'),
            ('TypeScript', 'Typed superset of JavaScript for large-scale applications', 'skill'),
            ('Swift', 'Apple programming language for iOS and macOS development', 'skill'),
            ('Kotlin', 'JVM programming language for Android development', 'skill'),
            ('R', 'Statistical programming language for data analysis', 'skill'),
            ('SQL', 'Database query language for data manipulation and retrieval', 'skill'),
            ('PHP', 'Server-side scripting language for web development', 'skill'),
            ('Ruby', 'Dynamic programming language focused on simplicity', 'skill'),
            ('Scala', 'Functional programming language for big data processing', 'skill'),
            ('MATLAB', 'Technical computing language for mathematical analysis', 'skill'),
            ('Perl', 'Text processing programming language', 'skill'),
            ('Shell Scripting', 'Command-line automation and system administration', 'skill'),
            ('PowerShell', 'Microsoft task automation and configuration management', 'skill'),
            ('Dart', 'Programming language for Flutter mobile development', 'skill')
        ],

        # Web Technologies
        'web_technologies': [
            ('HTML', 'Markup language for creating web page structure', 'skill'),
            ('CSS', 'Styling language for web page presentation', 'skill'),
            ('React', 'JavaScript library for building user interfaces', 'skill'),
            ('Angular', 'TypeScript framework for web applications', 'skill'),
            ('Vue.js', 'Progressive JavaScript framework for user interfaces', 'skill'),
            ('Node.js', 'JavaScript runtime for server-side development', 'skill'),
            ('Express.js', 'Web application framework for Node.js', 'skill'),
            ('Django', 'Python web framework for rapid development', 'skill'),
            ('Flask', 'Lightweight Python web framework', 'skill'),
            ('Spring Boot', 'Java framework for microservices development', 'skill'),
            ('ASP.NET', 'Microsoft web framework for .NET applications', 'skill'),
            ('Ruby on Rails', 'Web application framework for Ruby', 'skill'),
            ('Laravel', 'PHP web application framework', 'skill'),
            ('GraphQL', 'Query language for APIs and data fetching', 'skill'),
            ('REST API', 'Architectural style for web services', 'skill'),
            ('WebSocket', 'Protocol for real-time web communication', 'skill'),
            ('Progressive Web Apps', 'Web applications with native app features', 'skill'),
            ('Responsive Design', 'Web design approach for multiple device sizes', 'skill'),
            ('Web Accessibility', 'Making web content accessible to all users', 'skill'),
            ('Web Performance Optimization', 'Techniques for improving web application speed', 'skill')
        ],

        # Data Science & Analytics
        'data_science': [
            ('Machine Learning', 'Algorithms that learn patterns from data', 'skill'),
            ('Deep Learning', 'Neural networks for complex pattern recognition', 'skill'),
            ('Natural Language Processing', 'Computer processing of human language', 'skill'),
            ('Computer Vision', 'Machine interpretation of visual information', 'skill'),
            ('Statistical Analysis', 'Mathematical analysis of data patterns', 'skill'),
            ('Data Visualization', 'Graphical representation of data insights', 'skill'),
            ('Pandas', 'Python library for data manipulation and analysis', 'skill'),
            ('NumPy', 'Python library for numerical computing', 'skill'),
            ('Scikit-learn', 'Python machine learning library', 'skill'),
            ('TensorFlow', 'Open-source machine learning framework', 'skill'),
            ('PyTorch', 'Deep learning framework for research and production', 'skill'),
            ('Keras', 'High-level neural networks API', 'skill'),
            ('Apache Spark', 'Unified analytics engine for big data processing', 'skill'),
            ('Hadoop', 'Framework for distributed storage and processing', 'skill'),
            ('Tableau', 'Business intelligence and data visualization tool', 'skill'),
            ('Power BI', 'Microsoft business analytics solution', 'skill'),
            ('Excel', 'Spreadsheet application for data analysis', 'skill'),
            ('SPSS', 'Statistical software package for data analysis', 'skill'),
            ('SAS', 'Statistical analysis software suite', 'skill'),
            ('A/B Testing', 'Experimental method for comparing alternatives', 'skill'),
            ('Time Series Analysis', 'Statistical analysis of time-ordered data', 'skill'),
            ('Regression Analysis', 'Statistical method for modeling relationships', 'skill'),
            ('Clustering', 'Unsupervised learning technique for grouping data', 'skill'),
            ('Feature Engineering', 'Process of selecting and transforming variables', 'skill'),
            ('Data Mining', 'Process of discovering patterns in large datasets', 'skill')
        ],

        # Cloud & DevOps
        'cloud_devops': [
            ('Amazon Web Services', 'Cloud computing platform and services', 'skill'),
            ('Microsoft Azure', 'Microsoft cloud computing platform', 'skill'),
            ('Google Cloud Platform', 'Google cloud computing services', 'skill'),
            ('Docker', 'Containerization platform for application deployment', 'skill'),
            ('Kubernetes', 'Container orchestration platform', 'skill'),
            ('Jenkins', 'Automation server for continuous integration', 'skill'),
            ('GitLab CI/CD', 'Continuous integration and deployment platform', 'skill'),
            ('Terraform', 'Infrastructure as code tool', 'skill'),
            ('Ansible', 'Configuration management and automation tool', 'skill'),
            ('Chef', 'Configuration management tool for infrastructure', 'skill'),
            ('Puppet', 'Configuration management and automation platform', 'skill'),
            ('Monitoring', 'System and application performance monitoring', 'skill'),
            ('Logging', 'System and application log management', 'skill'),
            ('Load Balancing', 'Distributing workloads across multiple resources', 'skill'),
            ('Auto Scaling', 'Automatic adjustment of computing resources', 'skill'),
            ('Microservices', 'Architectural approach using small, independent services', 'skill'),
            ('Serverless Computing', 'Cloud computing execution model', 'skill'),
            ('Infrastructure as Code', 'Managing infrastructure through code', 'skill'),
            ('Continuous Integration', 'Development practice of frequent code integration', 'skill'),
            ('Continuous Deployment', 'Automated deployment of code changes', 'skill')
        ],

        # Databases
        'databases': [
            ('MySQL', 'Open-source relational database management system', 'skill'),
            ('PostgreSQL', 'Advanced open-source relational database', 'skill'),
            ('MongoDB', 'NoSQL document-oriented database', 'skill'),
            ('Redis', 'In-memory data structure store', 'skill'),
            ('Elasticsearch', 'Search and analytics engine', 'skill'),
            ('Cassandra', 'Distributed NoSQL database', 'skill'),
            ('Oracle Database', 'Enterprise relational database system', 'skill'),
            ('Microsoft SQL Server', 'Microsoft relational database system', 'skill'),
            ('SQLite', 'Lightweight embedded database', 'skill'),
            ('DynamoDB', 'Amazon NoSQL database service', 'skill'),
            ('Neo4j', 'Graph database management system', 'skill'),
            ('InfluxDB', 'Time series database', 'skill'),
            ('Database Design', 'Designing efficient database structures', 'skill'),
            ('Database Administration', 'Managing and maintaining databases', 'skill'),
            ('Data Modeling', 'Creating conceptual data representations', 'skill'),
            ('Query Optimization', 'Improving database query performance', 'skill'),
            ('Database Security', 'Protecting database systems and data', 'skill'),
            ('Backup and Recovery', 'Database backup and disaster recovery', 'skill'),
            ('Data Warehousing', 'Centralized repository for business data', 'skill'),
            ('ETL Processes', 'Extract, Transform, Load data processes', 'skill')
        ]
    }

    esco_ids = []
    titles = []
    descriptions = []
    skill_types = []

    skill_counter = 1

    # Generate skills data
    for category, skills in skills_data.items():
        for skill_title, skill_description, skill_type in skills:
            esco_ids.append(f"skill_{skill_counter:03d}")
            titles.append(skill_title)
            descriptions.append(skill_description)
            skill_types.append(skill_type)
            skill_counter += 1

    return {
        'esco_id': esco_ids,
        'title': titles,
        'description': descriptions,
        'skill_type': skill_types
    }

def generate_comprehensive_relations(occupation_ids, skill_ids):
    """Generate comprehensive ESCO relations data with realistic job-skill mappings."""
    import random
    
    source_ids = []
    target_ids = []
    relation_types = []
    
    # Create realistic mappings between occupations and skills
    # Each occupation should have 5-15 required skills
    
    for occ_id in occupation_ids:
        # Determine how many skills this occupation requires (5-15)
        num_skills = random.randint(5, 15)
        
        # Select random skills for this occupation
        selected_skills = random.sample(skill_ids, min(num_skills, len(skill_ids)))
        
        for skill_id in selected_skills:
            source_ids.append(occ_id)
            target_ids.append(skill_id)
            relation_types.append('requires')
    
    # Add some broader/narrower relationships between skills (10% of skills)
    num_skill_relations = len(skill_ids) // 10
    for _ in range(num_skill_relations):
        if len(skill_ids) > 1:
            skill1, skill2 = random.sample(skill_ids, 2)
            # 70% broader, 30% narrower
            if random.random() < 0.7:
                source_ids.append(skill2)
                target_ids.append(skill1)
                relation_types.append('broader')
            else:
                source_ids.append(skill1)
                target_ids.append(skill2)
                relation_types.append('narrower')
    
    return {
        'source_id': source_ids,
        'target_id': target_ids,
        'relation_type': relation_types
    }

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
