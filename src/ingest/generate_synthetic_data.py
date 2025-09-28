#!/usr/bin/env python3
"""
Enhanced synthetic data generator for career recommender system.
Allows generation of large-scale datasets with configurable parameters.
"""

import argparse
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_large_career_data(num_users: int = 5000, 
                              num_jobs: int = 500,
                              min_path_length: int = 2,
                              max_path_length: int = 8) -> Dict[str, List]:
    """Generate large-scale career progression data."""
    logger.info(f"Generating career data for {num_users} users with {num_jobs} job types...")
    
    # Extended career templates with more diversity
    career_templates = {
        'technology': [
            ['intern', 'junior_developer', 'software_engineer', 'senior_engineer', 'tech_lead', 'engineering_manager', 'director_engineering'],
            ['qa_intern', 'qa_engineer', 'senior_qa', 'qa_lead', 'qa_manager'],
            ['data_intern', 'data_analyst', 'data_scientist', 'senior_data_scientist', 'principal_data_scientist'],
            ['devops_intern', 'devops_engineer', 'senior_devops', 'devops_architect', 'platform_engineer'],
            ['ui_designer', 'ux_designer', 'senior_designer', 'design_lead', 'head_of_design'],
            ['product_intern', 'product_analyst', 'product_manager', 'senior_pm', 'director_product'],
            ['security_analyst', 'security_engineer', 'senior_security', 'security_architect', 'ciso'],
            ['mobile_developer', 'senior_mobile_dev', 'mobile_architect', 'mobile_lead'],
            ['frontend_developer', 'senior_frontend', 'frontend_architect', 'frontend_lead'],
            ['backend_developer', 'senior_backend', 'backend_architect', 'backend_lead'],
            ['fullstack_developer', 'senior_fullstack', 'fullstack_architect', 'engineering_lead'],
            ['ml_engineer', 'senior_ml_engineer', 'ml_architect', 'ai_research_lead'],
            ['cloud_engineer', 'cloud_architect', 'principal_cloud_architect', 'cloud_director'],
            ['systems_admin', 'systems_engineer', 'infrastructure_engineer', 'infrastructure_architect'],
            ['database_admin', 'database_engineer', 'data_architect', 'data_platform_lead']
        ],
        'business': [
            ['business_intern', 'business_analyst', 'senior_analyst', 'manager', 'director', 'vp'],
            ['sales_intern', 'sales_rep', 'senior_sales', 'sales_manager', 'sales_director'],
            ['marketing_intern', 'marketing_coordinator', 'marketing_manager', 'marketing_director'],
            ['hr_intern', 'hr_coordinator', 'hr_manager', 'hr_director', 'chief_people_officer'],
            ['finance_intern', 'financial_analyst', 'senior_analyst', 'finance_manager', 'cfo'],
            ['operations_intern', 'operations_coordinator', 'operations_manager', 'operations_director'],
            ['strategy_analyst', 'strategy_manager', 'strategy_director', 'chief_strategy_officer'],
            ['consultant', 'senior_consultant', 'principal_consultant', 'partner'],
            ['account_manager', 'senior_account_manager', 'key_account_manager', 'enterprise_sales'],
            ['business_development', 'senior_bd', 'bd_manager', 'bd_director'],
            ['project_coordinator', 'project_manager', 'senior_pm', 'program_manager', 'pmo_director'],
            ['customer_success', 'senior_cs', 'cs_manager', 'cs_director'],
            ['procurement_specialist', 'procurement_manager', 'procurement_director', 'chief_procurement_officer']
        ],
        'healthcare': [
            ['medical_intern', 'resident', 'attending_physician', 'senior_physician', 'department_head'],
            ['nursing_student', 'registered_nurse', 'senior_nurse', 'nurse_manager', 'chief_nursing_officer'],
            ['pharmacy_intern', 'pharmacist', 'senior_pharmacist', 'pharmacy_manager'],
            ['lab_technician', 'senior_lab_tech', 'lab_supervisor', 'lab_director'],
            ['medical_assistant', 'senior_medical_assistant', 'clinic_supervisor', 'clinic_manager'],
            ['therapist', 'senior_therapist', 'therapy_supervisor', 'therapy_director'],
            ['healthcare_admin', 'healthcare_manager', 'healthcare_director', 'hospital_administrator']
        ],
        'education': [
            ['teaching_assistant', 'teacher', 'senior_teacher', 'department_head', 'principal'],
            ['research_assistant', 'postdoc', 'assistant_professor', 'associate_professor', 'full_professor'],
            ['tutor', 'instructor', 'senior_instructor', 'curriculum_director'],
            ['education_coordinator', 'education_manager', 'education_director', 'superintendent'],
            ['librarian', 'senior_librarian', 'head_librarian', 'library_director']
        ],
        'finance': [
            ['finance_intern', 'analyst', 'senior_analyst', 'associate', 'vp', 'managing_director'],
            ['investment_analyst', 'portfolio_manager', 'senior_pm', 'investment_director'],
            ['risk_analyst', 'risk_manager', 'senior_risk_manager', 'chief_risk_officer'],
            ['compliance_analyst', 'compliance_manager', 'compliance_director', 'chief_compliance_officer'],
            ['trader', 'senior_trader', 'trading_manager', 'head_of_trading'],
            ['credit_analyst', 'credit_manager', 'credit_director', 'chief_credit_officer'],
            ['actuarial_analyst', 'actuary', 'senior_actuary', 'chief_actuary']
        ],
        'legal': [
            ['paralegal', 'junior_associate', 'associate', 'senior_associate', 'partner'],
            ['legal_intern', 'staff_attorney', 'senior_attorney', 'legal_director', 'general_counsel'],
            ['compliance_specialist', 'compliance_counsel', 'deputy_general_counsel', 'chief_legal_officer']
        ],
        'manufacturing': [
            ['production_worker', 'line_supervisor', 'production_manager', 'plant_manager'],
            ['quality_inspector', 'quality_engineer', 'quality_manager', 'quality_director'],
            ['maintenance_tech', 'maintenance_supervisor', 'maintenance_manager', 'facilities_director'],
            ['process_engineer', 'senior_process_engineer', 'process_manager', 'manufacturing_director'],
            ['supply_chain_analyst', 'supply_chain_manager', 'supply_chain_director', 'chief_supply_chain_officer']
        ],
        'retail': [
            ['sales_associate', 'shift_supervisor', 'assistant_manager', 'store_manager', 'district_manager'],
            ['cashier', 'customer_service', 'customer_service_manager', 'operations_manager'],
            ['merchandiser', 'senior_merchandiser', 'merchandising_manager', 'merchandising_director'],
            ['buyer', 'senior_buyer', 'buying_manager', 'buying_director']
        ]
    }
    
    users = []
    job_sequences = []
    timestamps = []
    user_count = 0
    
    base_date = datetime(2020, 1, 1)
    
    # Calculate how many users per template to reach target
    total_templates = sum(len(templates) for templates in career_templates.values())
    users_per_template = max(1, num_users // total_templates)
    
    # Generate career paths for each template
    for career_field, templates in career_templates.items():
        for template in templates:
            # Generate multiple variations of each template
            for variation in range(users_per_template):
                if user_count >= num_users:
                    break
                    
                user_count += 1
                user_id = f"user_{user_count:06d}"
                
                # Create variation in career path
                path_length = random.randint(min_path_length, min(max_path_length, len(template)))
                start_position = random.randint(0, max(0, len(template) - path_length))
                career_path = template[start_position:start_position + path_length]
                
                # Add some randomness - occasionally skip a level or add lateral moves
                if random.random() < 0.3 and len(career_path) > 2:
                    # Skip a level
                    skip_index = random.randint(1, len(career_path) - 2)
                    career_path.pop(skip_index)
                
                # Add cross-functional moves occasionally
                if random.random() < 0.15:
                    # Add a cross-functional role
                    cross_functional_roles = [
                        'project_manager', 'business_analyst', 'consultant', 
                        'product_manager', 'operations_manager'
                    ]
                    if len(career_path) > 1:
                        insert_pos = random.randint(1, len(career_path))
                        career_path.insert(insert_pos, random.choice(cross_functional_roles))
                
                users.append(user_id)
                job_sequences.append(','.join(career_path))
                
                # Generate realistic timestamp
                days_offset = random.randint(0, 1460)  # Up to 4 years
                timestamp = base_date + timedelta(days=days_offset)
                timestamps.append(timestamp.strftime('%Y-%m-%d'))
    
    # Fill remaining users with random combinations if needed
    while user_count < num_users:
        user_count += 1
        user_id = f"user_{user_count:06d}"
        
        # Pick random template from random field
        field = random.choice(list(career_templates.keys()))
        template = random.choice(career_templates[field])
        
        path_length = random.randint(min_path_length, min(max_path_length, len(template)))
        start_position = random.randint(0, max(0, len(template) - path_length))
        career_path = template[start_position:start_position + path_length]
        
        users.append(user_id)
        job_sequences.append(','.join(career_path))
        
        days_offset = random.randint(0, 1460)
        timestamp = base_date + timedelta(days=days_offset)
        timestamps.append(timestamp.strftime('%Y-%m-%d'))
    
    logger.info(f"Generated {len(users)} career sequences")
    return {
        'user_id': users,
        'job_sequence': job_sequences,
        'timestamp': timestamps
    }

def generate_large_occupations(num_jobs: int = 500) -> Dict[str, List]:
    """Generate large-scale ESCO occupations data."""
    logger.info(f"Generating {num_jobs} occupation records...")
    
    # Base job data with more comprehensive coverage
    job_domains = {
        'technology': [
            ('software_engineer', 'Software Engineer', 'Develops and maintains software applications'),
            ('data_scientist', 'Data Scientist', 'Analyzes complex data to derive business insights'),
            ('devops_engineer', 'DevOps Engineer', 'Manages development and operations infrastructure'),
            ('product_manager', 'Product Manager', 'Manages product development and strategy'),
            ('ux_designer', 'UX Designer', 'Designs user experiences for digital products'),
            ('security_engineer', 'Security Engineer', 'Implements and maintains cybersecurity measures'),
            ('mobile_developer', 'Mobile Developer', 'Develops applications for mobile platforms'),
            ('frontend_developer', 'Frontend Developer', 'Develops user-facing web applications'),
            ('backend_developer', 'Backend Developer', 'Develops server-side applications and APIs'),
            ('fullstack_developer', 'Full Stack Developer', 'Develops both frontend and backend systems'),
            ('ml_engineer', 'Machine Learning Engineer', 'Develops and deploys machine learning models'),
            ('cloud_architect', 'Cloud Architect', 'Designs cloud infrastructure and solutions'),
            ('database_admin', 'Database Administrator', 'Manages and maintains database systems'),
            ('systems_admin', 'Systems Administrator', 'Manages computer systems and networks'),
            ('qa_engineer', 'Quality Assurance Engineer', 'Tests software for quality and functionality')
        ],
        'business': [
            ('business_analyst', 'Business Analyst', 'Analyzes business processes and requirements'),
            ('sales_manager', 'Sales Manager', 'Manages sales teams and strategies'),
            ('marketing_manager', 'Marketing Manager', 'Develops and executes marketing campaigns'),
            ('hr_manager', 'Human Resources Manager', 'Manages human resources policies and practices'),
            ('finance_manager', 'Finance Manager', 'Manages financial planning and analysis'),
            ('operations_manager', 'Operations Manager', 'Oversees daily business operations'),
            ('strategy_consultant', 'Strategy Consultant', 'Provides strategic business advice'),
            ('account_manager', 'Account Manager', 'Manages client relationships and accounts'),
            ('project_manager', 'Project Manager', 'Plans and executes projects'),
            ('business_development', 'Business Development Manager', 'Identifies and develops new business opportunities')
        ]
        # Add more domains as needed...
    }
    
    esco_ids = []
    titles = []
    descriptions = []
    occ_counter = 1
    
    # Generate base occupations
    for domain, jobs in job_domains.items():
        for job_key, job_title, job_description in jobs:
            esco_ids.append(f"occ_{occ_counter:05d}")
            titles.append(job_title)
            descriptions.append(job_description)
            occ_counter += 1
    
    # Generate additional synthetic occupations to reach target
    job_prefixes = ['Senior', 'Junior', 'Lead', 'Principal', 'Associate', 'Assistant', 'Chief', 'Head of']
    job_suffixes = ['Specialist', 'Coordinator', 'Analyst', 'Manager', 'Director', 'Officer', 'Engineer', 'Developer']
    job_areas = ['Technology', 'Business', 'Finance', 'Marketing', 'Operations', 'Strategy', 'Product', 'Data']
    
    while occ_counter <= num_jobs:
        prefix = random.choice(job_prefixes)
        area = random.choice(job_areas)
        suffix = random.choice(job_suffixes)
        
        title = f"{prefix} {area} {suffix}"
        description = f"Professional responsible for {area.lower()}-related {suffix.lower()} activities"
        
        esco_ids.append(f"occ_{occ_counter:05d}")
        titles.append(title)
        descriptions.append(description)
        occ_counter += 1
    
    return {
        'esco_id': esco_ids,
        'title': titles,
        'description': descriptions
    }

def generate_large_skills(num_skills: int = 2000) -> Dict[str, List]:
    """Generate large-scale ESCO skills data."""
    logger.info(f"Generating {num_skills} skill records...")

    # Comprehensive skill categories
    skill_categories = {
        'programming': [
            'Python Programming', 'Java Programming', 'JavaScript Programming', 'C++ Programming',
            'C# Programming', 'Go Programming', 'Rust Programming', 'Swift Programming',
            'Kotlin Programming', 'TypeScript Programming', 'PHP Programming', 'Ruby Programming',
            'Scala Programming', 'R Programming', 'MATLAB Programming', 'SQL Programming',
            'HTML/CSS', 'React Development', 'Angular Development', 'Vue.js Development',
            'Node.js Development', 'Django Framework', 'Flask Framework', 'Spring Framework',
            'Express.js Framework', 'Laravel Framework', 'Ruby on Rails', '.NET Framework'
        ],
        'data_science': [
            'Data Analysis', 'Statistical Analysis', 'Machine Learning', 'Deep Learning',
            'Natural Language Processing', 'Computer Vision', 'Data Visualization',
            'Predictive Modeling', 'A/B Testing', 'Experimental Design', 'Big Data Analytics',
            'Time Series Analysis', 'Regression Analysis', 'Classification Algorithms',
            'Clustering Analysis', 'Neural Networks', 'Random Forest', 'Support Vector Machines',
            'Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'PyTorch', 'Keras',
            'Tableau', 'Power BI', 'D3.js', 'Matplotlib', 'Seaborn', 'Plotly'
        ],
        'cloud_devops': [
            'Amazon Web Services', 'Microsoft Azure', 'Google Cloud Platform', 'Docker',
            'Kubernetes', 'Jenkins', 'GitLab CI/CD', 'GitHub Actions', 'Terraform',
            'Ansible', 'Chef', 'Puppet', 'Vagrant', 'Linux Administration', 'Windows Server',
            'Network Administration', 'Load Balancing', 'Auto Scaling', 'Monitoring',
            'Log Management', 'Infrastructure as Code', 'Microservices Architecture',
            'Serverless Computing', 'Container Orchestration', 'Service Mesh'
        ],
        'business': [
            'Project Management', 'Agile Methodology', 'Scrum Framework', 'Kanban',
            'Strategic Planning', 'Business Analysis', 'Requirements Gathering',
            'Stakeholder Management', 'Risk Management', 'Change Management',
            'Process Improvement', 'Lean Six Sigma', 'Financial Analysis',
            'Budget Planning', 'Market Research', 'Competitive Analysis',
            'Customer Relationship Management', 'Sales Strategy', 'Marketing Strategy',
            'Digital Marketing', 'Content Marketing', 'Social Media Marketing',
            'Email Marketing', 'SEO/SEM', 'Brand Management', 'Product Management'
        ],
        'soft_skills': [
            'Leadership', 'Team Management', 'Communication', 'Public Speaking',
            'Presentation Skills', 'Negotiation', 'Conflict Resolution', 'Problem Solving',
            'Critical Thinking', 'Decision Making', 'Time Management', 'Multitasking',
            'Adaptability', 'Creativity', 'Innovation', 'Emotional Intelligence',
            'Mentoring', 'Coaching', 'Cross-functional Collaboration', 'Remote Work',
            'Cultural Sensitivity', 'Customer Service', 'Active Listening', 'Empathy'
        ],
        'design': [
            'User Experience Design', 'User Interface Design', 'Graphic Design',
            'Web Design', 'Mobile App Design', 'Prototyping', 'Wireframing',
            'Design Thinking', 'User Research', 'Usability Testing', 'Information Architecture',
            'Interaction Design', 'Visual Design', 'Typography', 'Color Theory',
            'Adobe Creative Suite', 'Figma', 'Sketch', 'InVision', 'Adobe XD',
            'Photoshop', 'Illustrator', 'After Effects', 'Premiere Pro'
        ],
        'security': [
            'Cybersecurity', 'Information Security', 'Network Security', 'Application Security',
            'Cloud Security', 'Penetration Testing', 'Vulnerability Assessment',
            'Security Auditing', 'Incident Response', 'Forensics', 'Risk Assessment',
            'Compliance Management', 'Identity Management', 'Access Control',
            'Encryption', 'Firewall Management', 'SIEM Tools', 'Security Monitoring'
        ]
    }

    esco_ids = []
    titles = []
    descriptions = []
    skill_counter = 1

    # Generate base skills
    for category, skills in skill_categories.items():
        for skill in skills:
            esco_ids.append(f"skill_{skill_counter:05d}")
            titles.append(skill)
            descriptions.append(f"Professional skill in {skill.lower()}")
            skill_counter += 1

    # Generate additional synthetic skills to reach target
    skill_modifiers = ['Advanced', 'Intermediate', 'Basic', 'Expert', 'Professional']
    skill_types = ['Analysis', 'Development', 'Management', 'Strategy', 'Implementation', 'Optimization']
    skill_domains = ['Business', 'Technical', 'Creative', 'Analytical', 'Strategic', 'Operational']

    while skill_counter <= num_skills:
        modifier = random.choice(skill_modifiers)
        domain = random.choice(skill_domains)
        skill_type = random.choice(skill_types)

        title = f"{modifier} {domain} {skill_type}"
        description = f"Skill in {modifier.lower()} {domain.lower()} {skill_type.lower()}"

        esco_ids.append(f"skill_{skill_counter:05d}")
        titles.append(title)
        descriptions.append(description)
        skill_counter += 1

    return {
        'esco_id': esco_ids,
        'title': titles,
        'description': descriptions
    }

def generate_large_relations(occupation_ids: List[str], skill_ids: List[str],
                           relations_per_job: int = 8) -> Dict[str, List]:
    """Generate large-scale ESCO relations data."""
    total_relations = len(occupation_ids) * relations_per_job
    logger.info(f"Generating {total_relations} skill-job relations...")

    source_ids = []
    target_ids = []
    relation_types = []

    for occ_id in occupation_ids:
        # Randomly select skills for this occupation
        selected_skills = random.sample(skill_ids, min(relations_per_job, len(skill_ids)))

        for skill_id in selected_skills:
            source_ids.append(occ_id)
            target_ids.append(skill_id)
            # Vary relation types
            relation_types.append(random.choice(['requires', 'benefits_from', 'related_to']))

    return {
        'source_id': source_ids,
        'target_id': target_ids,
        'relation_type': relation_types
    }

def create_large_sample_data(num_users: int = 5000, num_jobs: int = 500,
                           num_skills: int = 2000, relations_per_job: int = 8):
    """Create large-scale sample data files for testing."""
    logger.info(f"Creating large sample data: {num_users} users, {num_jobs} jobs, {num_skills} skills...")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Generate large career sequences
    karrierewege_data = generate_large_career_data(num_users)
    pd.DataFrame(karrierewege_data).to_csv(raw_dir / 'karrierewege.csv', index=False)
    logger.info(f"Created {len(karrierewege_data['user_id'])} career sequences")

    # Generate large ESCO occupations
    occupations_data = generate_large_occupations(num_jobs)
    pd.DataFrame(occupations_data).to_csv(raw_dir / 'esco_occupations.csv', index=False)
    logger.info(f"Created {len(occupations_data['esco_id'])} occupations")

    # Generate large ESCO skills
    skills_data = generate_large_skills(num_skills)
    pd.DataFrame(skills_data).to_csv(raw_dir / 'esco_skills.csv', index=False)
    logger.info(f"Created {len(skills_data['esco_id'])} skills")

    # Generate large ESCO relations
    relations_data = generate_large_relations(
        occupations_data['esco_id'],
        skills_data['esco_id'],
        relations_per_job
    )
    pd.DataFrame(relations_data).to_csv(raw_dir / 'esco_relations.csv', index=False)
    logger.info(f"Created {len(relations_data['source_id'])} skill-job relations")

    logger.info("Large-scale sample data files created successfully!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate large-scale synthetic data for Career Recommender')
    parser.add_argument('--num-users', type=int, default=5000,
                       help='Number of users to generate (default: 5000)')
    parser.add_argument('--num-jobs', type=int, default=500,
                       help='Number of job types to generate (default: 500)')
    parser.add_argument('--num-skills', type=int, default=2000,
                       help='Number of skills to generate (default: 2000)')
    parser.add_argument('--relations-per-job', type=int, default=8,
                       help='Average number of skills per job (default: 8)')
    parser.add_argument('--output-dir', default='data/raw',
                       help='Output directory for generated data')

    args = parser.parse_args()

    # Create large sample data
    create_large_sample_data(
        num_users=args.num_users,
        num_jobs=args.num_jobs,
        num_skills=args.num_skills,
        relations_per_job=args.relations_per_job
    )

if __name__ == "__main__":
    main()
