#!/usr/bin/env python3
"""
Streamlit Web Application for Career Recommender System

This app provides a user-friendly interface for the three-stage career recommendation system:
1. User onboarding (job history + skills input)
2. Career path recommendations with feasibility scores
3. Learning resource recommendations
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Set

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

# Import our modules
try:
    from ingest.download_and_prepare import create_sample_data, DataIngestionPipeline
    from ingest.esco_loader import create_esco_loader
    from ingest.text_to_esco_mapper import create_text_mapper
    from models.bert4rec import BERT4RecConfig, create_bert4rec_model
    from models.career_path_recommender import create_career_recommender
    from models.train_path_model import create_synthetic_data
    from reasoner.skill_gap import create_skill_gap_analyzer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Career Path Recommender",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_initialized' not in st.session_state:
    st.session_state.data_initialized = False
if 'esco_loader' not in st.session_state:
    st.session_state.esco_loader = None
if 'skill_analyzer' not in st.session_state:
    st.session_state.skill_analyzer = None
if 'recommender' not in st.session_state:
    st.session_state.recommender = None

def initialize_data():
    """Initialize the data and models."""
    with st.spinner("Initializing data and models..."):
        try:
            # Create sample data if it doesn't exist
            if not (project_root / "data" / "raw" / "karrierewege.csv").exists():
                st.info("Creating sample data...")
                create_sample_data()
                
                # Run data ingestion
                pipeline = DataIngestionPipeline()
                success = pipeline.run()
                
                if not success:
                    st.error("Data ingestion failed!")
                    return False
            
            # Initialize ESCO loader
            st.session_state.esco_loader = create_esco_loader(str(project_root / "data" / "processed"))
            
            # Initialize skill gap analyzer
            st.session_state.skill_analyzer = create_skill_gap_analyzer(str(project_root / "configs" / "system_config.yaml"))
            
            # Initialize career path recommender
            try:
                st.session_state.recommender = create_career_recommender(
                    model_dir=str(project_root / "models" / "bert4rec"),
                    esco_data_dir=str(project_root / "data" / "processed")
                )
                st.success("Career path recommender loaded successfully!")
            except Exception as e:
                st.warning(f"Could not load career recommender: {e}")
                st.session_state.recommender = None
            
            st.session_state.data_initialized = True
            return True
            
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            return False

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Career Path Recommender</h1>', unsafe_allow_html=True)
    st.markdown("**Knowledge-Aware Hybrid Recommender for Sustainable Career Path and Proactive Upskilling**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "User Onboarding", "Path Recommendations", "Learning Resources", "System Demo", "About"]
    )
    
    # Initialize data if needed
    if not st.session_state.data_initialized:
        if st.sidebar.button("Initialize System"):
            if initialize_data():
                st.success("System initialized successfully!")
                st.rerun()
            else:
                st.error("System initialization failed!")
                return
        else:
            st.warning("Please initialize the system using the sidebar button.")
            return
    
    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "User Onboarding":
        show_onboarding_page()
    elif page == "Path Recommendations":
        show_recommendations_page()
    elif page == "Learning Resources":
        show_resources_page()
    elif page == "System Demo":
        show_demo_page()
    elif page == "About":
        show_about_page()

def show_home_page():
    """Show the home page."""
    st.markdown('<h2 class="section-header">Welcome to the Career Path Recommender</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Stage 1: Path Generation
        - BERT4Rec transformer model
        - Bidirectional sequence modeling
        - Masked language modeling
        - Beam search for diversity
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 Stage 2: Skill Gap Analysis
        - ESCO knowledge graph integration
        - Ontology-based skill distances
        - Feasibility scoring
        - Combined model + feasibility scores
        """)
    
    with col3:
        st.markdown("""
        ### 📚 Stage 3: Resource Recommendation
        - Content-based filtering
        - TF-IDF + semantic embeddings
        - FAISS nearest neighbor search
        - Learning path optimization
        """)
    
    st.markdown("---")
    
    # System status
    st.markdown('<h3 class="section-header">System Status</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Status", "✅ Ready", "Sample data loaded")
    
    with col2:
        if st.session_state.esco_loader:
            num_occupations = len(st.session_state.esco_loader.occupations)
            st.metric("ESCO Occupations", num_occupations, "Available")
        else:
            st.metric("ESCO Occupations", "❌", "Not loaded")
    
    with col3:
        if st.session_state.esco_loader:
            num_skills = len(st.session_state.esco_loader.skills)
            st.metric("ESCO Skills", num_skills, "Available")
        else:
            st.metric("ESCO Skills", "❌", "Not loaded")
    
    with col4:
        if st.session_state.skill_analyzer:
            st.metric("Skill Analyzer", "✅ Ready", "Initialized")
        else:
            st.metric("Skill Analyzer", "❌", "Not ready")

def show_onboarding_page():
    """Show the user onboarding page."""
    st.markdown('<h2 class="section-header">User Onboarding</h2>', unsafe_allow_html=True)
    
    # Job history input
    st.markdown("### 📋 Career History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Enter your job history (most recent first):**")
        job_history = st.text_area(
            "Job titles (one per line)",
            value="Software Engineer\nJunior Developer\nIntern",
            height=150,
            help="Enter your job titles in reverse chronological order"
        )
    
    with col2:
        st.markdown("**Available ESCO Occupations (sample):**")
        if st.session_state.esco_loader:
            sample_jobs = list(st.session_state.esco_loader.occupations.items())[:5]
            for esco_id, job_data in sample_jobs:
                st.write(f"• {job_data['title']} ({esco_id})")
        else:
            st.write("ESCO data not loaded")
    
    # Skills input
    st.markdown("### 🛠️ Current Skills")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Enter your current skills:**")
        skills_input = st.text_area(
            "Skills (one per line)",
            value="Python Programming\nData Analysis\nMachine Learning",
            height=150,
            help="Enter your current skills and competencies"
        )
    
    with col2:
        st.markdown("**Available ESCO Skills (sample):**")
        if st.session_state.esco_loader:
            sample_skills = list(st.session_state.esco_loader.skills.items())[:5]
            for esco_id, skill_data in sample_skills:
                st.write(f"• {skill_data['title']} ({esco_id})")
        else:
            st.write("ESCO data not loaded")
    
    # Save to session state
    if st.button("Save Profile", type="primary"):
        job_list = [job.strip() for job in job_history.split('\n') if job.strip()]
        skills_list = [skill.strip() for skill in skills_input.split('\n') if skill.strip()]
        
        st.session_state.user_job_history = job_list
        st.session_state.user_skills = skills_list
        
        st.success(f"Profile saved! {len(job_list)} jobs and {len(skills_list)} skills recorded.")
        
        # Show summary
        with st.expander("Profile Summary"):
            st.write("**Job History:**")
            for i, job in enumerate(job_list, 1):
                st.write(f"{i}. {job}")
            
            st.write("**Skills:**")
            for skill in skills_list:
                st.write(f"• {skill}")

def generate_candidate_paths(user_job_history: List[str], top_k: int = 5) -> List[tuple]:
    """
    Generate candidate career paths using the enhanced recommender.
    
    Args:
        user_job_history: List of user's job history
        top_k: Number of candidate paths to generate
        
    Returns:
        List of (path, probability) tuples
    """
    # Use the enhanced recommender if available
    if st.session_state.recommender is not None:
        try:
            st.info("Generating personalized recommendations using the trained model...")
            
            # Generate recommendations using the enhanced recommender
            recommendations = st.session_state.recommender.generate_recommendations(
                user_job_history, top_k=top_k
            )
            
            # Convert to the format expected by the rest of the system
            candidate_paths = []
            for rec in recommendations:
                # Use the ESCO info if available, otherwise fall back to model key
                if rec['esco_info']:
                    path_element = rec['esco_info']['esco_id']
                else:
                    # Create a mock ESCO ID for model-generated jobs
                    path_element = f"model_{rec['model_job_id']}"
                
                path = [path_element]
                probability = rec['probability']
                candidate_paths.append((path, probability))
            
            if candidate_paths:
                st.success(f"Generated {len(candidate_paths)} personalized recommendations!")
                return candidate_paths
            
        except Exception as e:
            st.error(f"Error with enhanced recommender: {e}")
    
    # Fallback to mock data if recommender is not available or fails
    st.warning("Using mock recommendations (recommender not available)")
    return [
        (['occ_001', 'occ_002'], 0.85),  # Software Engineer -> Data Scientist
        (['occ_001', 'occ_003'], 0.72),  # Software Engineer -> Marketing Manager
        (['occ_002', 'occ_003'], 0.58),  # Data Scientist -> Marketing Manager
    ]

def show_recommendations_page():
    """Show the career path recommendations page."""
    st.markdown('<h2 class="section-header">Career Path Recommendations</h2>', unsafe_allow_html=True)
    
    # Check if user profile exists
    if 'user_job_history' not in st.session_state or 'user_skills' not in st.session_state:
        st.warning("Please complete the user onboarding first!")
        return
    
    st.markdown("### 🎯 Recommended Career Paths")
    
    # Generate candidate paths using the model
    with st.spinner("Generating personalized career path recommendations..."):
        candidate_paths = generate_candidate_paths(st.session_state.user_job_history)
    
    # Map ESCO IDs to titles (for mock data) or use actual job titles
    job_titles = {}
    if st.session_state.esco_loader:
        # Use actual ESCO data
        for esco_id, job_data in st.session_state.esco_loader.occupations.items():
            job_titles[esco_id] = job_data['title']
    else:
        # Fallback to mock data
        job_titles = {
            'occ_001': 'Software Engineer',
            'occ_002': 'Data Scientist', 
            'occ_003': 'Marketing Manager'
        }
    
    # Convert user skills to ESCO IDs
    user_skill_set = set()
    if st.session_state.esco_loader:
        # Try to map user skills to ESCO IDs
        text_mapper = create_text_mapper(str(project_root / "data" / "processed"))
        for skill in st.session_state.user_skills:
            matches = text_mapper.map_text_to_skills(skill, top_k=1)
            if matches:
                user_skill_set.add(matches[0]['esco_id'])
    else:
        # Fallback to mock data
        user_skill_set = {'skill_001', 'skill_002'}
    
    if st.session_state.skill_analyzer:
        # Analyze paths
        analyses = st.session_state.skill_analyzer.analyze_multiple_paths(user_skill_set, candidate_paths)
        
        # Display recommendations
        for i, analysis in enumerate(analyses, 1):
            with st.container():
                # Create path string
                path_titles = []
                for job_id in analysis.path:
                    if job_id in job_titles:
                        path_titles.append(job_titles[job_id])
                    elif job_id.startswith('model_'):
                        # Handle model-generated jobs
                        path_titles.append(f"Recommended Role ({job_id})")
                    else:
                        path_titles.append(job_id)
                
                st.markdown(f"#### Path {i}: {' → '.join(path_titles)}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Model Confidence", f"{analysis.model_prob:.1%}")
                
                with col2:
                    st.metric("Feasibility Score", f"{analysis.feasibility_score:.1%}")
                
                with col3:
                    st.metric("Combined Score", f"{analysis.combined_score:.1%}")
                
                with col4:
                    st.metric("Missing Skills", analysis.total_missing_skills)
                
                # Detailed breakdown
                with st.expander(f"Detailed Analysis for Path {i}"):
                    for job_id, gap in analysis.per_job_gaps.items():
                        job_title = job_titles.get(job_id, job_id)
                        if job_id.startswith('model_'):
                            job_title = f"Recommended Role ({job_id})"
                            
                        st.write(f"**{job_title}:**")
                        st.write(f"- Required skills: {len(gap.required_skills)}")
                        st.write(f"- Missing skills: {len(gap.missing_skills)}")
                        st.write(f"- Gap score: {gap.gap_score:.3f}")
                        
                        if gap.missing_skills and st.session_state.esco_loader:
                            # Show skill titles instead of IDs
                            missing_skill_titles = []
                            for skill_id in gap.missing_skills:
                                if skill_id in st.session_state.esco_loader.skills:
                                    missing_skill_titles.append(st.session_state.esco_loader.skills[skill_id]['title'])
                                else:
                                    missing_skill_titles.append(skill_id)
                            st.write("- Missing skills:", ", ".join(missing_skill_titles))
                        elif gap.missing_skills:
                            st.write("- Missing skill IDs:", ", ".join(gap.missing_skills))
                
                st.markdown("---")
    
    else:
        st.error("Skill analyzer not initialized!")

def show_resources_page():
    """Show the learning resources page."""
    st.markdown('<h2 class="section-header">Learning Resources</h2>', unsafe_allow_html=True)
    
    # Mock learning resources
    st.markdown("### 📚 Recommended Learning Resources")
    
    resources = [
        {
            "title": "Advanced Python Programming",
            "type": "Course",
            "provider": "Online University",
            "duration": "6 weeks",
            "difficulty": "Intermediate",
            "skills": ["Python Programming", "Object-Oriented Programming"],
            "url": "https://example.com/python-course"
        },
        {
            "title": "Data Science Fundamentals",
            "type": "Specialization",
            "provider": "Tech Academy",
            "duration": "3 months",
            "difficulty": "Beginner",
            "skills": ["Data Analysis", "Statistics", "Machine Learning"],
            "url": "https://example.com/data-science"
        },
        {
            "title": "Machine Learning in Practice",
            "type": "Workshop",
            "provider": "ML Institute",
            "duration": "2 days",
            "difficulty": "Advanced",
            "skills": ["Machine Learning", "Deep Learning", "TensorFlow"],
            "url": "https://example.com/ml-workshop"
        }
    ]
    
    for resource in resources:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{resource['title']}**")
                st.write(f"*{resource['type']} by {resource['provider']}*")
                st.write(f"Duration: {resource['duration']} | Difficulty: {resource['difficulty']}")
                st.write(f"Skills: {', '.join(resource['skills'])}")
            
            with col2:
                st.link_button("View Course", resource['url'])
            
            st.markdown("---")

def show_demo_page():
    """Show the system demonstration page."""
    st.markdown('<h2 class="section-header">System Demonstration</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This page demonstrates the core functionality of the career recommender system
    using synthetic data. Click the buttons below to test different components.
    """)
    
    # Test ESCO loader
    st.markdown("### 🔍 Test ESCO Knowledge Graph")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Job Skills Query"):
            if st.session_state.esco_loader:
                skills = st.session_state.esco_loader.get_job_skills('occ_001')
                st.write("Skills for Software Engineer (occ_001):")
                for skill in skills:
                    st.write(f"• {skill.get('title', skill.get('esco_id'))}")
            else:
                st.error("ESCO loader not initialized")
    
    with col2:
        if st.button("Test Skill Distance"):
            if st.session_state.esco_loader:
                distance = st.session_state.esco_loader.get_skill_distance('skill_001', 'skill_002')
                st.write(f"Distance between skill_001 and skill_002: {distance:.3f}")
            else:
                st.error("ESCO loader not initialized")
    
    # Test skill gap analyzer
    st.markdown("### 🧠 Test Skill Gap Analysis")
    
    if st.button("Run Skill Gap Analysis"):
        if st.session_state.skill_analyzer:
            user_skills = {'skill_001', 'skill_002'}
            test_path = ['occ_001', 'occ_002']
            
            analysis = st.session_state.skill_analyzer.analyze_path(user_skills, test_path, 0.8)
            
            st.write("**Analysis Results:**")
            st.write(f"- Model Probability: {analysis.model_prob:.3f}")
            st.write(f"- Feasibility Score: {analysis.feasibility_score:.3f}")
            st.write(f"- Combined Score: {analysis.combined_score:.3f}")
            st.write(f"- Total Missing Skills: {analysis.total_missing_skills}")
            
            # Show explanation
            explanation = st.session_state.skill_analyzer.explain_path_feasibility(analysis)
            
            with st.expander("Detailed Explanation"):
                st.json(explanation)
        else:
            st.error("Skill analyzer not initialized")

def show_about_page():
    """Show the about page."""
    st.markdown('<h2 class="section-header">About the Career Recommender System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Overview
    
    This is a **Knowledge-Aware Hybrid Recommender for Sustainable Career Path and Proactive Upskilling**
    that implements a novel three-stage cascaded architecture:
    
    ### Stage 1: Transformer-based Path Generation
    - **BERT4Rec Model**: Bidirectional transformer for career sequence modeling
    - **Masked Language Modeling**: Predicts next career steps based on history
    - **Beam Search**: Generates diverse candidate paths
    
    ### Stage 2: ESCO-based Skill-Gap Reasoning
    - **Knowledge Graph Integration**: Uses ESCO taxonomy for skill relationships
    - **Ontology Distance Calculation**: Computes semantic distances between skills
    - **Feasibility Scoring**: Combines model confidence with skill gap analysis
    
    ### Stage 3: Content-based Upskilling
    - **Resource Recommendation**: TF-IDF and semantic similarity matching
    - **Learning Path Optimization**: Orders resources by prerequisite relationships
    - **Effort Estimation**: Provides time and difficulty estimates
    
    ## 🛠️ Technical Stack
    
    - **Backend**: Python, PyTorch, Transformers, NetworkX
    - **Knowledge Graph**: ESCO (European Skills, Competences, Qualifications and Occupations)
    - **Embeddings**: Sentence-Transformers, FAISS
    - **Web Framework**: Streamlit, FastAPI
    - **Data Processing**: Pandas, Parquet
    
    ## 📊 Evaluation Metrics
    
    The system is evaluated using multiple metrics:
    - **Accuracy**: Precision@k, Recall@k, NDCG@k
    - **Diversity**: Intra-list similarity
    - **Novelty**: -log2 p(path)
    - **Serendipity**: Unexpectedness × Relevance
    
    ## 🚀 Getting Started
    
    1. **Initialize System**: Use the sidebar button to set up sample data
    2. **User Onboarding**: Enter your job history and current skills
    3. **Get Recommendations**: View personalized career path suggestions
    4. **Learning Resources**: Explore recommended courses and materials
    
    ## 📝 Note
    
    This system now uses the actual trained BERT4Rec model to generate personalized career recommendations
    based on your job history, combined with ESCO knowledge graph analysis for skill gap reasoning.
    """)

if __name__ == "__main__":
    main()