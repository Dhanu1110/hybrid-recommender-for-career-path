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
import re
from pathlib import Path
import pandas as pd
import json
# Optional visualization dependency: Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    go = None
    PLOTLY_AVAILABLE = False
import yaml
import requests
from typing import List, Dict, Set, Optional, Tuple

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

# Import our modules
try:
    from ingest.download_and_prepare import create_sample_data, DataIngestionPipeline
    from ingest.esco_loader import create_esco_loader, ESCOKnowledgeGraph
    from ingest.text_to_esco_mapper import create_text_mapper, map_text_to_occupations
    from models.bert4rec import BERT4RecConfig, create_bert4rec_model
    from models.career_path_recommender import create_career_recommender
    from models.train_path_model import create_synthetic_data
    from reasoner.skill_gap import create_skill_gap_analyzer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()
    
if not PLOTLY_AVAILABLE:
    # Avoid raising at import time; surface a friendly warning in the UI later
    import logging
    logging.getLogger(__name__).warning("plotly not installed - visualizations will be disabled.")

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
for key in ['data_initialized', 'config', 'esco_loader', 'skill_analyzer', 'recommender', 'text_mapper']:
    if key not in st.session_state:
        st.session_state[key] = None

# Set initial values
st.session_state.data_initialized = st.session_state.data_initialized or False

def load_config() -> dict:
    """Load system configuration from YAML."""
    config_path = project_root / "configs" / "system_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if not config.get('mapping', {}).get('confidence_threshold'):
                config.setdefault('mapping', {})['confidence_threshold'] = 0.4
            if not config.get('data', {}).get('processed_dir'):
                config.setdefault('data', {})['processed_dir'] = 'data/processed'
            return config
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {
            'mapping': {'confidence_threshold': 0.4},
            'data': {'processed_dir': 'data/processed'}
        }

def initialize_data():
    """Initialize the data and models."""
    with st.spinner("Initializing data and models..."):
        try:
            # Load configuration
            config = load_config()
            st.session_state.config = config
            
            # Setup data directories
            processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
            Path(processed_dir).mkdir(parents=True, exist_ok=True)
            
            # Create sample data if needed
            if not (project_root / "data" / "raw" / "karrierewege.csv").exists():
                st.info("Creating sample data...")
                create_sample_data()
                
                # Run data ingestion
                pipeline = DataIngestionPipeline()
                success = pipeline.run()
                if not success:
                    st.error("Data ingestion failed")
                    return False
                
            # Initialize ESCO loader
            st.info("Loading ESCO knowledge graph...")
            esco_loader = create_esco_loader(processed_dir)
            if not esco_loader:
                st.error("Failed to load ESCO data")
                return False
            st.session_state.esco_loader = esco_loader
            
            # Initialize text mapper
            st.info("Setting up text mapper...")
            text_mapper = create_text_mapper(processed_dir)
            if not text_mapper:
                st.error("Failed to initialize text mapper")
                return False
            st.session_state.text_mapper = text_mapper
            
            # Initialize skill analyzer
            st.info("Initializing skill analyzer...")
            # create_skill_gap_analyzer expects a config path (str). Pass the system config
            # and then attach our loaded esco_loader instance so the analyzer uses the same data.
            config_path = str(project_root / "configs" / "system_config.yaml")
            skill_analyzer = create_skill_gap_analyzer(config_path)
            if not skill_analyzer:
                st.error("Failed to initialize skill analyzer")
                return False
            try:
                # attach the ESCO loader instance for consistency
                skill_analyzer.esco_loader = esco_loader
            except Exception:
                # non-fatal: some analyzer implementations may ignore this
                pass
            st.session_state.skill_analyzer = skill_analyzer
            
            # Initialize career path recommender
            st.info("Loading career path model...")
            try:
                model_dir = str(project_root / "models" / "bert4rec")
                esco_data_dir = str(project_root / "data" / "processed")
                
                if not Path(model_dir).exists():
                    raise FileNotFoundError(f"Model directory not found: {model_dir}")
                    
                checkpoint_path = Path(model_dir) / "checkpoints" / "best_model.pt"
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
                
                st.session_state.recommender = create_career_recommender(
                    model_dir=model_dir,
                    esco_data_dir=esco_data_dir
                )
                st.success("Career path recommender loaded successfully!")
            except FileNotFoundError as e:
                st.error(f"Model files missing: {e}")
                st.session_state.recommender = None
            except Exception as e:
                st.error(f"Could not load career recommender: {str(e)}")
                st.session_state.recommender = None
            
            st.session_state.data_initialized = True
            return True
            
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            return False

def main():
    """Main application function."""
    
    try:
        with st.spinner("Processing..."):
            # Header
            st.markdown('<h1 class="main-header">🚀DA 2 SDG 8 Job Reccomender </h1>', unsafe_allow_html=True)
            st.markdown("**Knowledge-Aware Hybrid Recommender for Sustainable Career Path and Proactive Upskilling - SDG 8**")
            
            # Inform user if optional visualization dependency is missing
            if not globals().get('PLOTLY_AVAILABLE', False):
                st.warning("📊 Optional dependency 'plotly' is not installed — some visualizations are disabled. Install `plotly` to enable charts or continue without them.")
            
            # Sidebar
            st.sidebar.title("Navigation")
            page = st.sidebar.selectbox("Choose a page:", ["User Onboarding", "Career Paths", "Skill Analysis"])

            # Initialize data if needed
            if not st.session_state.get('data_initialized'):
                success = initialize_data()
                if not success:
                    st.error("System initialization failed!")
                    return
    except Exception as e:
        st.error("An error occurred during initialization. Please try refreshing the page.")
        return

    # Show selected page
    if page == "User Onboarding":
        show_onboarding_page()
    elif page == "Career Paths":
        show_recommendations_page()
    else:
        # Fallback or skill analysis
        try:
            show_skill_analysis_page()
        except NameError:
            st.info("Skill analysis is not yet available.")
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
        
        # Show mapping results in real-time
        if job_history:
            st.markdown("**🔄 Job Title Mapping Results:**")
            from utils import map_job_title_with_confidence
            
            for job in job_history.split('\n'):
                if job.strip():
                    mapped_title, confidence = map_job_title_with_confidence(job.strip())
                    if mapped_title:
                        emoji = "✅" if confidence >= 0.7 else "⚠️" if confidence >= 0.4 else "❌"
                        st.markdown(f"{emoji} '{job}' → '{mapped_title}' ({confidence:.1%} confidence)")
                    else:
                        st.markdown(f"❌ '{job}' → No match found ({confidence:.1%} confidence)")
    
    with col2:
        st.markdown("**Available ESCO Occupations (sample):**")
        if st.session_state.esco_loader:
            sample_jobs = list(st.session_state.esco_loader.occupations.items())[:5]
            for esco_id, job_data in sample_jobs:
                st.write(f"• {job_data['title']} ({esco_id})")
            
            st.markdown("---")
            st.markdown("""
            **Confidence Levels:**
            - ✅ High (>70%): Strong match
            - ⚠️ Medium (40-70%): Possible match
            - ❌ Low (<40%): No reliable match
            """)
    
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
        
        # Show mapping results in real-time
        if skills_input:
            st.markdown("**🔄 Skill Mapping Results:**")
            from utils import map_skill_with_confidence
            
            for skill in skills_input.split('\n'):
                if skill.strip():
                    mapped_skill, confidence = map_skill_with_confidence(skill.strip())
                    if mapped_skill:
                        emoji = "✅" if confidence >= 0.7 else "⚠️" if confidence >= 0.4 else "❌"
                        st.markdown(f"{emoji} '{skill}' → '{mapped_skill}' ({confidence:.1%} confidence)")
                    else:
                        st.markdown(f"❌ '{skill}' → No match found ({confidence:.1%} confidence)")
    
    with col2:
        st.markdown("**Available ESCO Skills (sample):**")
        if st.session_state.esco_loader:
            sample_skills = list(st.session_state.esco_loader.skills.items())[:5]
            for esco_id, skill_data in sample_skills:
                st.write(f"• {skill_data['title']} ({esco_id})")
            
            st.markdown("---")
            st.markdown("""
            **Mapping Confidence:**
            - ✅ High (>70%): Direct ESCO match
            - ⚠️ Medium (40-70%): Related skill found
            - ❌ Low (<40%): No suitable match
            """)
            
            st.info("💡 Tip: Try to use standard skill descriptions for better matches.")
    
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
    Generate candidate career paths using text mapping and AI recommendations.
    
    Args:
        user_job_history: List of user's job history
        top_k: Number of candidate paths to generate
        
    Returns:
        List of (path, probability) tuples
    """
    if not user_job_history:
        st.error("No job history provided")
        return []
    
    current_job = user_job_history[-1]
    
    # Try using text mapper first
    if st.session_state.text_mapper:
        try:
            matches = []
            with st.spinner(f"Finding relevant career paths based on: {current_job}"):
                matches = st.session_state.text_mapper.map_text_to_occupations(
                    current_job, top_k=top_k*2, score_threshold=0.3
                )
                
            if isinstance(matches, (list, tuple)) and matches:
                candidate_paths = []
                for match in matches:
                    if isinstance(match, dict):
                        path = [match.get('esco_id')]
                        probability = match.get('score', 0.5)
                        if path[0]:  # Only add if we got a valid ESCO ID
                            candidate_paths.append((path, probability))
                
                if candidate_paths:
                    st.success(f"Found {len(candidate_paths)} relevant career paths!")
                    return sorted(candidate_paths, key=lambda x: x[1], reverse=True)[:top_k]
        except:
            pass  # Silently handle mapping errors and continue to next method
    
    # Try AI recommender as backup
    if st.session_state.recommender is not None:
        try:
            st.info("Using AI recommender...")
            recs = st.session_state.recommender.generate_recommendations(
                user_job_history, top_k=top_k
            )
            
            candidate_paths = []
            for rec in recs:
                if rec.get('esco_info'):
                    path = [rec['esco_info']['esco_id']]
                    probability = rec.get('probability', 0.5)
                    candidate_paths.append((path, probability))
            
            if candidate_paths:
                st.success(f"Generated {len(candidate_paths)} AI recommendations!")
                return candidate_paths
                
        except Exception as e:
            st.error(f"AI recommender failed: {str(e)}")
    
    # Fallback to fuzzy matching
    if st.session_state.esco_loader:
        st.warning("Using simple job matching...")
        from difflib import SequenceMatcher
        matches = []
        
        for esco_id, job_data in st.session_state.esco_loader.occupations.items():
            title = job_data.get('title', '')
            if title:
                score = SequenceMatcher(None, current_job.lower(), title.lower()).ratio()
                if score >= 0.3:
                    matches.append((esco_id, score))
        
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            paths = [([esco_id], score) for esco_id, score in matches[:top_k]]
            st.success(f"Found {len(paths)} similar jobs!")
            return paths
    
    st.error("Could not generate recommendations. Please try a different job title.")
    return []

def get_gemini_explanation(prompt: str) -> str:
    """Generate career path explanation using templates when API is unavailable."""
    try:
        # First try template-based explanation
        explanation = generate_path_explanation(prompt)
        if explanation:
            return explanation
            
        # Fallback to API only if template fails
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return "[AI explanation unavailable: No API key provided]"
            
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
        
    except Exception as e:
        return generate_path_explanation(prompt) or f"[Explanation generation failed: {str(e)}]"

def generate_path_explanation(prompt: str) -> str:
    """Generate explanation using templates based on the career path."""
    if not prompt:
        return ""
        
    # Extract job titles from prompt
    path_match = re.search(r"career path (.*?) is a good recommendation", prompt)
    if not path_match:
        return ""
    
    path = path_match.group(1)
    jobs = [job.strip() for job in path.split('→')]
    
    if len(jobs) < 1:
        return ""
        
    current_job = jobs[0]
    target_job = jobs[-1] if len(jobs) > 1 else jobs[0]
    
    # Get user skills from prompt
    skills_match = re.search(r"user's skills: (.*?)$", prompt)
    user_skills = skills_match.group(1) if skills_match else "N/A"
    
    # Template-based explanation
    explanation = f"""This career path recommendation from {current_job} to {target_job} is based on:

1. Skills Alignment: The path leverages your current skills in {user_skills}

2. Career Progression: This transition represents a natural career progression that many professionals have successfully made.

3. Market Demand: Both roles are in high demand in the current job market, providing good career stability.

4. Growth Potential: This path offers opportunities for professional growth and skill development.

To increase your success in this transition, focus on:
• Developing the identified missing skills through training and certifications
• Building practical experience in key technical areas
• Networking with professionals in the target role"""

    return explanation


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
            matches = text_mapper.map_text_to_skills(skill, top_k=2)
            if matches:
                # mapper returns list of dicts; take top match
                user_skill_set.add(matches[0].get('esco_id'))
            else:
                # Fallback: try simple fuzzy/title match against ESCO skills
                from difflib import SequenceMatcher
                best = (None, 0.0)
                for esco_id, sdata in st.session_state.esco_loader.skills.items():
                    title = sdata.get('title', '')
                    score = SequenceMatcher(None, skill.lower(), title.lower()).ratio()
                    if score > best[1]:
                        best = (esco_id, score)
                # Accept fallback match if reasonably similar
                if best[0] and best[1] >= 0.6:
                    user_skill_set.add(best[0])
                    # optional: show a note to the user about the fallback mapping
                    st.info(f"Fallback mapped skill '{skill}' → {st.session_state.esco_loader.skills[best[0]]['title']} ({best[1]:.0%})")
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
                
                # AI explanation
                with st.expander(f"AI Explanation for Path {i}"):
                    user_skills = ', '.join(st.session_state.user_skills) if 'user_skills' in st.session_state else 'N/A'
                    prompt = f"Explain why the career path {' → '.join(path_titles)} is a good recommendation. Consider the user's skills: {user_skills}."
                    explanation = get_gemini_explanation(prompt)
                    st.write(explanation)
                
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

def show_skill_analysis_page():
    """Placeholder for the skill analysis page."""
    st.markdown('<h2 class="section-header">Skill Analysis</h2>', unsafe_allow_html=True)
    st.info("Skill analysis is under construction. Use the Demo or Career Paths pages for now.")

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