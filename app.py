import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Career Path Recommender",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Career Path Recommender System")

st.success("✅ App is working! All dependencies loaded successfully.")

st.markdown("""
## Welcome to the Career Path Recommender

This is a **lightweight demo version** of our AI-powered career recommendation system.

### 🎯 System Overview

Our system uses a three-stage hybrid approach:

1. **🤖 Path Generation**: BERT4Rec transformer model for career sequence prediction
2. **🧠 Skill Gap Analysis**: ESCO knowledge graph for feasibility scoring  
3. **🔗 Text Mapping**: Semantic similarity for job title normalization

### 📊 Interactive Demo

Try our career path analyzer below:
""")

# Simple demo
st.subheader("Career Path Analyzer")

col1, col2 = st.columns(2)

with col1:
    current_job = st.text_input("Current Job Title", "Software Developer")
    current_skills = st.text_area("Current Skills", "Python\nJavaScript\nSQL")

with col2:
    target_job = st.text_input("Target Job Title", "Data Scientist") 
    experience = st.slider("Years of Experience", 0, 20, 3)

if st.button("🚀 Analyze Career Path"):
    st.success("Analysis Complete!")
    
    # Simple calculations
    skills_list = [s.strip() for s in current_skills.split('\n') if s.strip()]
    skill_score = min(95, len(skills_list) * 15 + 20)
    exp_bonus = min(20, experience * 2)
    transition_prob = min(95, skill_score + exp_bonus - 10)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Transition Probability", f"{transition_prob}%")
    with col2:
        st.metric("Skill Match", f"{skill_score}%")
    with col3:
        st.metric("Experience Bonus", f"+{exp_bonus}%")
    
    st.markdown("### 🎯 Recommendations")
    st.info("• Build portfolio projects in your target domain")
    st.info("• Take online courses to fill skill gaps")
    st.info("• Network with professionals in your target role")
    st.info("• Consider transitional roles to gain experience")

st.markdown("---")
st.markdown("### 🔗 Full System")
st.info("For the complete ML pipeline with BERT4Rec and ESCO integration, visit: [GitHub Repository](https://github.com/Dhanu1110/hybrid-recommender-for-career-path)")

# Test data visualization
st.subheader("📊 Sample Data")

sample_data = pd.DataFrame({
    'Career Path': ['Developer → Data Scientist', 'Analyst → Manager', 'Designer → UX Lead'],
    'Probability': [78, 85, 92],
    'Timeline': ['6-12 months', '4-8 months', '3-6 months']
})

st.dataframe(sample_data, use_container_width=True)

st.success("🎉 Demo completed successfully!")
