#!/usr/bin/env python3
"""
Demo script showing how to use the production career recommender system programmatically
"""

import sys
from pathlib import Path
import torch
import json

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from ingest.download_and_prepare import DataIngestionPipeline
from ingest.generate_synthetic_data import create_large_sample_data
from ingest.esco_loader import create_esco_loader
from ingest.text_to_esco_mapper import create_text_mapper
from models.bert4rec import BERT4RecConfig, create_bert4rec_model
from reasoner.skill_gap import create_skill_gap_analyzer

def demo_production_system():
    """Demonstrate the production system usage"""
    print("🚀 Career Recommender System - Production Demo")
    print("=" * 50)
    
    try:
        # Step 1: Initialize the system
        print("\n📋 Step 1: Initializing system...")
        
        # Create and process large-scale sample data
        print("   Creating large-scale synthetic data (10,000 users, 1,000 jobs, 3,000 skills)...")
        create_large_sample_data(
            num_users=10000,      # Increased from ~500 to 10,000 users
            num_jobs=1000,        # Increased from ~200 to 1,000 jobs
            num_skills=3000,      # Increased from ~800 to 3,000 skills
            relations_per_job=12  # Increased skill-job relations
        )
        print("   Processing data through ingestion pipeline...")
        pipeline = DataIngestionPipeline()
        pipeline.run()
        
        # Load components
        esco_loader = create_esco_loader(str(project_root / "data" / "processed"))
        skill_analyzer = create_skill_gap_analyzer(str(project_root / "configs" / "system_config.yaml"))
        
        # Load the trained model
        model_config_path = project_root / "models" / "bert4rec" / "model_config.json"
        model_checkpoint_path = project_root / "models" / "bert4rec" / "checkpoints" / "best_model.pt"
        job_vocab_path = project_root / "models" / "bert4rec" / "job_vocab.json"
        
        with open(model_config_path, 'r') as f:
            config_dict = json.load(f)
        model_config = BERT4RecConfig.from_dict(config_dict)
        
        model = create_bert4rec_model(model_config)
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with open(job_vocab_path, 'r') as f:
            job_to_id = json.load(f)
        id_to_job = {v: k for k, v in job_to_id.items()}
        
        print("✅ System initialized successfully!")
        print(f"   - {len(esco_loader.occupations)} occupations loaded")
        print(f"   - {len(esco_loader.skills)} skills loaded")
        print(f"   - Model with {sum(p.numel() for p in model.parameters())} parameters loaded")
        
        # Step 2: Define user profile
        print("\n👤 Step 2: Defining user profile...")
        user_job_history = ["Software Engineer", "Junior Developer"]
        user_skills = ["Python Programming", "Data Analysis"]
        
        print(f"   User job history: {user_job_history}")
        print(f"   User skills: {user_skills}")
        
        # Step 3: Map user input to ESCO IDs
        print("\n🔍 Step 3: Mapping user input to ESCO taxonomy...")
        text_mapper = create_text_mapper(str(project_root / "data" / "processed"))
        
        # Map job history to ESCO IDs
        job_ids = []
        for job in user_job_history:
            matches = text_mapper.map_text_to_occupations(job, top_k=1)
            if matches:
                job_ids.append(matches[0]['esco_id'])
                print(f"   '{job}' → '{matches[0]['title']}' (ESCO ID: {matches[0]['esco_id']})")
            else:
                job_ids.append('occ_001')  # Default fallback
                print(f"   '{job}' → 'Software Engineer' (default fallback)")
        
        # Map skills to ESCO IDs
        skill_ids = set()
        for skill in user_skills:
            matches = text_mapper.map_text_to_skills(skill, top_k=1)
            if matches:
                skill_ids.add(matches[0]['esco_id'])
                print(f"   '{skill}' → '{matches[0]['title']}' (ESCO ID: {matches[0]['esco_id']})")
            else:
                skill_ids.add('skill_001')  # Default fallback
                print(f"   '{skill}' → 'Python Programming' (default fallback)")
        
        # Step 4: Generate career path recommendations
        print("\n🎯 Step 4: Generating career path recommendations...")
        
        # Convert job history to model input
        input_ids = []
        for job in user_job_history:
            # Simple mapping for demo - in practice, you'd use the text mapper results
            if job in job_to_id:
                input_ids.append(job_to_id[job])
            else:
                # Find closest match
                found = False
                for job_key in job_to_id.keys():
                    if job.lower() in job_key.lower() or job_key.lower() in job.lower():
                        input_ids.append(job_to_id[job_key])
                        found = True
                        break
                if not found:
                    input_ids.append(2)  # Default to first job in vocab
        
        if not input_ids:
            input_ids = [2]  # Default to first job in vocab
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        print(f"   Model input: {[id_to_job.get(id, f'ID:{id}') for id in input_ids]}")
        
        # Generate recommendations
        with torch.no_grad():
            recommendations = model.generate_next_jobs(input_tensor, top_k=3)
        
        # Convert to candidate paths
        candidate_paths = []
        for rec in recommendations[0]:  # First batch element
            job_id, probability = rec
            if job_id in id_to_job:
                path = [id_to_job[job_id]]
                candidate_paths.append((path, probability))
                print(f"   Recommended next job: {id_to_job[job_id]} (Probability: {probability:.3f})")
        
        # Step 5: Analyze skill gaps
        print("\n🧮 Step 5: Analyzing skill gaps...")
        
        # For demo, we'll use the first recommended path
        if candidate_paths:
            recommended_job = candidate_paths[0][0][0]  # First job in first path
            test_path = [job_ids[0] if job_ids else 'occ_001', recommended_job]
            
            analysis = skill_analyzer.analyze_path(skill_ids, test_path, candidate_paths[0][1])
            
            print(f"   Path: {esco_loader.occupations.get(test_path[0], {}).get('title', test_path[0])} → {esco_loader.occupations.get(test_path[1], {}).get('title', test_path[1])}")
            print(f"   Model Probability: {analysis.model_prob:.3f}")
            print(f"   Feasibility Score: {analysis.feasibility_score:.3f}")
            print(f"   Combined Score: {analysis.combined_score:.3f}")
            print(f"   Missing Skills: {analysis.total_missing_skills}")
            
            if analysis.total_missing_skills > 0:
                print("   Missing skills:")
                for job_id, gap in analysis.per_job_gaps.items():
                    if gap.missing_skills:
                        job_title = esco_loader.occupations.get(job_id, {}).get('title', job_id)
                        print(f"     For {job_title}:")
                        for skill_id in gap.missing_skills:
                            skill_title = esco_loader.skills.get(skill_id, {}).get('title', skill_id)
                            print(f"       - {skill_title}")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 To use the full web interface:")
        print("   Run: streamlit run app/streamlit_app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_production_system()
    sys.exit(0 if success else 1)