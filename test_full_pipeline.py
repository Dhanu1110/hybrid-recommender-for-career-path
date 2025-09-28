#!/usr/bin/env python3
"""
Test script to verify the full career recommendation pipeline
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from ingest.download_and_prepare import create_sample_data, DataIngestionPipeline
from ingest.esco_loader import create_esco_loader
from reasoner.skill_gap import create_skill_gap_analyzer

def test_full_pipeline():
    """Test the full career recommendation pipeline"""
    print("Testing full career recommendation pipeline...")
    
    try:
        # Step 1: Create and process sample data
        print("Step 1: Creating and processing sample data...")
        create_sample_data()
        
        # Run data ingestion
        pipeline = DataIngestionPipeline()
        success = pipeline.run()
        if not success:
            print("❌ Data ingestion failed!")
            return False
        print("✅ Sample data created and processed successfully")
        
        # Step 2: Load ESCO data
        print("Step 2: Loading ESCO data...")
        esco_loader = create_esco_loader(str(project_root / "data" / "processed"))
        print(f"✅ ESCO data loaded: {len(esco_loader.occupations)} occupations, {len(esco_loader.skills)} skills")
        
        # Step 3: Initialize skill gap analyzer
        print("Step 3: Initializing skill gap analyzer...")
        skill_analyzer = create_skill_gap_analyzer(str(project_root / "configs" / "system_config.yaml"))
        print("✅ Skill gap analyzer initialized")
        
        # Step 4: Test skill gap analysis
        print("Step 4: Testing skill gap analysis...")
        user_skills = {'skill_001', 'skill_002'}
        test_path = ['occ_001', 'occ_002']
        
        analysis = skill_analyzer.analyze_path(user_skills, test_path, 0.8)
        print(f"✅ Analysis completed:")
        print(f"  - Model Probability: {analysis.model_prob:.3f}")
        print(f"  - Feasibility Score: {analysis.feasibility_score:.3f}")
        print(f"  - Combined Score: {analysis.combined_score:.3f}")
        print(f"  - Total Missing Skills: {analysis.total_missing_skills}")
        
        print("✅ Full pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)