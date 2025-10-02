#!/usr/bin/env python3
"""
Debug script to test the recommender system
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

def test_recommender():
    """Test the recommender system"""
    print("Testing recommender system...")
    
    try:
        # Import the recommender
        from models.career_path_recommender import create_career_recommender
        
        # Create recommender
        print("Creating recommender...")
        recommender = create_career_recommender(
            model_dir=str(project_root / "models" / "bert4rec"),
            esco_data_dir=str(project_root / "data" / "processed")
        )
        print("Recommender created successfully!")
        
        # Test with different inputs
        test_cases = [
            ["Software Engineer"],
            ["Data Scientist"],
            ["Marketing Manager"],
            ["Software Engineer", "Senior Developer"],
            ["Intern", "Junior Developer", "Software Engineer"]
        ]
        
        for i, job_history in enumerate(test_cases):
            print(f"\nTest case {i+1}: {job_history}")
            try:
                recommendations = recommender.generate_recommendations(job_history, top_k=3)
                print(f"  Generated {len(recommendations)} recommendations:")
                for rec in recommendations:
                    print(f"    {rec['rank']}. {rec['model_job_key']} (Probability: {rec['probability']:.4f})")
                    if rec['esco_info']:
                        print(f"       ESCO: {rec['esco_info']['esco_title']} ({rec['esco_info']['esco_id']})")
            except Exception as e:
                print(f"  Error generating recommendations: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error testing recommender: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_recommender()
    sys.exit(0 if success else 1)