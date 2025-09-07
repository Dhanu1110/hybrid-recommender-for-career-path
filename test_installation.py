#!/usr/bin/env python3
"""
Test script to verify the career recommender system installation.

This script tests the core components to ensure everything is working correctly.
"""

import sys
import os
from pathlib import Path
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    # Optional imports
    try:
        import sentence_transformers
        print(f"✅ Sentence-Transformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"⚠️  Sentence-Transformers not available: {e}")
    
    try:
        import faiss
        print(f"✅ FAISS available")
    except ImportError as e:
        print(f"⚠️  FAISS not available: {e}")
    
    return True

def test_project_structure():
    """Test that the project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "src/ingest",
        "src/models",
        "src/reasoner",
        "configs",
        "notebooks",
        "app"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} missing")
            return False
    
    required_files = [
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        "src/ingest/download_and_prepare.py",
        "src/ingest/esco_loader.py",
        "src/models/bert4rec.py",
        "src/reasoner/skill_gap.py",
        "configs/system_config.yaml",
        "notebooks/quick_demo.ipynb",
        "app/streamlit_app.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def test_data_ingestion():
    """Test the data ingestion pipeline."""
    print("\nTesting data ingestion...")
    
    try:
        # Add src to path
        sys.path.append(str(Path("src")))
        
        from ingest.download_and_prepare import create_sample_data, DataIngestionPipeline
        
        # Create sample data
        print("Creating sample data...")
        create_sample_data()
        print("✅ Sample data created")
        
        # Run ingestion pipeline
        print("Running data ingestion pipeline...")
        pipeline = DataIngestionPipeline()
        success = pipeline.run()
        
        if success:
            print("✅ Data ingestion completed")
        else:
            print("❌ Data ingestion failed")
            return False
        
        # Check output files
        processed_files = [
            "data/processed/career_paths.parquet",
            "data/processed/esco_occupations.parquet",
            "data/processed/esco_skills.parquet",
            "data/processed/esco_relations.parquet"
        ]
        
        for file_path in processed_files:
            if Path(file_path).exists():
                print(f"✅ {file_path} created")
            else:
                print(f"❌ {file_path} not created")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        traceback.print_exc()
        return False

def test_esco_loader():
    """Test the ESCO knowledge graph loader."""
    print("\nTesting ESCO loader...")
    
    try:
        from ingest.esco_loader import create_esco_loader
        
        # Create ESCO loader
        esco_loader = create_esco_loader("data/processed")
        print("✅ ESCO loader created")
        
        # Test basic functions
        job_skills = esco_loader.get_job_skills('occ_001')
        print(f"✅ Job skills query returned {len(job_skills)} skills")
        
        skill_parents = esco_loader.get_skill_parents('skill_001')
        print(f"✅ Skill parents query returned {len(skill_parents)} parents")
        
        distance = esco_loader.get_skill_distance('skill_001', 'skill_002')
        print(f"✅ Skill distance calculation: {distance:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ESCO loader test failed: {e}")
        traceback.print_exc()
        return False

def test_bert4rec_model():
    """Test the BERT4Rec model."""
    print("\nTesting BERT4Rec model...")
    
    try:
        from models.bert4rec import BERT4RecConfig, create_bert4rec_model
        
        # Create small model for testing
        config = BERT4RecConfig(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=20
        )
        
        model = create_bert4rec_model(config)
        print(f"✅ BERT4Rec model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        import torch
        input_ids = torch.randint(2, 100, (2, 10))  # Batch of 2, sequence length 10
        
        with torch.no_grad():
            logits = model(input_ids)
            print(f"✅ Forward pass successful, output shape: {logits.shape}")
        
        # Test recommendation generation
        recommendations = model.generate_next_jobs(input_ids, top_k=5)
        print(f"✅ Recommendation generation successful, got {len(recommendations)} sequences")
        
        return True
        
    except Exception as e:
        print(f"❌ BERT4Rec model test failed: {e}")
        traceback.print_exc()
        return False

def test_skill_gap_analyzer():
    """Test the skill gap analyzer."""
    print("\nTesting skill gap analyzer...")
    
    try:
        from reasoner.skill_gap import create_skill_gap_analyzer
        
        # Create analyzer
        analyzer = create_skill_gap_analyzer("configs/system_config.yaml")
        print("✅ Skill gap analyzer created")
        
        # Test path analysis
        user_skills = {'skill_001', 'skill_002'}
        test_path = ['occ_001', 'occ_002']
        
        analysis = analyzer.analyze_path(user_skills, test_path, 0.8)
        print(f"✅ Path analysis completed")
        print(f"   - Model prob: {analysis.model_prob:.3f}")
        print(f"   - Feasibility: {analysis.feasibility_score:.3f}")
        print(f"   - Combined: {analysis.combined_score:.3f}")
        print(f"   - Missing skills: {analysis.total_missing_skills}")
        
        # Test explanation generation
        explanation = analyzer.explain_path_feasibility(analysis)
        print(f"✅ Explanation generated with {len(explanation)} components")
        
        return True
        
    except Exception as e:
        print(f"❌ Skill gap analyzer test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_app():
    """Test that the Streamlit app can be imported."""
    print("\nTesting Streamlit app...")
    
    try:
        import streamlit as st
        print("✅ Streamlit available")
        
        # Check if app file exists and can be imported
        app_path = Path("app/streamlit_app.py")
        if app_path.exists():
            print("✅ Streamlit app file exists")
            print("   To run the app: streamlit run app/streamlit_app.py")
        else:
            print("❌ Streamlit app file missing")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Streamlit not available: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Career Recommender System - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Data Ingestion", test_data_ingestion),
        ("ESCO Loader", test_esco_loader),
        ("BERT4Rec Model", test_bert4rec_model),
        ("Skill Gap Analyzer", test_skill_gap_analyzer),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run the Jupyter demo: jupyter notebook notebooks/quick_demo.ipynb")
        print("2. Start the Streamlit app: streamlit run app/streamlit_app.py")
        print("3. Train a model: python src/models/train_path_model.py --synthetic --epochs 5")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
