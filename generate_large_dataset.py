#!/usr/bin/env python3
"""
Convenience script to generate large datasets for the career recommender system.
This script provides preset configurations for different dataset sizes.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from ingest.generate_synthetic_data import create_large_sample_data
from ingest.download_and_prepare import DataIngestionPipeline

def generate_dataset(size: str = "large"):
    """Generate dataset with predefined size configurations."""
    
    size_configs = {
        "small": {
            "num_users": 1000,
            "num_jobs": 200,
            "num_skills": 800,
            "relations_per_job": 6
        },
        "medium": {
            "num_users": 5000,
            "num_jobs": 500,
            "num_skills": 2000,
            "relations_per_job": 8
        },
        "large": {
            "num_users": 10000,
            "num_jobs": 1000,
            "num_skills": 3000,
            "relations_per_job": 12
        },
        "xlarge": {
            "num_users": 25000,
            "num_jobs": 2000,
            "num_skills": 5000,
            "relations_per_job": 15
        },
        "xxlarge": {
            "num_users": 50000,
            "num_jobs": 3000,
            "num_skills": 8000,
            "relations_per_job": 20
        }
    }
    
    if size not in size_configs:
        print(f"❌ Unknown size '{size}'. Available sizes: {list(size_configs.keys())}")
        return False
    
    config = size_configs[size]
    
    print(f"🚀 Generating {size.upper()} dataset...")
    print(f"   📊 Configuration:")
    print(f"      - Users: {config['num_users']:,}")
    print(f"      - Jobs: {config['num_jobs']:,}")
    print(f"      - Skills: {config['num_skills']:,}")
    print(f"      - Relations per job: {config['relations_per_job']}")
    print(f"      - Total relations: ~{config['num_jobs'] * config['relations_per_job']:,}")
    
    try:
        # Generate the data
        print("\n📝 Generating synthetic data...")
        create_large_sample_data(**config)
        
        # Process through pipeline
        print("\n⚙️ Processing data through ingestion pipeline...")
        pipeline = DataIngestionPipeline()
        success = pipeline.run()
        
        if success:
            print(f"\n✅ {size.upper()} dataset generated successfully!")
            print(f"   📁 Data saved to: data/processed/")
            
            # Show file sizes
            processed_dir = Path("data/processed")
            if processed_dir.exists():
                print(f"\n📈 Generated files:")
                for file_path in processed_dir.glob("*.parquet"):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"      - {file_path.name}: {size_mb:.1f} MB")
            
            return True
        else:
            print("❌ Data processing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate large datasets for career recommender system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Size Options:
  small    - 1K users, 200 jobs, 800 skills (good for quick testing)
  medium   - 5K users, 500 jobs, 2K skills (balanced for development)
  large    - 10K users, 1K jobs, 3K skills (realistic scale)
  xlarge   - 25K users, 2K jobs, 5K skills (large scale testing)
  xxlarge  - 50K users, 3K jobs, 8K skills (stress testing)

Examples:
  python generate_large_dataset.py --size large
  python generate_large_dataset.py --size xlarge
  python generate_large_dataset.py --custom --users 15000 --jobs 1500 --skills 4000
        """
    )
    
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'xlarge', 'xxlarge'],
                       default='large', help='Predefined dataset size (default: large)')
    
    # Custom configuration options
    parser.add_argument('--custom', action='store_true',
                       help='Use custom configuration instead of preset sizes')
    parser.add_argument('--users', type=int, default=10000,
                       help='Number of users (for custom config)')
    parser.add_argument('--jobs', type=int, default=1000,
                       help='Number of jobs (for custom config)')
    parser.add_argument('--skills', type=int, default=3000,
                       help='Number of skills (for custom config)')
    parser.add_argument('--relations-per-job', type=int, default=12,
                       help='Relations per job (for custom config)')
    
    args = parser.parse_args()
    
    if args.custom:
        print(f"🚀 Generating CUSTOM dataset...")
        print(f"   📊 Configuration:")
        print(f"      - Users: {args.users:,}")
        print(f"      - Jobs: {args.jobs:,}")
        print(f"      - Skills: {args.skills:,}")
        print(f"      - Relations per job: {args.relations_per_job}")
        
        try:
            create_large_sample_data(
                num_users=args.users,
                num_jobs=args.jobs,
                num_skills=args.skills,
                relations_per_job=args.relations_per_job
            )
            
            pipeline = DataIngestionPipeline()
            success = pipeline.run()
            
            if success:
                print(f"\n✅ Custom dataset generated successfully!")
            else:
                print("❌ Data processing failed!")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Dataset generation failed: {e}")
            sys.exit(1)
    else:
        success = generate_dataset(args.size)
        if not success:
            sys.exit(1)
    
    print(f"\n💡 Next steps:")
    print(f"   - Run the demo: python demo_production_usage.py")
    print(f"   - Train a model: python src/models/train_path_model.py")
    print(f"   - Start web app: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
