#!/usr/bin/env bash
set -euo pipefail
# Smoke script: run ingestion, train (short) and a sample recommend using synthetic data

PYTHON=$(which python || true)
if [ -z "$PYTHON" ]; then
    PYTHON=python
fi

cd src/ingest
# create minimal sample data if not present
python download_and_prepare.py --create-sample || true
cd ../../

# Train on synthetic small data
python src/models/train_path_model.py --synthetic --models_dir models/bert4rec --epochs 1

# Run a sample recommendation (may use model defaults)
python -c "from src.models.career_path_recommender import create_career_recommender; r=create_career_recommender('models/bert4rec'); print(r.generate_recommendations(['software_engineer'], top_k=1))"

echo "SMOKE_OK"
