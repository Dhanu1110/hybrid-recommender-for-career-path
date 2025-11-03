# Knowledge-Aware Hybrid Recommender for Sustainable Career Path and Proactive Upskilling

A production-ready three-stage cascaded hybrid recommender system that combines transformer-based path generation, ESCO-based skill-gap reasoning, and content-based upskilling recommendations.

## 🏗️ Architecture Overview

The system implements a novel three-stage architecture:

1. **Stage 1: Path Generation** - BERT4Rec transformer model generates candidate career paths
2. **Stage 2: Skill Gap Reasoning** - ESCO knowledge graph analyzes feasibility and skill gaps
3. **Stage 3: Resource Recommendation** - Content-based system suggests learning resources

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd career-recommender
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
# OR using pip-tools
pip install -e .
```

3. **Create sample data and test the system:**
```bash
# Create synthetic data for testing
python src/ingest/download_and_prepare.py --create-sample

# Run data ingestion pipeline
python src/ingest/download_and_prepare.py

# Test with Jupyter notebook
jupyter notebook notebooks/quick_demo.ipynb
```

### Training a Model

```bash
# Train BERT4Rec on synthetic data
python src/models/train_path_model.py --synthetic --epochs 10 --batch-size 16

# Train on real data (when available)
python src/models/train_path_model.py --data-path data/processed/career_paths.parquet --epochs 50
```

## 📁 Repository Structure

```
career-recommender/
├── data/
│   ├── raw/                    # Raw input data
│   └── processed/              # Processed Parquet files
├── src/
│   ├── ingest/                 # Data ingestion and preprocessing
│   │   ├── download_and_prepare.py
│   │   ├── esco_loader.py
│   │   └── text_to_esco_mapper.py
│   ├── models/                 # ML models
│   │   ├── bert4rec.py
│   │   └── train_path_model.py
│   ├── reasoner/               # Skill gap reasoning
│   │   └── skill_gap.py
│   ├── resources/              # Resource recommendation (Stage 3)
│   ├── eval/                   # Evaluation framework
│   ├── api/                    # FastAPI backend
│   └── app/                    # Streamlit frontend
├── notebooks/                  # Jupyter notebooks
│   └── quick_demo.ipynb
├── models/                     # Trained model artifacts
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── configs/                    # Configuration files
├── logs/                       # Application logs
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md
```

## 📊 Data Requirements

### Expected Input Data

Place the following files in `data/raw/`:

1. **karrierewege.csv** - Career path data
   ```csv
   user_id,job_sequence,timestamp
   user_001,"software_engineer,senior_software_engineer,tech_lead",2023-01-01
   ```

2. **esco_occupations.csv** - ESCO occupation data
   ```csv
   esco_id,title,description
   occ_001,Software Engineer,Develops software applications
   ```

3. **esco_skills.csv** - ESCO skills data
   ```csv
   esco_id,title,description,skill_type
   skill_001,Python Programming,Programming in Python,skill
   ```

4. **esco_relations.csv** - ESCO relations data
   ```csv
   source_id,target_id,relation_type
   occ_001,skill_001,requires
   ```

### Data Sources

- **Karrierewege**: Career transition data from job platforms
- **ESCO**: European Skills, Competences, Qualifications and Occupations framework
  - Download from: https://ec.europa.eu/esco/portal/download
  - Use CSV exports or RDF data

## 🔧 Configuration

### System Configuration (`configs/system_config.yaml`)

```yaml
# Stage combination weights
skill_gap:
  alpha: 0.6  # Model probability weight
  beta: 0.4   # Feasibility score weight

# Model configuration
path_generation:
  model_type: "bert4rec"
  config_path: "configs/bert4rec_config.json"
```

### Model Configuration (`configs/bert4rec_config.json`)

```json
{
  "vocab_size": 10000,
  "d_model": 256,
  "n_layers": 6,
  "n_heads": 8,
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 20
  }
}
```

## 🧪 Testing

Run the complete pipeline with synthetic data:

```bash
# Create and process synthetic data
python src/ingest/download_and_prepare.py --create-sample
python src/ingest/download_and_prepare.py

# Train a small model
python src/models/train_path_model.py --synthetic --epochs 5

# Run the demo notebook
jupyter notebook notebooks/quick_demo.ipynb
```

## 🏃‍♂️ Running the Application

### Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

### FastAPI Backend

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
docker-compose up -d
```

## 📈 Evaluation

The system includes comprehensive evaluation metrics:

- **Accuracy**: Precision@k, Recall@k, NDCG@k
- **Diversity**: Intra-list similarity
- **Novelty**: -log2 p(path)
- **Serendipity**: Unexpectedness × Relevance

```bash
python src/eval/evaluate.py --data-path data/processed/career_paths.parquet
```

## 🔍 Key Features

### Stage 1: BERT4Rec Path Generation
- Bidirectional transformer for career sequence modeling
- Masked language modeling for path prediction
- Configurable architecture (layers, heads, dimensions)
- Beam search for diverse path generation

### Stage 2: Skill Gap Reasoning
- ESCO knowledge graph integration
- Ontology-based skill distance calculation
- Feasibility scoring with skill gap analysis
- Weighted combination of model confidence and feasibility

### Stage 3: Resource Recommendation
- TF-IDF and sentence embedding similarity
- FAISS-based nearest neighbor search
- Skill prerequisite ordering
- Learning effort estimation

### Additional Features
- **Text-to-ESCO Mapping**: Fuzzy matching + semantic similarity
- **Caching**: Redis-based caching for performance
- **Logging**: Structured JSON logging for monitoring
- **API**: RESTful API with FastAPI
- **UI**: Interactive Streamlit dashboard

## 🛠️ Development

### Adding New Models

1. Create model class in `src/models/`
2. Implement training script
3. Update configuration files
4. Add evaluation metrics

### Extending ESCO Integration

1. Modify `src/ingest/esco_loader.py`
2. Add new relation types
3. Update skill distance calculations
4. Test with `notebooks/quick_demo.ipynb`

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📚 Documentation

- **Architecture**: `docs/architecture.md`
- **API Reference**: `docs/api.md`
- **Evaluation Guide**: `docs/evaluation.md`
- **Deployment Guide**: `docs/deployment.md`

## 🐛 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install sentence-transformers faiss-cpu
   ```

2. **CUDA Issues**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Data Not Found**
   ```bash
   python src/ingest/download_and_prepare.py --create-sample
   ```

### Performance Optimization

- Use GPU for model training: `--device cuda`
- Enable FAISS GPU: `pip install faiss-gpu`
- Increase batch size for faster training
- Use model checkpointing for long training runs

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Citation

If you use this system in your research, please cite:

```bibtex
@software{career_recommender_2024,
  title={Knowledge-Aware Hybrid Recommender for Sustainable Career Path and Proactive Upskilling},
  author={Career Recommender Team},
  year={2024},
  url={https://github.com/Dhanu1110/hybrid-recommender-for-career-path/}
}
```



**Note**: This system is designed for research and educational purposes. For production deployment, ensure proper data privacy compliance and security measures.
