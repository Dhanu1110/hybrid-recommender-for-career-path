# Career Recommender System - Production Deployment Guide

This guide explains how to deploy and run the production-ready career recommender system.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd career-recommender
```

2. **Set up virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements-full.txt
```

### Initialize the System

1. **Create sample data (for testing):**
```bash
python src/ingest/download_and_prepare.py --create-sample
```

2. **Process the data:**
```bash
python src/ingest/download_and_prepare.py
```

3. **Train the model (optional, if you want to retrain):**
```bash
python src/models/train_path_model.py --synthetic --epochs 5
```

### Run the Application

```bash
streamlit run app/streamlit_app.py
```

The application will be available at `http://localhost:8501`

## 🏗️ System Architecture

The production system implements a three-stage cascaded architecture:

### Stage 1: Path Generation (BERT4Rec)
- Uses a trained transformer model to generate candidate career paths
- Based on user's job history
- Produces probabilistic next-job recommendations

### Stage 2: Skill Gap Reasoning (ESCO)
- Analyzes skill gaps using the ESCO knowledge graph
- Calculates feasibility scores based on required vs. possessed skills
- Combines model confidence with feasibility for final scoring

### Stage 3: Resource Recommendation
- Recommends learning resources to bridge skill gaps
- Content-based filtering with semantic similarity

## 🧪 Testing

### Model Inference Test
```bash
python test_model_inference.py
```

### Full Pipeline Test
```bash
python test_full_pipeline.py
```

## 🛠️ Key Features

### Production-Ready Components
- **Actual Model Inference**: Uses the trained BERT4Rec model instead of mock data
- **Dynamic Path Generation**: Generates personalized recommendations based on user input
- **ESCO Integration**: Leverages real ESCO knowledge graph for skill gap analysis
- **Text Mapping**: Maps user input to ESCO taxonomy using semantic similarity

### Improvements Over Demo Version
1. **Personalized Recommendations**: No longer uses hardcoded paths
2. **Real Model Integration**: Loads and uses the actual trained model
3. **Dynamic Candidate Generation**: Generates recommendations based on user's job history
4. **Enhanced User Experience**: More accurate and personalized results

## 📁 Project Structure

```
career-recommender/
├── app/                    # Streamlit web application
├── configs/                # Configuration files
├── data/                   
│   ├── raw/               # Raw input data
│   └── processed/         # Processed Parquet files
├── models/                 # Trained model artifacts
├── src/                   
│   ├── ingest/            # Data ingestion and preprocessing
│   ├── models/            # ML models
│   ├── reasoner/          # Skill gap reasoning
│   └── resources/         # Resource recommendation
├── tests/                  # Unit tests
└── requirements-full.txt   # Full dependencies
```

## ⚙️ Configuration

### System Configuration
Located at `configs/system_config.yaml`

Key parameters:
- `alpha`, `beta`: Weights for combining model probability and feasibility score
- Model paths and parameters

### Model Configuration
Located at `configs/bert4rec_config.json`

Key parameters:
- Model architecture (layers, heads, dimensions)
- Training parameters

## 🐛 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements-full.txt
   ```

2. **Model Files Not Found**
   Ensure you've run the training script or have the model files in the correct location

3. **CUDA Issues**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

### Performance Optimization

- Use GPU for model inference: Set `device: cuda` in configuration
- Enable FAISS GPU: `pip install faiss-gpu`
- Increase batch size for faster processing

## 📈 Monitoring

The system includes logging for monitoring:
- Model loading and inference
- Data processing
- Error tracking

Logs are configured in `configs/system_config.yaml`

## 🔄 Maintenance

### Updating the Model
1. Retrain with new data:
   ```bash
   python src/models/train_path_model.py --data-path data/processed/career_paths.parquet --epochs 50
   ```

### Updating ESCO Data
1. Place new ESCO files in `data/raw/`
2. Run data ingestion:
   ```bash
   python src/ingest/download_and_prepare.py
   ```

## 📚 API Reference

### Core Components

**ESCO Loader**
```python
from ingest.esco_loader import create_esco_loader
loader = create_esco_loader('data/processed')
```

**BERT4Rec Model**
```python
from models.bert4rec import create_bert4rec_model
model = create_bert4rec_model(config)
```

**Skill Gap Analyzer**
```python
from reasoner.skill_gap import create_skill_gap_analyzer
analyzer = create_skill_gap_analyzer('configs/system_config.yaml')
```

## 📞 Support

For issues with the production deployment, please check:
1. All dependencies are installed
2. Model files are in the correct location
3. Data has been processed correctly
4. Configuration files are properly set up