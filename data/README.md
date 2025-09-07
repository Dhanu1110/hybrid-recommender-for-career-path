# Data Directory Structure and Requirements

This directory contains the data files required for the Career Recommender System. The system expects specific data formats for optimal performance.

## Directory Structure

```
data/
├── raw/                    # Raw input data files
│   ├── karrierewege.csv   # Career path sequences
│   ├── esco_occupations.csv
│   ├── esco_skills.csv
│   └── esco_relations.csv
├── processed/              # Processed data (auto-generated)
│   ├── career_paths.parquet
│   ├── esco_occupations.parquet
│   ├── esco_skills.parquet
│   ├── esco_relations.parquet
│   └── esco_cache.pkl
└── README.md              # This file
```

## Required Input Files

### 1. karrierewege.csv

Career transition data containing user career sequences.

**Required Columns:**
- `user_id` (string): Unique identifier for each user
- `job_sequence` (string): Comma-separated list of job titles/IDs in chronological order
- `timestamp` (string): When the career path was recorded (ISO format preferred)

**Example:**
```csv
user_id,job_sequence,timestamp
user_001,"software_engineer,senior_software_engineer,tech_lead",2023-01-01
user_002,"data_analyst,data_scientist,senior_data_scientist",2023-01-02
user_003,"marketing_assistant,marketing_specialist,marketing_manager",2023-01-03
```

**Data Quality Requirements:**
- Minimum sequence length: 2 jobs
- Maximum sequence length: 50 jobs (configurable)
- Job titles should be consistent across users
- Timestamps should be valid dates

### 2. esco_occupations.csv

ESCO occupation/job data.

**Required Columns:**
- `esco_id` (string): Unique ESCO occupation identifier
- `title` (string): Human-readable occupation title

**Optional Columns:**
- `description` (string): Detailed occupation description
- `isco_code` (string): ISCO classification code
- `alternative_labels` (string): Alternative names for the occupation

**Example:**
```csv
esco_id,title,description
occ_001,Software Engineer,Develops and maintains software applications
occ_002,Data Scientist,Analyzes complex data to extract business insights
occ_003,Marketing Manager,Plans and executes marketing strategies
```

### 3. esco_skills.csv

ESCO skills and competences data.

**Required Columns:**
- `esco_id` (string): Unique ESCO skill identifier
- `title` (string): Human-readable skill title

**Optional Columns:**
- `description` (string): Detailed skill description
- `skill_type` (string): Type of skill (knowledge, skill, competence)
- `alternative_labels` (string): Alternative names for the skill

**Example:**
```csv
esco_id,title,description,skill_type
skill_001,Python Programming,Programming in Python language,skill
skill_002,Data Analysis,Statistical analysis of datasets,skill
skill_003,Digital Marketing,Online marketing strategies and tools,competence
```

### 4. esco_relations.csv

Relationships between ESCO entities (occupations and skills).

**Required Columns:**
- `source_id` (string): Source ESCO ID (occupation or skill)
- `target_id` (string): Target ESCO ID (occupation or skill)
- `relation_type` (string): Type of relationship

**Supported Relation Types:**
- `requires`: Occupation requires skill
- `related_to`: General relatedness
- `broader`: Hierarchical relationship (target is broader than source)
- `narrower`: Hierarchical relationship (target is narrower than source)

**Example:**
```csv
source_id,target_id,relation_type
occ_001,skill_001,requires
occ_002,skill_002,requires
skill_001,skill_004,broader
```

## Data Sources

### Karrierewege Data
- **LinkedIn Career Paths**: Export career transition data
- **Job Board APIs**: Indeed, Monster, Glassdoor career progression data
- **HR Systems**: Internal company career progression records
- **Survey Data**: Self-reported career transition surveys

### ESCO Data
- **Official ESCO Portal**: https://ec.europa.eu/esco/portal/download
- **ESCO API**: https://ec.europa.eu/esco/api
- **RDF Downloads**: Full ESCO taxonomy in RDF format
- **CSV Exports**: Simplified CSV exports from ESCO portal

## Data Preparation Steps

### 1. Download ESCO Data

```bash
# Download from ESCO portal
wget https://ec.europa.eu/esco/api/resource/download?uri=...

# Or use the ESCO API
curl "https://ec.europa.eu/esco/api/suggest?text=software&type=occupation"
```

### 2. Process Raw Data

```bash
# Create sample data for testing
python src/ingest/download_and_prepare.py --create-sample

# Process real data
python src/ingest/download_and_prepare.py
```

### 3. Validate Data Quality

The ingestion pipeline automatically validates:
- Required columns are present
- Data types are correct
- No duplicate IDs
- Sequence lengths are within bounds
- Relations reference valid entities

## Data Privacy and Ethics

### Privacy Considerations
- **Anonymization**: Remove personally identifiable information
- **Aggregation**: Consider aggregating rare career paths
- **Consent**: Ensure proper consent for career data usage
- **Retention**: Implement data retention policies

### Bias Mitigation
- **Representation**: Ensure diverse career paths are included
- **Historical Bias**: Account for historical gender/demographic biases
- **Geographic Diversity**: Include global career patterns
- **Industry Coverage**: Represent all major industry sectors

## Troubleshooting

### Common Data Issues

1. **Missing Files**
   ```
   Error: File not found: data/raw/karrierewege.csv
   Solution: Create sample data with --create-sample flag
   ```

2. **Invalid Sequences**
   ```
   Warning: Sequence too short, skipping user_123
   Solution: Ensure sequences have at least 2 jobs
   ```

3. **Missing Relations**
   ```
   Warning: Job occ_999 not found in relations
   Solution: Add missing job-skill relationships
   ```

4. **Encoding Issues**
   ```
   Error: UnicodeDecodeError
   Solution: Ensure CSV files are UTF-8 encoded
   ```

### Data Quality Checks

Run data validation:
```bash
python src/ingest/validate_data.py --data-dir data/raw
```

Check processed data:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/career_paths.parquet')
print(f'Processed {len(df)} career paths')
print(f'Average sequence length: {df.sequence_length.mean():.2f}')
"
```

## Sample Data Generation

For testing and development, use the built-in sample data generator:

```bash
# Generate minimal sample data
python src/ingest/download_and_prepare.py --create-sample

# Generate larger sample dataset
python src/ingest/generate_synthetic_data.py --num-users 1000 --num-jobs 100
```

## Performance Considerations

### Large Datasets
- **Chunking**: Process large files in chunks
- **Parallel Processing**: Use multiprocessing for data ingestion
- **Memory Management**: Monitor memory usage during processing
- **Incremental Updates**: Support incremental data updates

### Optimization Tips
- Use Parquet format for faster I/O
- Index frequently queried columns
- Cache processed ESCO relationships
- Precompute skill distances for common pairs

## Data Updates

### Regular Updates
- **ESCO Updates**: ESCO releases updates annually
- **Career Data**: Update career sequences monthly/quarterly
- **Skill Trends**: Monitor emerging skills and technologies

### Update Process
1. Download new ESCO version
2. Map old IDs to new IDs
3. Reprocess career sequences
4. Retrain models if necessary
5. Update skill distance cache

---

**Note**: Always backup your data before running processing scripts. The system creates processed files automatically but does not modify raw input files.
