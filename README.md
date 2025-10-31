# AI Project - NFWBS Financial Well-Being Analysis

Machine learning project for analyzing and predicting financial well-being using the National Financial Well-Being Survey (NFWBS) dataset.

## Project Structure

```
AI-Project/
├── config/                     # Configuration files
├── data/
│   ├── raw/                   # Original, immutable data
│   └── processed/             # Cleaned, transformed data
├── docs/                      # Documentation and data codebooks
├── notebooks/
│   ├── experiments/           # Model experiments and comparisons
│   ├── explainability/        # Model explainability (XAI)
│   └── exploratory/           # Exploratory data analysis
├── src/
│   ├── api/                   # API and application interfaces
│   ├── data/                  # Data processing modules
│   ├── models/                # Model definitions
│   └── utils/                 # Utility functions
├── tests/                     # Unit and integration tests
└── requirements.txt           # Project dependencies
```

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd AI-Project
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
cp .env.example .env  # Create .env file with your API keys
```

## Usage

### Running the Streamlit Application
```bash
streamlit run src/api/streamlit_app.py
```

### Data Processing
```bash
python src/data/data_cleaning.py
python src/data/eda.py
```

## Notebooks

- **experiments/** - Model training, fine-tuning, and comparison notebooks
  - `end-to-end.ipynb` - Complete ML pipeline
  - `fine_tuning.ipynb` - Model fine-tuning experiments
  - `model_comparison.ipynb` - Compare different model architectures
  - `open_ai.ipynb` - OpenAI integration experiments

- **explainability/** - Model interpretation and explainability
  - `Xai.ipynb` - LIME/SHAP analysis for model interpretability

- **exploratory/** - Data exploration and visualization
  - `univariate.html` - Univariate analysis report

## Data

The project uses the **NFWBS (National Financial Well-Being Survey) PUF 2016** dataset.
- Data codebook: `docs/cfpb_nfwbs-puf-codebook.pdf`
- Raw data: `data/raw/NFWBS_PUF_2016_data.csv`

## Key Features

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Machine learning model training and evaluation
- Model explainability using LIME
- OpenAI integration for enhanced analysis
- Interactive Streamlit dashboard

## Technologies

- **ML/Data Science**: TensorFlow, scikit-learn, pandas, numpy
- **Visualization**: Plotly, Seaborn
- **Web App**: Streamlit
- **Explainability**: LIME
- **APIs**: OpenAI
- **Cloud**: AWS (boto3)

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## License

[Add your license here]

## Contact

[Add your contact information here]
