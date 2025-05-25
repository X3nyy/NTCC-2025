# Comparative Analysis of Deep Learning Models for Mental Health Detection from Textual Data

## Project Overview
This project conducts a comprehensive comparison of various deep learning architectures for detecting mental health conditions from text data. We evaluate traditional sequence models (LSTM, BiLSTM, GRU), convolutional networks (CNN), and transformer-based models (BERT, RoBERTa, DistilBERT, ALBERT) to determine which approaches are most effective for mental health text classification.

## Objective
To evaluate and compare the effectiveness of various deep learning models for detecting mental health conditions from textual data, identifying which architectures provide optimal performance for clinical and support applications.

## Models
- Traditional sequence models:
  - LSTM (Long Short-Term Memory)
  - BiLSTM (Bidirectional LSTM)
  - GRU (Gated Recurrent Unit)
  - CNN (Convolutional Neural Network)
- Transformer models:
  - BERT
  - RoBERTa
  - DistilBERT
  - ALBERT

## Datasets
- Twitter Mental Health Dataset
  - Source: Kaggle Mental Health Tweets Dataset
  - ~15,000 labeled tweets
  - Classes: Depression, Anxiety, Normal

- Reddit Mental Health Dataset
  - Source: Kaggle Reddit Mental Health Dataset
  - Labelled Data:
    - Mental health-related posts with classifications
    - Categories: Anxiety, Bipolar, Depression, Mental Health
  - Raw Data:
    - Unfiltered posts from mental health subreddits
    - Subreddits: Anxiety, BPD, Depression, Mental_illness, Suicide_watch

- CLPsych Shared Task Dataset (pending approval)

## Setup
```bash
# Clone this repository
git clone https://github.com/username/mental-health-detection.git
cd mental-health-detection

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
mental_health_detection/
├── data/
│   ├── raw/                  # Original unmodified datasets
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Final processed datasets ready for modeling
├── models/
│   ├── traditional/          # LSTM, BiLSTM, GRU, CNN models
│   └── transformers/         # BERT, RoBERTa, DistilBERT, ALBERT models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_development.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py   # Scripts to download or generate data
│   │   └── preprocess.py     # Text preprocessing functions
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py # Feature engineering scripts
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py    # Model training functions
│   │   ├── predict_model.py  # Prediction functions
│   │   └── evaluate_model.py # Evaluation metrics
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py      # Visualization functions
├── reports/
│   ├── figures/              # Generated graphics and figures
│   └── weekly_reports/       # Weekly progress reports
├── README.md                 # Project overview
├── requirements.txt          # Project dependencies
└── setup.py                  # Make project pip installable
```

## Timeline
8-week project with the following phases:
1. Project setup & data acquisition
2. Data preprocessing & preparation
3. Traditional models I (LSTM & BiLSTM)
4. Traditional models II (GRU & CNN)
5. Transformer models I (BERT & RoBERTa)
6. Transformer models II (DistilBERT & ALBERT)
7. Comprehensive evaluation & analysis
8. Final analysis & documentation