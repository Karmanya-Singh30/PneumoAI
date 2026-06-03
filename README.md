# PneumoAI

AI-powered respiratory disease detection system using lung sound analysis, signal processing, and machine learning.

## Overview

PneumoAI is an AI-powered respiratory health assessment system that analyzes lung sounds and classifies respiratory abnormalities using machine learning.

The system processes respiratory audio recordings, extracts acoustic features, and predicts whether a breathing cycle is normal or abnormal. It further estimates disease probabilities for respiratory conditions such as COPD, Pneumonia, Asthma, and URTI.


## Key Features

- Automated lung sound classification
- Respiratory abnormality detection
- Disease probability estimation
- Audio signal preprocessing and segmentation
- Class imbalance mitigation using SMOTETomek
- Explainable and lightweight machine learning architecture


## Tech Stack

- Python
- Scikit-Learn
- Librosa
- NumPy
- Pandas
- Random Forest
- SMOTETomek
- Matplotlib

## Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 95.94% |
| Macro F1 Score | 82.05% |

## Screenshots

### Home Page
![Home Page](screenshots/ui.png)

### Probability breakdown
![Upload Page](screenshots/probability_breakdown.png)

The model achieved high sensitivity for detecting abnormal respiratory conditions while maintaining strong overall classification performance.

