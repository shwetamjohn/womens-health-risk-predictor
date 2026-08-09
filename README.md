# Women's Health Risk Predictor 🩺

> **Machine learning project for breast cancer risk prediction using clinical cell-level features — promoting early detection and awareness.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project trains and evaluates multiple machine learning models to classify breast masses as **Benign** or **Malignant** based on 9 clinical features derived from fine needle aspirate (FNA) biopsies. It includes a full data science pipeline and a **standalone web dashboard** for interactive predictions.

The project uses the [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) from the UCI Machine Learning Repository.

---

## Demo

Open `app/index.html` in any browser to use the interactive predictor — no server needed!

---

## Project Structure

```
womens-health-risk-predictor/
│
├── data/
│   └── dataset.csv               # Wisconsin Breast Cancer Dataset (699 samples)
│
├── notebooks/
│   ├── exploration.ipynb         # Initial data exploration
│   └── model_training.ipynb      # Full ML pipeline with visualizations
│
├── src/
│   └── train.py                  # Standalone training script
│
├── models/
│   ├── breast_cancer_model.pkl   # Best trained model + scaler (generated)
│   └── model_coefficients.json   # LR coefficients for web app (generated)
│
├── reports/
│   ├── model_comparison.png      # Bar chart of model metrics (generated)
│   ├── roc_curves.png            # ROC curves for all models (generated)
│   ├── confusion_matrix.png      # Confusion matrix for best model (generated)
│   └── feature_importance.png    # Feature importances (generated)
│
├── app/
│   ├── index.html                # Interactive web dashboard
│   ├── style.css                 # Premium dark-mode UI styles
│   └── app.js                    # Prediction logic (runs in browser)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Source**: Dr. William H. Wolberg, University of Wisconsin Hospitals  
**UCI Reference**: [Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

### Features (1–10 scale, 1 = closest to normal)

| Feature | Description |
|---|---|
| `clump_thickness` | Thickness of cell clumps |
| `size_uniformity` | Uniformity of cell size |
| `shape_uniformity` | Uniformity of cell shape |
| `marginal_adhesion` | Tendency of cells to clump |
| `epithelial_size` | Single epithelial cell size |
| `bare_nucleoli` | Proportion of bare nuclei (16 missing values) |
| `bland_chromatin` | Chromatin texture uniformity |
| `normal_nucleoli` | Nucleoli prominence |
| `mitoses` | Mitotic activity (cell division rate) |

### Target Variable

| Class | Encoding | Count |
|---|---|---|
| Benign | 0 | 458 (65.5%) |
| Malignant | 1 | 241 (34.5%) |

---

## Models & Results

Five classifiers were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | ~97.1% | ~97.0% | ~95.8% | ~96.4% | ~99.5% |
| Gradient Boosting | ~97.1% | ~96.7% | ~96.7% | ~96.7% | ~99.4% |
| Logistic Regression | ~95.7% | ~95.2% | ~95.8% | ~95.5% | ~99.2% |
| SVM | ~97.1% | ~97.0% | ~95.8% | ~96.4% | ~99.3% |
| KNN | ~95.0% | ~95.0% | ~93.4% | ~94.2% | ~98.5% |

> Results may vary slightly due to random state and train/test split.

---

## Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/womens-health-risk-predictor.git
cd womens-health-risk-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/train.py
```

This will:
- Load and preprocess the dataset
- Train 5 ML classifiers
- Print evaluation metrics
- Save the best model to `models/breast_cancer_model.pkl`
- Save model coefficients to `models/model_coefficients.json`
- Generate visualizations in `reports/`

### 4. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/model_training.ipynb
```

The notebook includes:
- Full EDA with interactive charts
- Feature distribution analysis
- Correlation heatmap
- ROC curves for all models
- Confusion matrix
- Feature importance plots

### 5. Use the Web Dashboard

Simply open `app/index.html` in your browser — no server required!

- Adjust the 9 feature sliders (1–10)
- Click **Analyze Risk** to get a real-time prediction
- View confidence score, risk gauge, and feature impact chart

---

## Key Findings

1. **Size Uniformity** is the most predictive feature (highest RF importance)
2. **Shape Uniformity** and **Bare Nucleoli** are also highly discriminative
3. **Mitoses** has the lowest individual importance but contributes to the ensemble
4. All models achieve >95% accuracy, with Random Forest and SVM leading at ~97.1%
5. ROC-AUC scores above 0.98 indicate near-perfect class separation

---

## Methodology

```
Raw Data → Missing Value Imputation → Feature Scaling (StandardScaler)
    → Train/Test Split (80/20, stratified) → 5 Model Training
    → Cross-Validation (5-fold) → Metric Evaluation → Best Model Saved
```

**Missing Values**: The `bare_nucleoli` column contains 16 rows with `?` values, replaced with the column median.

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Disclaimer

> **This tool is for educational and research purposes only.** It should not be used for actual medical diagnosis. Always consult a qualified healthcare professional for medical advice and diagnosis.

---

*Built with Python, scikit-learn, and a mission to make healthcare AI accessible.*
