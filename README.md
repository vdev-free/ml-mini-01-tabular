# Mini-project 01 — Tabular Classification

## Goal
Build a small production-style ML project for binary classification on tabular data.
The goal is to predict whether a tumor is malignant or benign.

## Dataset
We use the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.
The dataset contains numeric features computed from digitized images of breast tissue.

## Scope
- Train a baseline classification model
- Use a reproducible pipeline
- Track experiments with MLflow
- Expose inference via API (later)

## How to run

### 1) Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2) Install dependencies
```bash
pip install -e .
```

### 3) Run baseline training
```bash
python -m mini01.baseline
```

### 4) Run tests
```bash
python -m pytest -q
```

## Results (baseline)
- Model: `StandardScaler + LogisticRegression`
- Split: train/test = 80/20, stratified
- Metric: accuracy
- Accuracy (test): ~0.982
