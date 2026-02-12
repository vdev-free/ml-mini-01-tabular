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

## Mini03 — Customer Segmentation (KMeans)

**Goal:** segment customers by behavior (purchases, spend) to enable targeted marketing actions.

**Features:** `purchases_30d`, `spend_30d`  
**Model:** KMeans (k=3) with `StandardScaler`  
**Metric:** Silhouette score ≈ **0.623** (higher is better)

**Segments (mean values):**
- **VIP:** ~18 purchases / ~1243 spend
- **Regular:** ~9 purchases / ~365 spend
- **Low:** ~2 purchases / ~75 spend

**Business actions:**
- VIP → loyalty program / early access / premium support
- Regular → bundles / upsells / “free shipping over X”
- Low → onboarding emails / first-purchase coupon / win-back campaigns

Artifact: `artifacts/mini03/segments.png`

### Mini03 — Customer Segmentation Service

- KMeans clustering (3 segments)
- FastAPI service (`POST /segment`)
- Next.js demo UI
- Artifacts stored in `artifacts/mini03`
