# Rakuten Product Classification – Multimodal

This repository contains a **multimodal product classification system**
developed as part of a data science / MLOps training project, based on the [Rakuten France Multimodal Product Classification challenge](https://challengedata.ens.fr/challenges/35).

The objective is to predict the **Rakuten product category (27 categories)** by jointly leveraging
textual information and product images through a late-fusion, stacking-based architecture.

The project emphasizes not only model performance, but also **clean architecture,
reproducibility, and deployment readiness**.

---

## Project Overview

The system takes as input a product’s designation, optional description, and image, and outputs
a **predicted product category**.

For demonstration and interpretability purposes, the Streamlit interface also displays
confidence scores and modality-level contributions.

---

## Model Architecture

### Text Modality

- CamemBERT (French transformer)
- XLM-R (multilingual transformer)
- TF-IDF + linear SVM

### Image Modality

- ConvNeXt (modern convolutional architecture)
- Swin Transformer (hierarchical vision transformer)

### Fusion Strategy

1. Per-modality probability calibration and weighted late fusion
2. Meta-classifier (stacking) on concatenated probabilities

---

## Project Structure

```text
.
│
├── app/
│   ├── app.py                  # Streamlit demo application
│   ├── config.yaml             # Demo configuration (local inference / API switch)
│   └── demo_images/            # Images used for the demo
│
├── api/
│   ├── main.py                 # FastAPI inference service
│   ├── config.yaml             # API configuration
│   ├── openapi.json            # OpenAPI specification
│   └── doc.html                # Static documentation
│
├── artifacts/
│   └── canonical_classes.json  # Canonical label definition
│
├── src/
│   ├── config/
│   │   └── registry.yaml       # Logical model registry
│   │
│   ├── data/
│   │   ├── categories.py       # Category names & groups
│   │   ├── image_loading.py    # Image loading utilities
│   │   ├── image_paths.py      # Image path helpers
│   │   └── label_mapping.py    # Canonical label alignment
│   │
│   ├── features/
│   │   └── text/
│   │       ├── cleaning/        # Text cleaning transformers
│   │       ├── numeric_tokens/  # Numeric token handling
│   │       └── utils.py
│   │
│   ├── inference/
│   │   ├── load_model.py       # Multimodal model loader
│   │   └── predictor.py        # Inference logic
│   │
│   ├── models/
│   │   ├── text/               # Text model loaders & definitions
│   │   └── image/              # Image model loaders & definitions
│   │
│   ├── pipeline/
│   │   ├── text.py             # Text inference pipeline
│   │   ├── image.py            # Image inference pipeline
│   │   ├── blending.py         # Late-fusion logic
│   │   └── multimodal.py       # Final multimodal pipeline
│   │
│   └── utils/
│       ├── calibration.py      # Probability calibration
│       ├── hf.py               # Hugging Face Hub helpers
│       └── io.py               # Artifact loading utilities
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Reproducibility & Deployment

- Docker is used to ensure full environment reproducibility.
- Model artifacts are retrieved from the Hugging Face Hub at runtime.
- The Streamlit demo is deployed on **Hugging Face Spaces**.

Live demo:  
https://beautiful-pixel-rakuten-product-classification-demo.hf.space

---

## Optional: API-Based Inference

The project also supports an optional API-based inference mode (FastAPI),
intended for production deployments. The API is not deployed.

---

## Exploratory Work

Exploratory data analysis, feature engineering experiments,
early modeling iterations, **as well as a written report produced as part of the training program**,  
are available in a separate repository:

Exploration repository:  
https://github.com/beautiful-pixel/DS_rakuten

This separation keeps the present repository focused on the **final architecture,
inference pipeline, and deployment-ready implementation**.

---
