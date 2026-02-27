# Rakuten Product Classification – Multimodal

This repository contains a **multimodal product classification system**
developed as part of a data science / MLOps training project, based on the [Rakuten France Multimodal Product Classification challenge](https://challengedata.ens.fr/challenges/35).

The objective is to predict the **Rakuten product category (27 categories)** by jointly leveraging
textual information and product images.

The project emphasizes strong modeling design, structured experimentation,
and a clean separation between research and inference.

---

## Project Overview

Given:

- a product designation
- an optional description
- a product image

The system outputs:

- a predicted product category
- calibrated confidence scores
- modality-level contributions

The final multimodal model achieved a **weighted F1-score of 0.9273**  
on the official Rakuten test set.

Although the challenge took place in 2020, the model was submitted retrospectively  
and exceeded the best historical public score (0.9196).

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
├── app/                      # Streamlit demo
│   ├── app.py
│   ├── config.yaml
│   └── demo_images/
│
├── api/                      # FastAPI inference service
│   ├── main.py
│   ├── config.yaml
│   ├── openapi.json
│   └── doc.html
│
├── artifacts/                # Static artifacts (canonical classes)
│   └── canonical_classes.json
│
├── src/                      # Production-ready inference pipeline
│   ├── config/               # Logical model registry
│   ├── data/                 # Data utilities (labels, image loading)
│   ├── features/             # Final preprocessing logic
│   ├── inference/            # Model loading & prediction
│   ├── models/               # Model definitions (text / image)
│   ├── pipeline/             # Text / Image / Multimodal pipelines
│   └── utils/                # Calibration & I/O helpers
│
├── research/                 # Research & experimentation phase
│   ├── notebooks/            # EDA & modeling notebooks
│   ├── exploratory/          # Experimental training code
│   └── requirements-exploration.txt
│
├── rapport.pdf               # Written modeling report
├── Dockerfile
├── requirements.txt          # Inference & deployment dependencies
└── README.md
```

## Exploration Report (in French)

A detailed methodological report is available at the root of the repository:

`rapport.pdf`

It documents the full modeling journey, including:

- exploratory data analysis (text and image)
- vectorization optimization
- text Transformer fine-tuning and preprocessing experiments
- CNN and Vision Transformer fine-tuning with transfer learning
- intra-modality fusion (text-only, image-only)
- final multimodal stacking strategy
- detailed error analysis and category-level evaluation

The report summarizes the experimental reasoning and modeling decisions that led to the final architecture.

---

## Research Notebooks

Exploratory notebooks and experimental training code are available in `research/`.

This directory contains the experimentation layer used during model development,
separate from the refactored inference pipeline in `src/`.

---

## Demo & Deployment

- Docker is used for environment consistency.
- Model artifacts are retrieved from the Hugging Face Hub at runtime.
- The Streamlit demo is deployed on Hugging Face Spaces.

Live demo:  
https://beautiful-pixel-rakuten-product-classification-demo.hf.space

---

## Optional API

A FastAPI inference service is included for production-style deployment scenarios.
