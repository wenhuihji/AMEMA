# AMEMA: Adaptive Momentum and EMA-weighted Modeling for Imbalanced Label Distribution Learning

This repository contains the **code** and **appendix** materials for our paper:

**"Adaptive Momentum and EMA-weighted Modeling for Imbalanced Label Distribution Learning"**
Authors:Yongbiao Gao* , Xiangcheng Sun*, Chao Tan, Chunyu Hu, Guohua Lv  
(*These authors contributed equally.)

---

## 📘 Overview

**AMEMA** is a framework designed to address **imbalanced label distribution learning (LDL)** problems.  
It integrates **adaptive momentum adjustment** and **EMA-based weighting** to achieve a balanced optimization  
between dominant and non-dominant label components, improving model robustness and generalization.

---

## 🧩 Repository Structure

AMEMA/
├── Code and Appendix/ # Source code and appendix materials
│ ├── main.py # Main training script
│ ├── models/ # Model architectures
│ ├── utils/ # Helper functions
│ ├── datasets/ # Data loading scripts
│ └── appendix.pdf # Full appendix 
└── README.md # This file

---

## 🚀 Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy ≥ 1.19  
- SciPy, Matplotlib  
- CUDA-compatible GPU (recommended for training)

Install dependencies with:

Results

AMEMA achieves significant improvements on multiple LDL benchmarks (e.g., Movie, SCUT-FBP, RAF-ML),
demonstrating the effectiveness of adaptive EMA-based weighting and momentum allocation strategies.

Appendix

Detailed mathematical derivations, proofs, and additional ablation studies can be found in:
appendix.pdf
