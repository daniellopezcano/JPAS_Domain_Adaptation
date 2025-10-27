# JPAS_Domain_Adaptation

**Companion code for “Domain Adaptation for Sim-to-Obs Astrophysics: From DESI Mocks to J-PAS Observations.”**  
*A ready-to-use, reproducible pipeline for semi-supervised domain adaptation (SSDA) in astrophysics, and cosmology.*

License: MIT

---

## 🔭 Purpose

This repository provides an end-to-end pipeline to transfer machine-learned classifiers from **simulations** to **observations** under domain shift. Concretely, it implements an SSDA workflow that pretrains on **DESI→J-PAS pseudo-spectra mocks** and adapts to **J-PAS observations** using a small labeled subset.

**Task:** four-way pseudo-spectral classification  
`QSO_high (z ≥ 2.1)`, `QSO_low`, `GALAXY`, `STAR`.

**Why it matters:** Purely sim-trained models typically degrade on real data due to differences in passbands, calibration, and observing conditions. With modest target supervision, this pipeline closes most of that gap while preserving probability calibration and improving rare, science-critical classes.

---

## 🧪 Results at a glance (reproduced by this repo)

Held-out J-PAS test split (evaluation-only cut \(r \le 22.5\)):

- **Macro-F1:** 0.82 (SSDA) vs. 0.79 (target-only, same label budget) vs. 0.73 (zero-shot mocks)  
- **TPR:** 0.89 (SSDA)	**Macro-AUROC:** 0.975 (SSDA)	**ECE:** ≈ 0.05  
- **High-z QSO F1:** 0.66 (SSDA) vs. 0.55 (target-only) vs. 0.37 (zero-shot)

Configs and scripts to reproduce these numbers are included.

---

## 🧰 Features

- **End-to-end SSDA**: pretrain on mocks → adapt on a few target labels → evaluate.
- **Config-driven experiments** (YAML) with saved checkpoints & optional W&B tracking.
- **Balanced cross-entropy (BaCE)** for class imbalance; weight decay, dropout, grad-clipping.
- **Leak-safe splits** and **source-only normalization** to avoid target leakage.
- **Morphology integration** via one-hot *The Tractor* flags concatenated to photometry.
- **Plotting utilities** to regenerate figures: confusion matrices, ROC/AUC, calibration, etc.

---

## 🗂️ Repository structure

```
JPAS_Domain_Adaptation/
│
├── JPAS_DA/                 # Core package: data, models, training, evaluation, plots
├── configs/
├── notebooks/               # Optional demos & sanity checks (incl. toy walkthrough)
├── scripts/                 # CLI entrypoints for train/eval/plots
├── DATA/                    # Place datasets here (git-ignored)
├── README.md
├── setup.py
└── requirements.txt
```

---

## 📦 Installation

```bash
git clone https://github.com/daniellopezcano/JPAS_Domain_Adaptation
cd JPAS_Domain_Adaptation
pip install -e .
```

Requirements: Python ≥ 3.10, PyTorch (CUDA if available). `wandb` is optional.

---

## 📁 Data access & the included **toy pipeline**

> **Important:** The production datasets used in the paper (J-PAS observations and DESI→J-PAS mocks / cross-match) are **not distributed** with this repository.

To ensure the code runs out-of-the-box and to make the pipeline easy to understand, we provide a **toy pipeline** that mirrors the full workflow with small synthetic data.

This is sufficient to verify your environment and to understand how to adapt the code to your own surveys.

---

## 🔁 Reproducing paper results (requires data access)

If you have access to the production datasets:

1. **Prepare data** under `DATA/` following the schema in `configs/paper/`.  
2. **Pretrain on DESI→J-PAS mocks** (checkpoint selected by mock-validation BaCE).  
3. **SSDA on J-PAS** (freeze head; adapt encoder with the small labeled J-PAS train split).  
4. **Baselines:** target-only J-PAS supervised; zero-shot mocks→J-PAS.  
5. **Metrics & plots:** confusion matrices, per-class/macro F1, ROC/AUC with CIs, ECE.

All seeds/hyperparameters are in `configs/paper/`; top configs cluster tightly, indicating stable convergence.

---

## 👥 Who should use this

- Survey ops & target selection (AGN/QSO candidates; star–galaxy–QSO demixing).  
- LSS/clustering studies needing calibrated class probabilities under domain shift.  
- ML practitioners seeking a clean SSDA reference for sim-to-obs transfer.

---

## 📝 Citation

If you use this code, please cite the paper and this repository:

```bibtex
@misc{JPAS_Domain_Adaptation_Code,
  title  = {JPAS\_Domain\_Adaptation: Companion codebase for SSDA sim-to-obs transfer},
  author = {L{'o}pez-Cano, Daniel},
  year   = {2025},
  url    = {https://github.com/daniellopezcano/JPAS_Domain_Adaptation}
}
```

---

## 📬 Contact

- Maintainer: Daniel López-Cano — <daniellopezcano13@gmail.com>  
- For scientific questions, open a GitHub issue referencing the relevant section/figure.

---

## 📜 License

MIT © 2025 Daniel López-Cano. See `LICENSE` for details.

---

*Goal: make domain-aware learning standard practice in astro pipelines by providing a transparent, reproducible SSDA baseline that others can adapt to their surveys—even without access to proprietary data—via the included toy pipeline.*
