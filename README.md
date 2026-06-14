# JPAS_Domain_Adaptation

**Companion code for:**
> *J-PAS: Semi-Supervised Sim-to-Obs Transfer for Robust Star–Galaxy–Quasar Classification*
> D. López-Cano, L. R. Abramo, L. Nakazono, I. Pérez-Ràfols, G. Martínez-Solaeche, J. Chaves-Montero, M. M. Pieri, and the J-PAS collaboration.
> Submitted to *The Astrophysical Journal* (AAS Journals). Preprint: [arXiv:2602.13902](https://arxiv.org/abs/2602.13902)

---

## 🔭 Overview

This repository provides a complete, config-driven pipeline for **semi-supervised domain adaptation (SSDA)** applied to astrophysical source classification from narrow-band photometry (*J-spectra*). The pipeline transfers a four-class classifier trained on abundant labeled **DESI→J-PAS mock spectra** to real **J-PAS photometric observations**, where labeled examples are scarce, by adapting only the encoder with a small labeled target subset while keeping the classification head fixed.

**Classification task:** four-way source classification from 55-band J-spectra + Tractor morphology flag:

| Class | Definition |
|---|---|
| `QSO_high` | Quasars at *z* ≥ 2.1 (Lyα-forest quasars) |
| `QSO_low` | Quasars at *z* < 2.1 |
| `GALAXY` | Galaxies |
| `STAR` | Stars |

**Why it matters:** A model trained purely on simulated mock spectra typically degrades when applied to real observations due to differences in passbands, calibration, noise, and photometric coverage (*domain shift*). With modest target supervision (≈15,600 labeled J-PAS objects), our SSDA pipeline closes most of that gap — particularly for the rare, science-critical high-*z* quasar class — while maintaining well-calibrated class probabilities. Calibrated probabilities are operationally important for spectroscopic follow-up targeting (e.g., WEAVE-QSO), where false positives waste fibres and where the selection function must be explicit for downstream large-scale structure analyses.

---

## 🧪 Results at a glance

All numbers below are from the held-out J-PAS test subset (20,797 objects before the evaluation-only magnitude cut *r* ≤ 22.5; 14,777 after). The four columns correspond to the four evaluation regimes in the paper (three training regimes + one in-domain mock reference).

| Metric | J-PAS supervised | Mocks no-DA (in-domain) | J-PAS no-DA (zero-shot) | **J-PAS SSDA** |
|---|---|---|---|---|
| Accuracy | 0.950 | 0.934 | 0.917 | **0.952** |
| Macro-F1 | 0.792 | 0.798 | 0.729 | **0.824** |
| Macro-TPR | 0.853 | 0.889 | 0.865 | **0.894** |
| Macro-PPV | 0.755 | 0.747 | 0.697 | **0.778** |
| Macro-AUC | 0.957 | **0.977** | 0.959 | 0.975 |
| ECE (↓) | 0.054 | 0.047 | **0.045** | 0.048 |
| `QSO_high` F1 | 0.66 | — | 0.40 | **0.68** |

All configs, sweep specifications, and the best model checkpoints used to produce these numbers are included in the repository.

---

## 🗂️ Repository structure

```
JPAS_Domain_Adaptation/
│
├── JPAS_DA/                        # Core Python package
│   ├── data/                       # Data loading, cleaning, crossmatching, splitting
│   │   ├── loading_tools.py        # Modular dataset loading (JPAS, DESI mocks, Ignasi)
│   │   ├── cleaning_tools.py       # Quality cuts, QSO splitting, encoding, normalization
│   │   ├── crossmatch_tools.py     # ID-based crossmatching between survey catalogues
│   │   ├── process_dset_splits.py  # Train/val/test splitting (leak-safe)
│   │   ├── data_loaders.py         # Class-balanced batch samplers (DataLoader objects)
│   │   ├── wrapper_data_loaders.py # High-level wrapper: load → clean → crossmatch → split
│   │   └── generate_toy_data.py    # Synthetic toy dataset for smoke-testing the pipeline
│   ├── models/
│   │   └── model_building_tools.py # MLP encoder + downstream head; checkpoint I/O
│   ├── training/
│   │   ├── loss_functions.py       # Balanced cross-entropy (class-inverse-frequency weights)
│   │   ├── training_tools.py       # Training loop, AdamW + ReduceLROnPlateau, grad-clipping
│   │   └── save_load_tools.py      # Model serialization helpers
│   ├── evaluation/
│   │   └── evaluation_tools.py     # Confusion matrices, F1/TPR/PPV, ROC/AUC, ECE
│   ├── utils/
│   │   ├── plotting_utils.py       # All figure-generation functions (paper + diagnostics)
│   │   └── aux_tools.py            # Miscellaneous utilities
│   ├── wrapper_wandb/
│   │   ├── wrapper_tools.py        # W&B sweep runner (architecture + optimization sweeps)
│   │   └── wandb_tools.py          # Sweep initialization and logging utilities
│   └── global_setup.py             # Central config: paths, dataset manifests, cleaning/split options
│
├── configs/                        # All YAML configuration files
│   ├── config_no_DA.yaml           # Best no-DA model (source pretraining)
│   ├── config_DA.yaml              # Best SSDA model (encoder-only adaptation)
│   ├── config_supervised.yaml      # Best J-PAS-supervised baseline
│   ├── config_continue_training_no_DA.yaml
│   ├── wandb_no_DA.yaml            # W&B sweep spec: no-DA regime (200 configurations)
│   ├── wandb_DA.yaml               # W&B sweep spec: SSDA regime (200 configurations)
│   ├── wandb_supervised.yaml       # W&B sweep spec: supervised regime (200 configurations)
│   ├── aux_wandb_no_DA.yaml        # Architecture presets for the no-DA sweep
│   ├── aux_wandb_supervised.yaml   # Architecture presets for the supervised sweep
│   └── aux_wandb_DA_*.yaml         # Per-initialization configs for the 5 top no-DA seeds
│                                   #   used to initialize the SSDA sweep
│
├── notebooks/                      # Step-by-step walkthrough notebooks (13 total)
│   ├── 00_loading_tools.ipynb
│   ├── 01_cleaning_tools.ipynb
│   ├── 02_crossmatch_tools.ipynb
│   ├── 03_process_dset_splits.ipynb
│   ├── 04_data_loaders.ipynb
│   ├── 05_wrapper_data_loaders.ipynb
│   ├── 06_training_tools.ipynb
│   ├── 07_training_tools_domain_adaptation.ipynb
│   ├── 08_evaluation_tools.ipynb
│   ├── 09_wrapper_tools.ipynb
│   ├── 10_evaluation_models.ipynb
│   ├── 11_wandb_tools.ipynb
│   └── explore_datasets_and_compute_normalization.ipynb
│
├── SAVED_models/                   # Trained model checkpoints
│   ├── 09_no_DA/                   # Best no-DA pretrained model (paper results)
│   ├── 09_DA/                      # Best SSDA model (paper results)
│   ├── 09_supervised/              # Best supervised baseline (paper results)
│   ├── wandb_no_DA/                # Best W&B sweep checkpoint: no-DA
│   │   └── exalted-sweep-55/
│   ├── wandb_DA/                   # Best W&B sweep checkpoint: SSDA
│   │   └── ethereal-sweep-8/
│   ├── wandb_supervised/           # Best W&B sweep checkpoint: supervised
│   │   └── different-sweep-41/
│   ├── 06_example_model/           # Example models used in notebook 06
│   ├── 06_example_model_Supervised/
│   └── 07_example_model_DA/
│
├── SAVED_FIGURES/                  # All paper figures (PDF) and diagnostic outputs
│
├── DATA/                           # Production data directory (not distributed; see below)
│
├── requirements.txt
├── setup.py
├── MANIFEST.in
└── LICENSE
```

> **Note:** `DATA/` and `SAVED_models/` are not distributed via GitHub. The best model checkpoints (`09_no_DA`, `09_DA`, `09_supervised`) and the full production datasets are archived on Zenodo alongside this repository (see [Data and archive](#data-and-archive) below).

---

## 📦 Installation

```bash
git clone https://github.com/daniellopezcano/JPAS_Domain_Adaptation
cd JPAS_Domain_Adaptation
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch (GPU strongly recommended for the production sweeps). `wandb` is optional and only needed for hyperparameter sweeps. All dependencies are listed in `requirements.txt`.

---

## 🔑 Core pipeline features

- **SSDA (encoder-only adaptation).** The no-DA pretrained encoder is re-trained on the labeled J-PAS target subset with the downstream classification head frozen, so decision boundaries learned from abundant mock labels are preserved while the encoder representation is realigned to the target domain.
- **Balanced cross-entropy loss.** Class-inverse-frequency weights ensure that the rare `QSO_high` class (≈1.4–1.8% of objects) receives appropriate training signal rather than being drowned by the abundant `GALAXY`/`STAR` classes.
- **Leak-safe splits.** The DESI×J-PAS crossmatch is used to ensure that no mock object corresponding to a J-PAS source in the evaluation split appears in the mock training pool. The `TARGETID` identifier links mock and observed catalogues.
- **Source-only normalization.** Per-band standardization statistics are computed exclusively on the mock training split and then applied identically to both domains, preventing the target distribution from influencing feature scaling.
- **Morphology integration.** One-hot *Tractor* morphology flags (PSF, REX, EXP, DEV, SER, GGAL, GPSF) from the DESI Legacy Surveys are concatenated to the 55-band flux vector, providing a coarse size/shape cue that helps disambiguate stars and QSOs from galaxies at low S/N.
- **Config-driven hyperparameter sweeps.** All three regimes (no-DA, SSDA, supervised) were optimized with independent random sweeps of 200 configurations each using Weights & Biases (`wandb`). Sweep YAML files and the best-checkpoint configs are included.
- **Complete plotting suite.** `JPAS_DA/utils/plotting_utils.py` contains all functions to regenerate the paper figures: confusion-matrix tetraptychs, per-class F1/TPR/PPV/AUC/ECE bar plots, ROC curves, radar charts, magnitude-cut trade-off plots, and the probability-vs-redshift diagnostics.

---

## 📁 Data and archive

### Toy pipeline (no data access required)

A **toy pipeline** is included to allow the code to run out-of-the-box without access to the production datasets. `JPAS_DA/data/generate_toy_data.py` generates small synthetic two-domain data that mirrors the structure of the real problem. The toy pipeline exercises every component of the stack (loading, cleaning, crossmatching, splitting, training, SSDA, evaluation, plotting) and is the recommended starting point for users who want to adapt the code to a new survey pair.

---

## 🔁 Reproducing the paper results

### Step 1 — Pretrain the no-DA model (source domain: DESI→J-PAS mocks)

```bash
# Using the best-config file directly:
python -c "
from JPAS_DA.wrapper_wandb.wrapper_tools import run_from_config
run_from_config('configs/config_no_DA.yaml')
"
```

Or run the full W&B sweep (200 configurations) with:
```bash
wandb sweep configs/wandb_no_DA.yaml
wandb agent <sweep_id>
```

Model selection criterion: minimum mock-validation balanced cross-entropy loss.
The best checkpoint is saved automatically under `SAVED_models/`.

### Step 2 — SSDA (target-domain encoder adaptation)

The SSDA step loads the best no-DA encoder, freezes the downstream head, and re-trains the encoder on the labeled J-PAS training split:

```bash
python -c "
from JPAS_DA.wrapper_wandb.wrapper_tools import run_from_config
run_from_config('configs/config_DA.yaml')
"
```

The best SSDA runs in the full sweep were initialized from the five highest-ranked no-DA models (listed in `configs/wandb_DA.yaml` under `aux_config_path`). The configuration in `configs/config_DA.yaml` corresponds to the best-performing SSDA model reported in the paper.

### Step 3 — J-PAS-supervised baseline

```bash
python -c "
from JPAS_DA.wrapper_wandb.wrapper_tools import run_from_config
run_from_config('configs/config_supervised.yaml')
"
```

### Step 4 — Evaluate and regenerate figures

Open `notebooks/10_evaluation_models.ipynb` to compute all metrics (confusion matrices, F1, TPR/PPV, AUC, ECE) and regenerate the paper figures. All plotting functions are in `JPAS_DA/utils/plotting_utils.py`.

### Step-by-step walkthrough (notebooks)

The `notebooks/` directory contains 13 notebooks that walk through the pipeline component by component, in order:

| Notebook | Contents |
|---|---|
| `00_loading_tools.ipynb` | Loading J-PAS observations and DESI mock files |
| `01_cleaning_tools.ipynb` | Quality cuts, QSO subclass splitting, encoding, normalization |
| `02_crossmatch_tools.ipynb` | ID-based crossmatching; leakage-safe split construction |
| `03_process_dset_splits.ipynb` | Train/val/test splitting; `split_LoA` mechanics |
| `04_data_loaders.ipynb` | Class-balanced batch sampling |
| `05_wrapper_data_loaders.ipynb` | High-level `wrapper_build_dataloaders` call |
| `06_training_tools.ipynb` | Training a no-DA MLP on mocks |
| `07_training_tools_domain_adaptation.ipynb` | SSDA: freeze head, adapt encoder on J-PAS |
| `08_evaluation_tools.ipynb` | Per-class metrics, confusion matrices, calibration |
| `09_wrapper_tools.ipynb` | W&B sweep runner walkthrough |
| `10_evaluation_models.ipynb` | Full paper-result evaluation and figure regeneration |
| `11_wandb_tools.ipynb` | Loading and comparing sweep results |
| `explore_datasets_and_compute_normalization.ipynb` | Dataset statistics and normalization-parameter computation |

---

## ⚙️ Key hyperparameters

The table below summarizes the swept ranges and the final selected configuration per regime (full details in Appendix A of the paper):

| Parameter | Range (swept) | no-DA | SSDA | Supervised |
|---|---|---|---|---|
| Encoder layers | 1–3 layers, 8–512 units | [512, 128, 64] | (inherited) | [512, 128, 64] |
| Latent dim | 2–32 | 26 | (inherited) | 24 |
| Head layers | 1–3 layers, 4–512 units | [64, 128, 512] | (inherited) | [512] |
| Encoder dropout | 0.001–0.2 | 0.2 | 0.2 | 0.2 |
| Head dropout | 0.001–0.2 | 0.2 | 0.2 | 0.001 |
| Learning rate | 10⁻⁵–10⁻¹ | 1.3×10⁻³ | 2.0×10⁻⁴ | 1.2×10⁻⁴ |
| Weight decay | 10⁻⁶–10⁻¹ | 7.6×10⁻⁴ | 4.2×10⁻⁴ | 1.6×10⁻² |
| ℓ₂ penalty | 10⁻¹⁰–1 | 9.5×10⁻⁷ | 3.1×10⁻⁸ | 5.4×10⁻¹⁰ |
| Grad-clip norm | 10⁻⁵–10² | 11.0 | 3.2×10⁻⁴ | 6.5×10⁻⁴ |
| Batch size | 10³–1.3×10⁵ | 68,065 | 7,032 | 14,142 |
| Batches/epoch | 2,048–16,384 | 15,290 | 2,112 | 11,596 |
| Epochs | 2,500 | 2,500 | 2,500 | 2,500 |
| Head | — | trainable | **frozen** | trainable |

All models: MLP with ReLU activations, no batch normalization, AdamW optimizer with ReduceLROnPlateau scheduler (patience 30, factor 0.3, min lr 10⁻⁸), deterministic seed 137. The SSDA sweep was initialized from the five best no-DA models; the best SSDA result (`ethereal-sweep-8`) originated from `exalted-sweep-55` (the best no-DA run). Sweep counts: 200 configurations per regime.

---

## 👥 Who should use this

- **Survey scientists and target-selection teams** — for AGN/QSO candidate selection and star–galaxy–QSO demixing under domain shift.
- **Large-scale structure and clustering studies** — needing calibrated class probabilities with an explicit selection function.
- **ML researchers in astrophysics** — seeking a clean, well-documented SSDA reference implementation for sim-to-obs transfer in photometric surveys.
- **Anyone working on cross-survey or simulation-to-observation pipelines** in astronomy or related remote-sensing domains.

---

## 📝 Citation

If you use this code or build on this work, please cite both the paper and the repository/archive:

**Paper:**
```bibtex
@article{LopezCano2026_JPAS_SSDA,
  title   = {{J-PAS}: Semi-Supervised Sim-to-Obs Transfer for Robust Star--Galaxy--Quasar Classification},
  author  = {L{\'o}pez-Cano, Daniel and Abramo, L.\ Raul and Nakazono, L. and
             P{\'e}rez-R{\`a}fols, I. and Mart{\'i}nez-Solaeche, G. and
             Chaves-Montero, J. and Pieri, Matthew M. and {J-PAS Collaboration}},
  journal = {The Astrophysical Journal},
  year    = {2026},
  note    = {Submitted. Preprint: arXiv:2602.13902}
}
```

**Code and data archive:**
```bibtex
@software{LopezCano2026_JPAS_SSDA_code,
  title   = {{JPAS\_Domain\_Adaptation}: Companion codebase and data archive for
             SSDA sim-to-obs transfer in J-PAS},
  author  = {L{\'o}pez-Cano, Daniel},
  year    = {2026},
  url     = {https://github.com/daniellopezcano/JPAS_Domain_Adaptation},
  doi     = {PLACEHOLDER -- 10.5281/zenodo.XXXXXXX}
}
```

---

## 📬 Contact

- **Maintainer:** Daniel López-Cano — daniellopezcano13@gmail.com
- For bug reports or questions about the code, please open a GitHub issue referencing the relevant notebook or module.
- For scientific questions about the method or results, please reference the relevant section or figure of the paper in your issue.

---

## 📜 License

MIT © 2026 Daniel López-Cano. See [`LICENSE`](LICENSE) for details.

---

*This repository aims to make domain-aware learning a standard, reproducible part of astrophysical source-classification pipelines — providing a transparent SSDA baseline that others can adapt to new survey pairs, independently of whether they have access to the production datasets, via the included toy pipeline.*