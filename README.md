# JPAS_Domain_Adaptation

**Domain adaptation toolkit for photometric redshift classification using JPAS and DESI data.**

License: MIT

---

## 📦 Features

- Modular encoder and classifier architecture
- Custom DataLoader for JPAS/DESI data
- Cross-entropy and contrastive (Weinberger-style) losses
- Domain-aware sampling and class balancing
- Config-driven training and evaluation scripts

---

## 🗂️ Structure

```
JPAS_Domain_Adaptation/
│
├── JPAS_DA/         # Core package: data, models, training, eval
├── notebooks/       # Demos and debugging notebooks
├── configs/         # Training configs in YAML
├── DATA/            # Your datasets (excluded from Git)
├── README.md
├── setup.py
```

---

## 🚀 Getting Started

### Install

```bash
git clone https://github.com/daniellopezcano/JPAS_Domain_Adaptation
cd JPAS_Domain_Adaptation
pip install -e .
```

### Requirements

```bash
pip install -r requirements.txt
```

---

## 📁 Notes on Data

All data lives in the `DATA/` folder, which is excluded from version control.

---

## 📜 License

MIT © 2025 Daniel Lopez Cano