# Fairness-Aware Deepfake Detection

> Eliminating Demographic Bias Through Variance-Regularized Training and Explainable AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)

## Overview

This repository contains the official implementation of our research on fairness-aware deepfake detection. We propose a variance-based fairness-aware loss function that eliminates demographic disparities while maintaining (and even improving) overall accuracy in deepfake detection systems.

### Key Findings

- Modest training data imbalance (60% reduction for one demographic group) creates substantial fairness violations (33.3% TPR disparity) despite high overall accuracy (95%).
- Strong fairness regularization (lambda = 2.0) achieves simultaneously perfect fairness AND 100% accuracy, challenging the conventional fairness-accuracy trade-off assumption.
- Integrated Gradients analysis confirms that our fairness-aware model achieves equitable predictions through consistent reasoning across demographic groups.

## Research Contributions

1. Empirical demonstration of bias emergence from controlled training data imbalance
2. Novel variance-based fairness loss function that integrates seamlessly with standard frameworks
3. Discovery of threshold effects in fairness regularization
4. Integrated explainability framework combining fairness analysis with Integrated Gradients

## Results Summary

| Experiment | Training Data | Lambda | Accuracy | TPR Disparity | FPR Disparity |
|-----------|---------------|--------|----------|---------------|---------------|
| E1: Balanced Baseline | Balanced | 0.0 | 100.00% | 0.000 | 0.000 |
| E2: Biased Baseline | Biased | 0.0 | 95.00% | 0.333 | 0.000 |
| E3: Moderate Fairness | Biased | 0.5 | 93.33% | 0.333 | 0.000 |
| E4: Strong Fairness | Biased | 2.0 | 100.00% | 0.000 | 0.000 |

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.10+
- CUDA-compatible GPU (recommended)

### Installation

```bash
git clone https://github.com/kasra-kakavand/deepfake-fairness.git
cd deepfake-fairness
pip install -r requirements.txt
```

### Running the Experiments

```bash
# Run the main experiments
python src/train.py

# Generate XAI explanations
python src/explain.py
```

## Repository Structure
deepfake-fairness/
├── src/                    # Source code
│   ├── dataset.py         # Custom dataset with demographic labels
│   ├── models.py          # EfficientNet-B0 backbone
│   ├── losses.py          # Fairness-aware loss function
│   ├── metrics.py         # Fairness evaluation metrics
│   ├── train.py           # Training pipeline
│   └── explain.py         # XAI with Integrated Gradients
├── notebooks/             # Jupyter notebooks
│   └── main_experiments.ipynb
├── results/               # Experimental results and figures
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # This file
## Methodology

### Variance-Based Fairness Loss

Our proposed fairness-aware loss function:
L_fair(theta) = L_CE(theta) + lambda * R_var(theta)
Where:
- L_CE is the standard cross-entropy loss
- R_var is the variance regularization across demographic groups
- lambda controls the strength of the fairness constraint

### Architecture

We use EfficientNet-B0 pre-trained on ImageNet, fine-tuned for binary deepfake classification.

## Citation

If you find this work useful, please cite:

```bibtex
@article{kakavand2026fairness,
  title={Fairness-Aware Deepfake Detection: Eliminating Demographic Bias Through Variance-Regularized Training and Explainable AI},
  author={Kakavand, Kasra},
  journal={Zenodo},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Kasra Kakavand
- GitHub: [@kasra-kakavand](https://github.com/kasra-kakavand)

## Acknowledgments

- PyTorch team for the deep learning framework
- TIMM library for pre-trained image models
- Captum library for explainability tools

## Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

---

If you find this work helpful, please consider giving it a star!
