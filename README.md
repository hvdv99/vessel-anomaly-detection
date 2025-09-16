# Contextual Maritime Anomaly Detection
This repository contains the code, data, and experiments for my master's thesis
**"Contextual Maritime Anomaly Detection"** (JADS, MSc in Data Science for Business and Entrepreneurship, 2025).

The research investigates how combining AIS data with meteorological data (ERA5 reanalysis) into deep learning frameworks improves anomaly detection in maritime vessel trajectories, reducing false positives from weather-induced behavior.

---

## Repository Structure
```
.
├── data/ # Data to reproduce the experiments
├── docs/ # Thesis report and defense slides
├── experiments/ # Jupyter notebooks for inference and gathering results
├── figures/ # Plots and figures used in the thesisreport
├── notebooks/ # Data processing and model training notebooks
├── src/vad/ # Source code for PyTorch model architectures and data loader
├── environment.yml # Micromamba/conda environment specification
└── install-micromamba.sh # Helper script for environment setup
```

---

## Environment
It is recommended to use **micromamba** (or conda), to install micromamba and create the environment run:

```bash
chmod +x install-micromamba.sh
./install-micromamba.sh
```

---

## Model Training
To train the models run the Jupyter Notebooks in: `/notebooks/model-training`

---