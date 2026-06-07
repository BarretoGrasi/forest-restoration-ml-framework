# Forest Restoration ML Framework

This repository contains the dataset and source code associated with the manuscript:

**An ML-Based Approach for Ecological Restoration: Species Selection and Successional Planting Design**

## Contents

- `data/`: curated dataset used in the study.  
- `src/`: Python source code for preprocessing, ecological scoring, Random Forest surrogate modeling, hybrid ranking, and planting-plan generation.  
- `outputs/`: generated output tables after running the workflow.  
- `requirements.txt`: Python package dependencies.

## Reproducibility

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the curated dataset file in the `data/` folder, for example:

```text
data/dataset_floresta2.csv
```

3. Run the full workflow:

```bash
python src/main.py --data data/dataset_floresta2.csv --output-dir outputs
```

The workflow generates:

- `outputs/model_metrics.csv`
- `outputs/top_10_species.csv`
- `outputs/planting_plan.csv`

## Notes

The Random Forest model is used as a surrogate model (knowledge emulator) trained to reproduce expert-derived ecological compatibility scores. Therefore, the reported predictive performance reflects internal consistency and heuristic reproducibility rather than independent ecological validation or field performance.

## Data availability

The curated dataset and source code are made available to support transparency and reproducibility of the analyses reported in the manuscript.
