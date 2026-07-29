# Iris Classification with Spark MLlib

A comparative study of three classification models — **Decision Tree**, **Random Forest**, and **Logistic Regression** — on the Iris dataset using **Apache Spark MLlib**, covering exploratory data analysis, baseline models, hyperparameter tuning with cross-validation, and per-model confusion matrices.

Data Management — Assignment 1.

## Overview

The workflow in `P166175_DM_IRIS.ipynb`:

1. **Environment setup** — install PySpark, verify Java, initialise a single `SparkSession`.
2. **Load data** — download `iris.csv` and read it into a Spark DataFrame.
3. **EDA** — schema, missing-value check, class balance, and feature-by-species distributions.
4. **Preprocessing** — `StringIndexer` for labels, `VectorAssembler` for the feature vector.
5. **Split** — 80/20 train/test with a fixed seed.
6. **Baseline models** — Decision Tree, Random Forest, Logistic Regression with default parameters, each with a confusion matrix.
7. **Tuning** — `CrossValidator` + `ParamGridBuilder` (5-fold) for each model.
8. **Comparison** — a results table and bar chart generated directly from the computed values, plus an automatically selected best model.

## Reproducibility

Every source of randomness is seeded, so *Restart & Run All* reproduces the same numbers on the same Spark version:

- train/test split: `randomSplit(..., seed=1)`
- `DecisionTreeClassifier(seed=42)`, `RandomForestClassifier(seed=42)`
- `CrossValidator(seed=42)`

The results table and comparison chart are built from the live model outputs, so the reported numbers always match what the notebook prints.

## Results

> Generate this table from your own clean run (Restart & Run All) — the values below are from a reference run on Spark 4.2.0 and will be reproducible on that version; Spark 3.5.1 may differ slightly, but your notebook's table will always match your own output.

| Model | Baseline Accuracy | Baseline F1 | Tuned Accuracy | Tuned F1 |
|---|---|---|---|---|
| Logistic Regression | 1.00 | 1.00 | 0.92 | 0.92 |
| Decision Tree | 0.92 | 0.92 | 0.88 | 0.88 |
| Random Forest | 0.92 | 0.92 | 0.88 | 0.88 |

**Reading the results:** *setosa* is classified perfectly by every model. All errors are *versicolor* ↔ *virginica* confusions, driven by overlapping petal measurements (visible in both the EDA plots and the confusion matrices). Tuning trades a little peak accuracy on this split for better cross-validated robustness.

## Requirements

- Python 3.x
- PySpark 3.5+
- Java 11+ (JDK)
- pandas, matplotlib, numpy

## Running

```bash
pip install pyspark pandas matplotlib numpy
jupyter notebook P166175_DM_IRIS.ipynb
# then: Kernel -> Restart & Run All
```

## Repository contents

- `P166175_DM_IRIS.ipynb` — the analysis notebook
- `iris.csv` — dataset (downloaded automatically by the notebook)
- a few conceptual diagrams referenced inline (ML workflow, decision tree, random forest, cross-validation)
