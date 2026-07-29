# Iris Classification with Spark MLlib

A comparative study of three classification models — **Decision Tree**, **Random Forest**, and **Logistic Regression** on the Iris dataset using **Apache Spark MLlib**, covering exploratory data analysis, baseline models, hyperparameter tuning with cross-validation, and per-model confusion matrices.

Data Management — Assignment 1.

## Overview

The workflow in `P166175_DM_IRIS.ipynb`:

1. **Environment setup** install PySpark, verify Java, initialise a single `SparkSession`.
2. **Load data** download `iris.csv` and read it into a Spark DataFrame.
3. **EDA** schema, missing-value check, class balance, and feature-by-species distributions.
4. **Preprocessing** `StringIndexer` for labels, `VectorAssembler` for the feature vector.
5. **Split** 80/20 train/test with a fixed seed.
6. **Baseline models** Decision Tree, Random Forest, Logistic Regression with default parameters, each with a confusion matrix.
7. **Tuning** `CrossValidator` + `ParamGridBuilder` (5-fold) for each model.
8. **Comparison** a results table and bar chart generated directly from the computed values, plus an automatically selected best model.

## Reproducibility

Every source of randomness is seeded, so *Restart & Run All* reproduces the same numbers on the same Spark version:

- train/test split: `randomSplit(..., seed=1)`
- `DecisionTreeClassifier(seed=42)`, `RandomForestClassifier(seed=42)`
- `CrossValidator(seed=42)`

The results table and comparison chart are built from the live model outputs, so the reported numbers always match what the notebook prints.

## Results

These values are produced directly by the notebook and have been verified to reproduce identically on **Spark 3.5.1** and **Spark 4.2.0**, on two separate machines.

| Model | Baseline Accuracy | Baseline F1 | Tuned Accuracy | Tuned F1 |
|---|---|---|---|---|
| Decision Tree | 0.9231 | 0.9231 | 0.8846 | 0.8833 |
| Random Forest | 0.9231 | 0.9231 | 0.8846 | 0.8833 |
| Logistic Regression | 1.0000 | 1.0000 | 0.9231 | 0.9231 |

**Best tuned model:** Logistic Regression (accuracy 0.9231, F1 0.9231).

**Reading the results**

- *Setosa* is classified perfectly by every model it is linearly separable, as the EDA shows.
- All errors are *versicolor* ↔ *virginica* confusions, driven by overlapping petal measurements. This is visible in both the EDA distributions and the confusion matrices.
- Tuning lowers the score on this particular test split for all three models. Cross-validation optimises for average performance across folds rather than for one split, so the tuned models are the more robust choice even though their single-split numbers are lower.
- **Caveat on the margins:** the test set contains only 26 samples, so 0.9231 = 24/26 correct and 0.8846 = 23/26. The gap between the best and worst tuned model is a single sample. These differences are within noise and should not be read as a strong ranking of the algorithms; the confusion matrices are more informative than the headline accuracy.

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

- `P166175_DM_IRIS.ipynb` the analysis notebook
- `iris.csv` dataset (downloaded automatically by the notebook)
- a few conceptual diagrams referenced inline (ML workflow, decision tree, random forest, cross-validation)
