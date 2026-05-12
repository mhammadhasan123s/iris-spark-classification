# Iris Classification with Apache Spark

##  Overview
This project applies **Apache Spark MLlib** to classify species in the Iris dataset.  
Three models were implemented — **Decision Tree**, **Random Forest**, and **Logistic Regression** — with both baseline and tuned versions.  
The workflow was executed in **Jupyter Notebook** and reproduced in **PuTTY/HDFS** to demonstrate consistency across environments.

---

##  Dataset and Methodology
- **Dataset**: Iris dataset (`iris.csv`) stored in HDFS (`hdfs:///user/maria_dev/IRIS/iris.csv`)  
- **Features**: sepal length, sepal width, petal length, petal width  
- **Target**: species (setosa, versicolor, virginica)  

### Preprocessing
- `StringIndexer` → convert species names into numeric labels  
- `VectorAssembler` → combine features into a single vector column  

### Workflow
1. Train/test split (80/20, fixed seed)  
2. Train baseline models (Decision Tree, Random Forest, Logistic Regression)  
3. Apply hyperparameter tuning with `CrossValidator` + `ParamGridBuilder` (5-fold CV)  
4. Evaluate models using Accuracy and F1-score  
5. Compare baseline vs tuned results  

---

##  Results and Key Findings
| **Model**             | **Baseline Accuracy** | **Baseline F1-score** | **Tuned Accuracy** | **Tuned F1-score** |
|------------------------|-----------------------|-----------------------|--------------------|--------------------|
| Decision Tree          | 0.92                  | 0.92                  | 0.88               | 0.88               |
| Random Forest          | 0.96                  | 0.96                  | 0.96               | 0.96               |
| Logistic Regression    | 1.00                  | 1.00                  | 0.92               | 0.92               |

- **Random Forest** consistently achieved ~0.96 accuracy and F1-score, making it the most robust model.  
- Logistic Regression showed perfect baseline scores but dropped after tuning due to regularization.  
- Decision Tree was interpretable but less stable compared to Random Forest.

---

##  Running in PuTTY / HDFS
To reproduce results in the Hadoop sandbox:

1. Upload the script `iris_model.py` to `/home/maria_dev/IRIS/`.  
2. Verify the file:
   ```bash
   ls /home/maria_dev/IRIS/
