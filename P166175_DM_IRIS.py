#!/usr/bin/env python
# coding: utf-8

# # Comparative Analysis of Classification Models in Spark MLlib
# 
# ### Introduction
# This report presents a comparative study of three classification models — Decision Tree, Random Forest, and Logistic Regression — applied to the Iris dataset using Spark MLlib. 
# First, I evaluate baseline and tuned versions of each model, compare their performance using Accuracy and F1-score, and identify the best-performing model. 

# In[1]:


import os, sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


# In[2]:


#installs PySpark (the Python interface to Apache Spark).
get_ipython().system('pip install pyspark')
#After installation finishes, restart your Jupyter kernel (important so Python can see the new package).


# In[4]:


#Check if Java is installed
get_ipython().system('java -version')


# ### if 'java' is not recognized as an internal or external command, operable program or batch file. 
# 
# you need install the java following the steps A,B,C,D and E

# ##### Step A: Install Java JDK 
# Go to Adoptium OpenJDK (recommended) or Oracle JDK.
# https://adoptium.net/?utm_source=copilot.com
# 
# Download Java 11 (LTS) for Windows (x64 installer).
# Run the installer and complete installation. By default, it installs to something like:
# C:\Program Files\Java\jdk-11.0.x

# ##### Step B: Set JAVA_HOME
# Open Windows Start → type “Environment Variables” → Edit the system environment variables.
# 
# In System Properties → Advanced → Environment Variables:
# 
# Click New under System variables.
# 
# Name: JAVA_HOME
# 
# Value: C:\Program Files\Eclipse Adoptium\jdk-25.0.3.9-hotspot
#  (your actual install path).

# In[128]:


from IPython.display import Image
Image('java22.png', width =400)


# ##### Step C: Add Java to PATH
# 1- In the System variables section, scroll down and select the variable named Path.
# 
# 2- Click Edit….
# 
# 3- In the Path editor, click New.
# 
# 4- Type: 
# #### %JAVA_HOME%\bin
# This tells Windows to look inside the bin folder of your JDK installation whenever a program calls java.
# 
# 5- Click OK to save, then OK again to close all windows.

# ##### Step D: Verify in CMD (not Jupyter yet)
# Sometimes Jupyter doesn’t pick up new environment variables until you restart the whole system. To confirm Java is visible to Windows:
# 
# Open Command Prompt (CMD)Type:
# Code:
# #### java -version
# You should see:   openjdk version "25.0.3" ...
# If CMD shows the version correctly, then Java is installed and PATH is working.

# ##### Step E: Restart Jupyter
# Close Jupyter Notebook completely.
# 
# Restart Anaconda Navigator or Jupyter Lab/Notebook.
# 
# Now run in a Jupyter:

# In[129]:


get_ipython().system('java -version')


# After installing Temurin JDK 25.0.3.9-hotspot, the JAVA_HOME environment variable was set and %JAVA_HOME%\bin was added to the system PATH. Verification was done using java -version in CMD and Jupyter. Restarting Jupyter ensured the updated environment variables were recognized, allowing SparkSession to initialize successfully.

# # Apply Classification Models in Spark MLlib

# In[130]:


from IPython.display import Image
Image('ml-workflow-udacity.png', width =600)


# ## Step 0: Initialize Spark in Jupyter Notebook

# ### Environment Setup
# 
# - **Import SparkSession**: This is the entry point to Spark SQL and DataFrame APIs in PySpark.  
# - **Create or get SparkSession**: The `builder.appName("IrisClassification").getOrCreate()` command starts a new Spark application named *IrisClassification*. If a SparkSession already exists, it reuses it.  
# - **Confirm Spark is running**: Printing `spark.version` verifies that Spark has been initialized correctly and is ready for use.
# 

# In[5]:


# Import SparkSession
from pyspark.sql import SparkSession

# Create or get a SparkSession
spark = SparkSession.builder \
    .appName("IrisClassification") \
    .getOrCreate()

# Confirm Spark is running
print("Spark version:", spark.version)


# #### Now we begin working with Apache Spark in Jupyter Notebook.  
# 
# This step ensures that the Spark engine is connected to Python through PySpark, allowing us to load data, perform transformations, and apply MLlib models in later stages. Without a SparkSession, none of the subsequent machine learning tasks can be executed.
# 

# ## Step 1: Load Iris dataset
# This is where we prepare the Iris dataset so Spark MLlib can use it for classification.

# In[6]:


#1.  Download the dataset
# Since Spark cannot read directly from a web URL, first download the Iris dataset into your working directory:
import urllib.request

url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
urllib.request.urlretrieve(url, "iris.csv")


# This saves the file as iris.csv in your Jupyter folder.

# In[7]:


##Load the CSV into Spark
##Restart SparkSession before reading the CSV
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder \
    .appName("IrisLoader") \
    .master("local[1]") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .config("spark.sql.codegen.wholeStage", "false") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

csv_path = os.path.abspath("iris.csv")
df = spark.read.csv(csv_path, header=True, inferSchema=True)
df.show(5)



# ## Step 2: Preprocessing the data
# 

# In[8]:


# 1. Convert species names into numeric labels (setosa=0, versicolor=1, virginica=2)
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="species", outputCol="label")
df = indexer.fit(df).transform(df)
df.select("species", "label").show(5)


# StringIndexer takes the text column species and converts it into numbers.
# 
# Example: "setosa" → 0, "versicolor" → 1, "virginica" → 2.
# 
# This is necessary because ML models can only work with numeric labels.

# In[9]:


# 2. Combine the four numeric columns into a single 'features' vector
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="features"
)
df = assembler.transform(df)
df.select("features", "label").show(5)


# VectorAssembler merges the four measurement columns into one vector column called features.
# 
# Example: [5.1, 3.5, 1.4, 0.2] becomes a single row’s input vector.
# 
# MLlib expects this format: one features column and one label column.

# ## Step 3: Split dataset

# In[10]:


train, test = df.randomSplit([0.8, 0.2], seed=1)


# randomSplit divides the dataset into two parts:
# 
# 80% for training (train)
# 
# 20% for testing (test)

# ## Step 4: Model Training

# In[11]:


from IPython.display import Image
Image('MLLIB.png', width =1000)


# ##  Logistic Regression
# Logistic Regression models the probability of a sample belonging to a class using the **logistic (sigmoid) function**:
# 
# 
# 
# \[
# P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}}
# \]
# 
# 
# 
# - The model finds a linear boundary in the feature space.  
# - For multiclass problems (like Iris), Spark uses a **one‑vs‑rest strategy**: it fits multiple logistic models, one per species.  
# - The class with the highest probability is chosen.  
# 
#  *Logistic Regression is simple, interpretable, and highly effective when classes are linearly separable.*
# 

# In[12]:


from IPython.display import Image
Image('logistic.jpg', width =1000)


# In[13]:


#1. Logistic Regression
# the Spark MLlib LogisticRegression workflow looks different from the scikit‑learn style, but the logic is the same.

#brings in Spark’s distributed Logistic Regression class.
from pyspark.ml.classification import LogisticRegression

# Train Logistic Regression
#tell Spark which column contains the predictors (features) and which contains the target (label).
#maxIter=10 sets the maximum number of optimization iterations.
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
#trains the Logistic Regression model using the training DataFrame.
lr_model = lr.fit(train)

# Predict on test set
#applies the trained model to the test DataFrame.
lr_predictions = lr_model.transform(test)
lr_predictions.select("features", "label", "prediction").show(5)


# In these first 5 rows, the model’s predictions match the true labels exactly.

# In[14]:


# Evaluate Logistic Regression model 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#MulticlassClassificationEvaluator is Spark’s built‑in tool to measure classification performance.


# Accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(lr_predictions)
print("Logistic Regression Accuracy:", accuracy)
#proportion of correct predictions across the test set.


# F1-score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
f1_score = f1_evaluator.evaluate(lr_predictions)
print("Logistic Regression F1-score:", f1_score)
#harmonic mean of precision and recall, showing balance across classes.


# Accuracy 1.0 → The Logistic Regression model classified every single flower in the test set correctly.
# 
# F1‑score 1.0 → Precision and recall are both perfect, meaning the model didn’t make any false positives or false negatives.
# 
# This can happen with the Iris dataset because it’s small, clean, and the classes are well‑separated (especially setosa vs the others). 

# ##  Decision Tree
# A Decision Tree splits the dataset into branches based on feature thresholds. At each node, it chooses the split that maximizes **information gain** (or minimizes impurity).
# 
# One common impurity measure is the **Gini Index**:
# 
# 
# 
# \[
# Gini(D) = 1 - \sum_{i=1}^{k} p_i^2
# \]
# 
# 
# 
# where \(p_i\) is the proportion of class \(i\) in dataset \(D\).
# 
# - Example: If petal length < 2.5 → classify as setosa; otherwise, continue splitting.  
# - The process continues until leaves represent final class predictions.  
# 
#  *Decision Trees are easy to visualize and interpret, but a single tree can overfit and misclassify overlapping species.*
# 

# In[15]:


from IPython.display import Image
Image('Decision-Tree.png', width =800)


# In[16]:


#2. Decision Tree
from pyspark.ml.classification import DecisionTreeClassifier

# Train Decision Tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_model = dt.fit(train)

# Predict on test set
dt_predictions = dt_model.transform(test)
dt_predictions.select("features", "label", "prediction").show(5)


# Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
dt_accuracy = evaluator.evaluate(dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)



# Evaluate F1-score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
dt_f1 = f1_evaluator.evaluate(dt_predictions)
print("Decision Tree F1-score:", dt_f1)


# The Decision Tree achieved 92% accuracy and an F1‑score of 0.92. Most flowers were classified correctly, but some virginica samples were misclassified as versicolor due to overlapping feature values. This shows that while Decision Trees are interpretable, they can struggle with species that have similar measurements.

# ##  Random Forest
# Random Forest builds multiple decision trees (here `numTrees=10`).  
# 
# - Each tree votes on the class, and the forest takes the majority vote.  
# - This reduces overfitting compared to a single Decision Tree and usually improves accuracy.  
# - By aggregating many trees, Random Forest achieves stronger generalization while remaining interpretable.  
# 
#  *In the Iris dataset, Random Forest improves accuracy compared to a single Decision Tree, especially for overlapping species like versicolor and virginica.*

# In[17]:


from IPython.display import Image
Image('Random_forest.png', width =600)


# In[18]:


from pyspark.ml.classification import RandomForestClassifier

# Train Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
rf_model = rf.fit(train)

# Predict on test set
rf_predictions = rf_model.transform(test)
rf_predictions.select("features", "label", "prediction").show(5)



# Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
rf_accuracy = evaluator.evaluate(rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)



# Evaluate F1-score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
rf_f1 = f1_evaluator.evaluate(rf_predictions)
print("Random Forest F1-score:", rf_f1)


# Accuracy = 0.9615 (~96%) → The Random Forest correctly classified about 96% of the test flowers.
# 
# F1‑score = 0.9617 (~96%) → Precision and recall are balanced, meaning the model performs consistently across all three species.
# 
# Random Forest improved over the single Decision Tree (92%) by combining multiple trees and reducing overfitting.
# 
# However, it still made a few mistakes — especially confusing virginica vs versicolor, since their petal measurements overlap.

# ## Step 5: Apply model tuning techniques
# #### Systematically test different hyperparameter values using cross-validation, then select the best model.
# 
# what you did before in step 4 was training and evaluating with default parameters. That gave you a baseline accuracy and F1‑score for Decision Tree, Random Forest, and Logistic Regression.
# 
# Step 5 is different: instead of just fitting once, we now systematically test multiple parameter values using ParamGridBuilder + CrossValidator. This way Spark chooses the best hyperparameters automatically.
# 
# These defaults may not give the best performance.

# ## ParamGridBuilder
# 
# Defines the set of hyperparameters to test.
# 
# ---
# ### Explanation of Each Argument in ParamGridBuilder
# 
# 
# #### $$regParam$$ 
# ##### Controls the regularization strength (penalty on large coefficients).
# 
# Ex:Small values (0.01) → weak regularization → model fits more closely to training data.
# Large values (1.0) → strong regularization → coefficients shrink more, reducing overfitting.
# 
# 
# 
# #### $$elasticNetParam$$
# ##### Controls the type of regularization:
# 
# 0.0 → pure L2 (Ridge).
# 
# 1.0 → pure L1 (Lasso).
# 
# 0.5 → mix of L1 and L2 (Elastic Net)
# 
# 
# 
# 
# #### $$maxIter$$
# ##### Maximum number of iterations for the optimization algorithm.
# 
# 
# 
# 
# 
# #### $$maxDepth$$ 
# ##### Controls the maximum depth of the tree (how many splits from root to leaf).
# 
# Small values (2, 4) → shallow tree, less complex, lower risk of overfitting but may underfit.
# 
# Large values (8, 10) → deeper tree, more complex, can capture more detail but risk overfitting.
# 
# 
# 
# 
# #### $$maxBins$$
# 
# ##### Controls the number of bins used to discretize continuous features before splitting.
# 
# Higher values allow finer splits, but increase computation.
# 
# Lower values make the tree simpler, but may miss subtle patterns.
# 

# In[19]:


from IPython.display import Image
Image('GridBuilder.png', width =600)


# 
# ## CrossValidator
# 
# Cross-validation makes the models more generalizable across different data splits.
# 
# Even if tuned metrics are lower, the models are less likely to overfit.
# 
# 
# ---
# ### Explanation of Each Argument in CrossValidator
# ##### estimatorParamMaps=paramGrid
# The set of hyperparameter combinations to test.
# 
# ##### evaluator=evaluator
# 
# Defines how performance is measured.
# 
# Accuracy is used to compare models. You could also use F1-score, precision, or recall.

# In[20]:


from IPython.display import Image
Image('fold-cross-validation.WEBP', width =800)


# #### Tuning Logistic Regression model with CrossValidator + ParamGridBuilder

# In[23]:


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Build parameter grid (hyperparameters to test)
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.1, 0.5, 1.0])   # regularization strength
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])  # L2 (0.0), mix (0.5), L1 (1.0)
             .addGrid(lr.maxIter, [50, 100])                # number of iterations
             .build())

# Define evaluator (accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")

# CrossValidator setup
cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)   # 5-fold cross-validation

# Fit model with CV
cvModel = cv.fit(train)

# Best tuned model
best_lr_model = cvModel.bestModel


# Predict on test set
lr_predictions = best_lr_model.transform(test)
lr_predictions.select("features", "label", "prediction").show(5)

# Evaluate tuned model
lr_accuracy = evaluator.evaluate(lr_predictions)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
lr_f1 = f1_evaluator.evaluate(lr_predictions)

print("Tuned Logistic Regression Accuracy:", lr_accuracy)
print("Tuned Logistic Regression F1-score:", lr_f1)


# Baseline LR → Perfect scores (1.0 / 1.0).
# 
# Tuned LR → Slightly lower (0.92 / 0.92).
# 
# Baseline may have overfit or benefited from a lucky train/test split.
# 
# Tuning introduced regularization, making the model more robust across folds, but less perfect on this test set.
# 
# #### Baseline Logistic Regression looked “perfect,” but that’s not always reliable.
# 
# #### Tuned Logistic Regression is more realistic and generalizable, even though the metrics are lower.

# #### Tuning Decision Tree with CrossValidator + ParamGridBuilder

# In[22]:


# Tuning Decision Tree with CrossValidator + ParamGridBuilder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define Decision Tree model
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")

# Build parameter grid (hyperparameters to test)
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [2, 4, 6, 8, 10])   # tree depth
             .addGrid(dt.maxBins, [16, 32, 64])        # bins for feature splits
             .build())

# Define evaluator (accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")

# CrossValidator setup
cv = CrossValidator(estimator=dt,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)   # 5-fold cross-validation

# Fit model with CV
cvModel = cv.fit(train)

# Best tuned model
best_dt_model = cvModel.bestModel

# Predict on test set
dt_predictions = best_dt_model.transform(test)
dt_predictions.select("features", "label", "prediction").show(5)

# Evaluate tuned model
dt_accuracy = evaluator.evaluate(dt_predictions)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
dt_f1 = f1_evaluator.evaluate(dt_predictions)

print("Tuned Decision Tree Accuracy:", dt_accuracy)
print("Tuned Decision Tree F1-score:", dt_f1)


# #### Comparative Interpretation (Decision Tree)
# 
# Baseline DT: Higher accuracy and F1 (~0.92).
# 
# Tuned DT: Lower accuracy and F1 (~0.88).
# 
# Reason: Cross-validation optimizes for average performance across folds, not necessarily the single test split.
# 
# The tuned model may be more robust overall, but on this specific test set, it underperformed.
# 
# ##### Strength: Tuned model is less likely to overfit.
# 
# Limitation: May sacrifice performance on one test split compared to default parameters.

# #### Tuning Random Forest 

# In[24]:


from pyspark.ml.classification import RandomForestClassifier

# Define Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# Build parameter grid (hyperparameters to test)
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [10, 20, 50, 100])   # number of trees
             .addGrid(rf.maxDepth, [2, 4, 6, 8, 10])    # tree depth
             .build())

# Define evaluator (accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")

# CrossValidator setup
cv = CrossValidator(estimator=rf,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)   # 5-fold cross-validation

# Fit model with CV
cvModel = cv.fit(train)

# Best tuned model
best_rf_model = cvModel.bestModel

# Predict on test set
rf_predictions = best_rf_model.transform(test)
rf_predictions.select("features", "label", "prediction").show(5)

# Evaluate tuned model
rf_accuracy = evaluator.evaluate(rf_predictions)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
rf_f1 = f1_evaluator.evaluate(rf_predictions)

print("Tuned Random Forest Accuracy:", rf_accuracy)
print("Tuned Random Forest F1-score:", rf_f1)


# ## Step 6: Baseline vs Tuned Model Comparison
# 
# | **Model**             | **Baseline Accuracy** | **Baseline F1-score** | **Tuned Accuracy** | **Tuned F1-score** |
# |------------------------|-----------------------|-----------------------|--------------------|--------------------|
# | Decision Tree          | 0.92                  | 0.92                  | 0.88               | 0.88               |
# | Random Forest          | 0.96                  | 0.96                  | 0.96               | 0.96               |
# | Logistic Regression    | 1.00                  | 1.00                  | 0.92               | 0.92               |
# 

# In[25]:


import matplotlib.pyplot as plt
import pandas as pd

# Data: accuracy values for three models
data = {
    'Model': ['Decision Tree', 'Random Forest', 'Logistic Regression'],
    'Baseline Accuracy': [0.92, 0.96, 1.00],
    'Tuned Accuracy': [0.88, 0.96, 0.92]
}
df = pd.DataFrame(data)

# Plot grouped bar chart (Accuracy only)
# kind='bar' → bar chart, figsize → chart size, color → light creative colors
ax = df.plot(x='Model',
             y=['Baseline Accuracy','Tuned Accuracy'],
             kind='bar', figsize=(10,6),
             color=['#B0E0E6','#F5DEB3'])  # light turquoise + wheat

# Add title and axis labels
plt.title('Baseline vs Tuned Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0.8,1.05)  # zoom in to highlight differences
plt.xticks(rotation=0)  # keep model names horizontal
plt.legend(title='Metric', fontsize=10)

# Add values inside each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='center', fontsize=9, color='black')

# Show chart
plt.tight_layout()
plt.show()


# ### Interpretation
# 
# Tuning affected models differently. Decision Tree dropped slightly (0.92 → 0.88), showing a trade‑off between accuracy and generalization. Random Forest stayed strong at ~0.96, confirming its robustness. Logistic Regression fell from perfect (1.00 → 0.92), but tuning added regularization, making results more realistic and less overfit. Overall, Random Forest remained the most stable performer.
# 
# ### Winner Model
# Random Forest is the winner with ~0.96 accuracy and F1‑score, showing the most stable and robust performance after tuning.
# 

# ### “Getting the same results in PuTTY using Apache Spark.

# In[27]:


from IPython.display import Image
Image('puTTY_2.png', width =800)

