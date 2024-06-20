# 📱 Human Activity Classification through Smartphones

## 🚀 Introduction
Welcome to our project on **Human Activity Classification using Smartphones**! We analyze time series data collected from smartphones to recognize different human activities. Our focus is on feature extraction from temporal data and implementing machine learning models, specifically Random Forest and SVM with kernel, to classify these activities.

## 👥 Authors
- **David Herencia** - Departamento de Computer Science, Universidad de Ingeniería y Tecnología, Ocaña, Perú
- **Josué Arbulú** - Departamento de Data Science, Universidad de Ingeniería y Tecnología, Lima, Perú
- **Harold Canto** - Departamento de Computer Science, Universidad de Ingeniería y Tecnología, Lima, Perú
- **Neftalí Anderson Calixto Rojas** - Departamento de Computer Science, Universidad de Ingeniería y Tecnología, Lima, Perú

## 📄 Abstract
We present an analysis of time series data for human activity recognition. The project focuses on feature extraction from temporal data and implementing Random Forest and SVM with kernel for classification. We experiment with different hyperparameters to evaluate and compare the performance of these models.

## 🔑 Keywords
- **Time series**
- **Feature extraction**
- **Radial basis function kernel**
- **Random Forest**
- **Decision tree**
- **SVM**
- **OVO-OVR**
- **Dimensionality reduction**

## 🗂️ Dataset Overview
The dataset was measured using accelerometers and gyroscopes from a group of 30 volunteers aged 19 to 48, performing six activities (walking, walking upstairs, walking downstairs, sitting, standing, lying) using a smartphone (Samsung Galaxy S II) worn on the waist. Data was captured at 50Hz.

## 🛠️ Feature Extraction
We used the `tsfresh` library to extract features from the time series data, generating three different feature sets. Due to RAM limitations in Colab, we developed a function to process the data in smaller batches.

### ⏱ Extraction Times
| Method     | Number of Features | Extraction Time |
|------------|---------------------|-----------------|
| Minimal    | 90                  | 3 min           |
| Default    | 7049                | 7 hours         |
| Efficient  | 6994                | 2 hours         |

## 📊 Exploratory Data Analysis (EDA) and Feature Selection
For EDA, we used the `features_efficient` dataframe with 5727 columns and 7352 rows. We applied three statistical tests for feature selection:
- **Mutual Information**
- **F-score (ANOVA F-test)**
- **P-value**

### 🔍 Top Features
#### F-score
| Feature                                   | F-score  |
|-------------------------------------------|----------|
| total acc x root mean square              | 41075.77 |
| total acc x mean n absolute max number of | 40074.80 |

#### Mutual Information
| Feature                                   | Mutual Information |
|-------------------------------------------|---------------------|
| total acc x maximum                       | 1.322542            |
| total acc x absolute maximum              | 1.322432            |

We selected the top 650 features based on feature importance from a Random Forest Classifier, further reducing to 630 after matching with the test dataset.

## 📈 Visualization
We compared the original and reduced datasets using PCA, T-SNE, and LDA.

### 📉 PCA and T-SNE Visualizations
![PCA Original](path/to/pca_original.png)
![PCA Reduced](path/to/pca_reduced.png)
![T-SNE Original](path/to/tsne_original.png)
![T-SNE Reduced](path/to/tsne_reduced.png)

## 📝 Methodology
### Support Vector Machines (SVM)
Implemented using both linear and RBF kernels. We used OvR (One-vs-Rest) and OvO (One-vs-One) strategies for multi-class classification.

### Random Forest
Combined multiple decision trees to improve accuracy and reduce overfitting. Key concepts include entropy, information gain, and bootstrap sampling.

## 💻 Implementation
Our implementation is available in the [GitHub repository](https://github.com/DavidHerencia/Proyecto_2_ML/).

### Key Classes
- **SVM**
- **OVR_SVM**
- **OVO_SVM**
- **DecisionTree**
- **RandomForest**

## 🔬 Experimentation
### K-Fold Cross Validation
Used k=5 for a good balance between bias and variance.

### Hyperparameter Tuning
Experimented with different hyperparameters for both Random Forest and SVM models to optimize performance based on F1 score, precision, and recall.

### Results
#### Random Forest
| Accuracy | Precision | Recall | F1 Score | Hyperparameters         |
|----------|-----------|--------|----------|-------------------------|
| 0.980142 | 0.980701  | 0.980794 | 0.980733 | [25, 15, 2, sqrt, 4]    |

#### OVO SVM
| Accuracy | Precision | Recall | F1 Score | Hyperparameters                   |
|----------|-----------|--------|----------|-----------------------------------|
| 0.9748   | 0.9762    | 0.9764 | 0.9762   | [0.001, 0.01, 2000, rbf, 0.01]    |

## 📊 Visualization of Results
### Random Forest
![Precision vs Recall RF](path/to/precision_vs_recall_rf.png)
![F1 vs Accuracy RF](path/to/f1_vs_accuracy_rf.png)
![Confusion Matrix RF](path/to/confusion_matrix_rf.png)
![ROC Curve RF](path/to/roc_curve_rf.png)

### OVO SVM
![Precision vs Recall SVM](path/to/precision_vs_recall_svm.png)
![F1 vs Accuracy SVM](path/to/f1_vs_accuracy_svm.png)
![Confusion Matrix SVM](path/to/confusion_matrix_svm.png)
![ROC Curve SVM](path/to/roc_curve_svm.png)

## 📝 Discussion
Both Random Forest and OVO SVM models showed excellent performance with F1 scores above 0.94. The best hyperparameters for Random Forest were 25 trees, max depth of 15, min samples split of 2, max features as sqrt, and 4 jobs. For OVO SVM, the best settings were a learning rate of 0.001, lambda of 0.01, 2000 iterations, RBF kernel, and gamma of 0.01.

## 📜 Conclusions
- **High Performance**: Both models achieved high F1 scores and accurately predicted all classes.
- **Effective Feature Selection**: Using tsfresh and statistical tests to select the most important features improved model performance.
- **Dimensionality Reduction**: PCA and T-SNE provided clear separation of classes, while LDA was more effective with the original dataset.

## 🔗 Links
- [Project Drive](https://drive.google.com/drive/folders/1gzB8doKX3jdKGM9ApGx99cvkGuP7O22h?usp=drive_link)
- [Random Forest Experimentation Notebook](https://github.com/DavidHerencia/Proyecto_2_ML/blob/main/experimentacion_RandomForest.ipynb)
- [SVM Experimentation Notebook](https://github.com/DavidHerencia/Proyecto_2_ML/blob/main/experimentacion_SVM_OVO.ipynb)
- [GitHub Repository](https://github.com/DavidHerencia/Proyecto_2_ML/)
