## 🧬 Breast Cancer Classification using K-Nearest Neighbors (KNN)

This project uses the **Breast Cancer Wisconsin Diagnostic Dataset** from scikit-learn to build a **K-Nearest Neighbors (KNN)** classifier for detecting whether a tumor is **malignant** or **benign**.

### 🔍 Project Overview

* 📦 Dataset: 569 biopsy samples, 30 features per sample
* 🎯 Goal: Classify tumors as malignant (0) or benign (1)
* 🧠 Method: Train KNN models with varying `k` values (1–100) to evaluate accuracy
* 📈 Output: Plot showing validation accuracy for each `k`

### 📌 Key Features

* Data loading and preprocessing using `scikit-learn`
* Automatic model evaluation loop for different `k` values
* Accuracy visualization using `matplotlib`
* Helps determine the optimal `k` for best model performance

### 📊 Results

The project plots **validation accuracy vs. k** to identify the best-performing number of neighbors for the KNN classifier.
