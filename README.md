# FSM_Assignments

# Machine Learning Algorithms Implementation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-green.svg)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-yellow.svg)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-red.svg)](https://matplotlib.org)

Core machine learning algorithms implemented using fundamental Python libraries (NumPy, Pandas, Matplotlib) without high-level ML frameworks.

📋 Overview

Three comprehensive notebooks covering essential ML techniques:

| Notebook | Algorithms | Dataset | Key Features |
|----------|------------|---------|-------------|
| **Unsupervised Learning** | K-Means, PCA | Iris (150 samples) | Clustering, dimensionality reduction |
| **Supervised Classification** | Naive Bayes, KNN | Titanic (891 samples) | Binary classification, preprocessing |
| **Linear Regression** | Normal Equation | Insurance (1338 samples) | Regression, feature scaling |

🚀 Quick Start

```bash
# Setup
git clone <repository-url>
cd "new tp"
pip install numpy pandas matplotlib seaborn jupyter

# Run
jupyter notebook
```

📊 Key Results

- **PCA**: 95% variance explained in first 2 components
- **K-Means**: 90%+ clustering accuracy on species classification
- **Naive Bayes**: ~80% accuracy on survival prediction
- **KNN**: ~82% accuracy (optimal k=5)
- **Linear Regression**: 75% variance explained, ±$6K prediction error

## 🔬 Implementation Highlights

### Unsupervised Learning
- **K-Means**: Euclidean distance, centroid updates, convergence detection
- **PCA**: Covariance matrix, eigenvalue decomposition, data standardization

### Supervised Learning
- **Naive Bayes**: Gaussian probability density, class priors, Bayes theorem
- **KNN**: Distance calculation, majority voting, optimal k selection
- **Linear Regression**: Normal equation solution, feature scaling, R²/RMSE metrics

## 📁 Structure

```
new tp/
├── Unsupervised_Learning.ipynb    # K-Means & PCA
├── Supervised-Classification.ipynb # Naive Bayes & KNN  
├── Linear_Regression.ipynb        # Regression analysis
└── datasets/                      # Data files
```

## 💻 Usage Example

```python
# K-Means Clustering
labels, centroids, _ = kmeans(X, k=3)

# Naive Bayes Classification  
nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

# Linear Regression
theta, metrics = linear_regression(X_train, y_train)
print(f"R² Score: {metrics['r2_score']:.3f}")
```

## 🎯 Learning Outcomes

- Mathematical implementation of core ML algorithms
- Data preprocessing and feature engineering
- Model evaluation and performance metrics
- Effective data visualization techniques

## 📚 Requirements

```
numpy>=1.19.0
pandas>=1.2.0  
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 👤 Author

**Harshil Vadalia** - Algorithm implementation and documentation

---

*Educational project demonstrating fundamental machine learning concepts through clean, documented code.*
