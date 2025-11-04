# Steel Plate Fault Classification

This project implements and compares multiple machine learning classifiers for automated detection of faults in industrial steel plates using the UCI Steel Plates Faults dataset. The study evaluates K-Nearest Neighbors (KNN), Weighted KNN, and Multi-Layer Perceptron (MLP) classifiers on a dataset of nearly 2,000 instances with 27 features across 7 fault categories. Feature engineering techniques including Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) were applied to reduce dimensionality from 27 to 6 features while maintaining discriminative power. The models achieved strong performance with the MLP reaching 74% sensitivity and 96% specificity on the baseline dataset. Comprehensive analysis addressed challenges including class imbalance, non-normal feature distributions, and feature correlations. The implementation includes data visualization, correlation analysis, k-means clustering evaluation, and sensitivity analysis across all features. All code is implemented in Python using scikit-learn, pandas, and matplotlib for a complete end-to-end classification pipeline.

## Key Results
- **Best Overall Performance:** MLP with 74.27% sensitivity, 96.49% specificity
- **Feature Engineering:** LDA reduction achieved silhouette score of 0.135 (27→6 features)
- **Robust Classification:** KNN variants maintained ~70% accuracy post-dimensionality reduction

## Technologies
Python • Scikit-learn • Pandas • NumPy • Matplotlib • Seaborn • Jupyter Notebooks
