# Medical Disease Classification Project

## 📊 Project Overview
This project implements various machine learning models to predict medical conditions, with a focus on achieving high recall for disease detection. The project compares different classification algorithms to find the optimal approach for medical diagnosis.

## 🎯 Objective
To develop a reliable classification system that:
- Maximizes the detection of positive cases (high recall)
- Maintains acceptable precision
- Provides interpretable results for medical staff

## 📈 Models Evaluated
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- AdaBoost
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Networks (MLPClassifier)
- Logistic Regression
- Stacking Ensemble

## 💡 Key Findings

### Model Performance Comparison (Test Set)

| Model               | Recall (Class 1) | Precision (Class 1) | Overfitting | Stability |
|---------------------|------------------|---------------------|-------------|-----------|
| Naive Bayes         | 0.91 ⭐           | 0.64                | Low         | Good      |
| KNN                 | 0.87             | 0.70                | Low         | Good      |
| SVM                 | 0.83             | 0.70                | Low         | Good      |
| Adaboost            | 0.85             | 0.70                | Slight      | Fair      |
| Logistic Regression | 0.83             | 0.73                | Very Low    | Excellent |
| Neural Networks     | 0.79             | 0.71                | Medium      | Fair      |
| Random Forest       | 0.77             | 0.73                | High        | Poor      |
| Gradient Boosting   | 0.79             | 0.71                | Very High   | Poor      |
| XGBoost             | 0.78             | 0.70                | Extremely High | Poor  |
| Stacking            | 0.78             | 0.72                | Medium      | Fair      |

---