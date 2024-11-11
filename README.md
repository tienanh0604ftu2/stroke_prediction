# Medical Disease Classification Project

## 📊 Project Overview

This project implements various machine learning models to predict medical conditions, focusing on achieving high recall for disease detection. The project compares different classification algorithms to find the optimal approach for medical diagnosis.

## 🎯 Objective

To develop a reliable classification system that:

- Maximizes the detection of positive cases (high recall).
- Maintains acceptable precision.
- Provides interpretable results for medical staff.

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

| Model               | Recall (Class 1) | Precision (Class 1) | Overfitting       | Stability  |
|---------------------|------------------|---------------------|-------------------|------------|
| Naive Bayes         | 0.91 ⭐         | 0.64                | Low               | Good       |
| KNN                 | 0.87             | 0.70                | Low               | Good       |
| SVM                 | 0.83             | 0.70                | Low               | Good       |
| AdaBoost            | 0.85             | 0.70                | Slight            | Fair       |
| Logistic Regression | 0.83             | 0.73                | Very Low          | Excellent  |
| Neural Networks     | 0.79             | 0.71                | Medium            | Fair       |
| Random Forest       | 0.77             | 0.73                | High              | Poor       |
| Gradient Boosting   | 0.79             | 0.71                | Very High         | Poor       |
| XGBoost             | 0.78             | 0.70                | Extremely High    | Poor       |
| Stacking Ensemble   | 0.78             | 0.72                | Medium            | Fair       |

## 🚀 How to Use the Application

1. **Clone the repository**:
   ```sh
   git clone https://github.com/tienanh0604ftu2/medical-disease-classification.git
   ```

2. **Navigate to the project directory**:
   ```sh
   cd medical-disease-classification
   ```

3. **Install dependencies**:
   Make sure you have Python 3.7+ installed. Run the following command to install required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**:
   ```sh
   streamlit run app.py
   ```

5. **Access the app**:
   The app will be running on `http://localhost:8501`. You can use the user interface to input patient information and predict medical conditions.

## 🛠 Repository Structure

- **app.py**: The main application script for the Streamlit app.
- **models/**: Pre-trained models used for prediction.
- **data/**: Dataset used for training and testing models.
- **requirements.txt**: Contains the list of dependencies for the project.
- **README.md**: Overview and instructions for the project.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

