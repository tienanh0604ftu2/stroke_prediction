import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Cấu hình trang
st.set_page_config(
    page_title="Medical Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

# Tiêu đề
st.title("🏥 Medical Disease Prediction System")
st.write("This application helps predict medical conditions using machine learning.")

# Sidebar cho model selection
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Naive Bayes (High Recall)", "KNN (Balanced)", "Both Models"]
)

# Load models
@st.cache_resource
def load_models():
    nb_model = pickle.load(open('models/naive_bayes.pkl', 'rb'))
    knn_model = pickle.load(open('models/knn.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    feature_columns = pickle.load(open('models/feature_names.pkl', 'rb'))
    return nb_model, knn_model, scaler, feature_columns

try:
    nb_model, knn_model, scaler, feature_columns = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure model files exist.")

# Input form
st.header("Patient Information")

# Input fields
gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, value=25, step=1)
hypertension = st.selectbox("Hypertension", options=[0, 1], index=0)
heart_disease = st.selectbox("Heart Disease", options=[0, 1], index=0)
ever_married = st.selectbox("Ever Married", options=["No", "Yes"])
work_type = st.selectbox("Work Type", options=["Private", "Self-employed", "Govt job", "children", "Never_worked"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=85.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, value=22.0, step=0.1)
smoking_status = st.selectbox("Smoking Status", options=["never smoked", "formerly smoked", "smokes", "Unknown"])

def categorize_glucose_level(value):
    if value < 100:
        return 'Normal'
    elif 100 <= value <= 125:
        return 'Prediabetes'
    elif 126 <= value <= 216:
        return 'High Risk'
    else:
        return 'Very High Risk'

def prepare_features(input_data):
    # Create a dictionary with all input features
    data_dict = {
        'age': [input_data['age']],
        'hypertension': [input_data['hypertension']],
        'heart_disease': [input_data['heart_disease']],
        'avg_glucose_level': [input_data['avg_glucose_level']],
        'bmi': [input_data['bmi']],
        'gender': [input_data['gender']],
        'ever_married': [input_data['ever_married']],
        'work_type': [input_data['work_type']],
        'smoking_status': [input_data['smoking_status']],
        'glucose_category': [categorize_glucose_level(input_data['avg_glucose_level'])]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    
    # Get categorical columns
    cat_columns = ['gender', 'ever_married', 'work_type', 'smoking_status', 'glucose_category']
    
    # Create dummy variables
    df_encoded = pd.get_dummies(df, columns=cat_columns)
    
    # Ensure all feature columns from training are present
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Ensure columns are in the same order as during training
    df_encoded = df_encoded[feature_columns]
    
    return df_encoded

def make_prediction(features, model_type="both"):
    # Scale features
    features_scaled = scaler.transform(features)
    
    if model_type == "nb":
        pred = nb_model.predict(features_scaled)
        prob = nb_model.predict_proba(features_scaled)
        return pred[0], prob[0]
    
    elif model_type == "knn":
        pred = knn_model.predict(features_scaled)
        prob = knn_model.predict_proba(features_scaled)
        return pred[0], prob[0]
    
    else:  # both models
        nb_pred = nb_model.predict(features_scaled)[0]
        nb_prob = nb_model.predict_proba(features_scaled)[0]
        knn_pred = knn_model.predict(features_scaled)[0]
        knn_prob = knn_model.predict_proba(features_scaled)[0]
        return (nb_pred, nb_prob), (knn_pred, knn_prob)

# Prediction button
NO_DISEASE_LABEL = "No Disease"
DISEASE_LABEL = "Disease"

if st.button("Make Prediction"):
    # Prepare input data
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    # Prepare features in correct format
    features_df = prepare_features(input_data)
    
    st.header("Prediction Results")
    
    if selected_model == "Naive Bayes (High Recall)":
        prediction, probabilities = make_prediction(features_df, "nb")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", DISEASE_LABEL if prediction == 1 else NO_DISEASE_LABEL)
        with col2:
            st.metric("Confidence", f"{max(probabilities)*100:.2f}%")
        
        st.write("Probability Distribution:")
        prob_df = pd.DataFrame({
            'Condition': [NO_DISEASE_LABEL, DISEASE_LABEL],
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Condition'))
        
    elif selected_model == "KNN (Balanced)":
        prediction, probabilities = make_prediction(features_df, "knn")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", DISEASE_LABEL if prediction == 1 else NO_DISEASE_LABEL)
        with col2:
            st.metric("Confidence", f"{max(probabilities)*100:.2f}%")
        
        st.write("Probability Distribution:")
        prob_df = pd.DataFrame({
            'Condition': [NO_DISEASE_LABEL, DISEASE_LABEL],
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Condition'))
        
    else:
        (nb_pred, nb_prob), (knn_pred, knn_prob) = make_prediction(features_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Naive Bayes Prediction")
            st.metric("Prediction", DISEASE_LABEL if nb_pred == 1 else NO_DISEASE_LABEL)
            st.metric("Confidence", f"{max(nb_prob)*100:.2f}%")
            
        with col2:
            st.subheader("KNN Prediction")
            st.metric("Prediction", DISEASE_LABEL if knn_pred == 1 else NO_DISEASE_LABEL)
            st.metric("Confidence", f"{max(knn_prob)*100:.2f}%")

# Model information
st.header("Model Information")
st.write("""
This system uses two different models:
- **Naive Bayes**: Optimized for high recall (91% on test set)
- **KNN**: Balanced performance (87% recall, 70% precision)
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by [Ak]")