import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('fraud_detection_model.h5')

# Load dataset
data = pd.read_csv('creditcard.csv')  # Replace with your actual dataset path
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target labels

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the training set
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the distribution of the resampled dataset
st.write("Training set label distribution after SMOTE:")
st.write(pd.Series(y_train_resampled).value_counts())

# Function to create adversarial examples
def generate_adversarial_examples(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)  # Generate Gaussian noise
    X_adv = X + noise  # Add noise to create adversarial examples
    X_adv = np.clip(X_adv, 0, None)  # Ensure no negative values
    return X_adv

# Generate adversarial examples
X_adv = generate_adversarial_examples(X_test)
y_adv = y_test  # Assuming labels remain the same for this example

# Function to calculate model performance
def get_model_performance(model, X, y):
    y_pred = (model.predict(X) > 0.5).astype("int32")  # Assuming binary classification with sigmoid
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return acc, precision, recall, f1, y_pred

# Calculate initial performance metrics on clean data
clean_acc, clean_precision, clean_recall, clean_f1, y_pred = get_model_performance(model, X_test, y_test)

# Display classification report for better insights
st.subheader("Classification Report on Test Data")
st.text(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

# Adversarial Attacks Section
st.header("Adversarial Attacks")
    
# Before vs. After Attack Comparison
st.subheader("Before vs. After Attack")
st.write("Model accuracy before attack: ", clean_acc)

# Get performance on adversarial data
adv_acc, adv_precision, adv_recall, adv_f1, y_adv_pred = get_model_performance(model, X_adv, y_adv)
st.write("Model accuracy after attack: ", adv_acc)

# Generate adversarial example
st.subheader("Adversarial Example")
idx = st.slider("Select Transaction Index", 0, len(X_adv)-1)
st.write(f"Original Transaction: {X_test[idx]}")
st.write(f"Adversarial Transaction: {X_adv[idx]}")

# Reshape input for prediction
original_input = X_test[idx:idx+1]  # Ensure correct shape for model input
adversarial_input = X_adv[idx:idx+1]  # Ensure correct shape for model input

# Get predictions
original_pred = (model.predict(original_input) > 0.5).astype(int)[0][0]  # Reshape input
adv_pred = (model.predict(adversarial_input) > 0.5).astype(int)[0][0]  # Reshape input

# Indicate if the original prediction is fraud
if original_pred == 1:
    st.success("The original transaction is classified as Fraud.")
else:
    st.warning("The original transaction is classified as Not Fraud.")

# Indicate if the adversarial prediction is fraud
if adv_pred == 1:
    st.error("The adversarial transaction is classified as Fraud.")
else:
    st.info("The adversarial transaction is classified as Not Fraud.")
