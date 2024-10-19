import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # Import SMOTE
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('fraud_detection_model.h5')

# Load dataset
data = pd.read_csv('creditcard.csv')  # Replace with your actual dataset path
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target labels

# Separate the fraud and non-fraud transactions
fraud_cases = data[data['Class'] == 1]
non_fraud_cases = data[data['Class'] == 0]

# Limit the number of fraud cases in the test set to 1
test_fraud_case = fraud_cases.sample(n=1, random_state=42)

# Use all but one of the non-fraud cases for testing
non_fraud_cases_test = non_fraud_cases.sample(n=len(non_fraud_cases) - 1, random_state=42)

# Combine test set
X_test = np.vstack((test_fraud_case.iloc[:, :-1].values, non_fraud_cases_test.iloc[:, :-1].values))
y_test = np.hstack((test_fraud_case.iloc[:, -1].values, non_fraud_cases_test.iloc[:, -1].values))

# Prepare training data using all other non-fraud cases
remaining_non_fraud_cases = non_fraud_cases.drop(non_fraud_cases_test.index)

# Now create the training set with the remaining non-fraud cases
X_train = remaining_non_fraud_cases.iloc[:, :-1].values
y_train = remaining_non_fraud_cases.iloc[:, -1].values

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

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
    return acc, precision, recall, f1

# Calculate initial performance metrics on clean data
clean_acc, clean_precision, clean_recall, clean_f1 = get_model_performance(model, X_test, y_test)

# Create a SHAP explainer
explainer = shap.Explainer(model, X_train_resampled)

# Main app
st.title("Fraud Detection Model Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model Overview", "Adversarial Attacks", "Explainability", "Interactive Prediction Tool"])

# Model Overview Section
if section == "Model Overview":
    st.header("Model Overview")
    
    # Display performance metrics on clean data
    st.subheader("Performance on Clean Data")
    st.write(f"Accuracy: {clean_acc:.4f}")
    st.write(f"Precision: {clean_precision:.4f}")
    st.write(f"Recall: {clean_recall:.4f}")
    st.write(f"F1-Score: {clean_f1:.4f}")
    
    # Display performance metrics on adversarial data
    adv_acc, adv_precision, adv_recall, adv_f1 = get_model_performance(model, X_adv, y_adv)
    st.subheader("Performance on Adversarial Data")
    st.write(f"Accuracy: {adv_acc:.4f}")
    st.write(f"Precision: {adv_precision:.4f}")
    st.write(f"Recall: {adv_recall:.4f}")
    st.write(f"F1-Score: {adv_f1:.4f}")

    # Visualize fraud vs non-fraud transaction distribution
    st.subheader("Transaction Distribution")
    fraud_count = pd.Series(y_test).value_counts()
    sns.barplot(x=fraud_count.index, y=fraud_count.values)
    plt.title('Distribution of Fraud vs Non-Fraud Transactions')
    st.pyplot()

# Adversarial Attacks Section
elif section == "Adversarial Attacks":
    st.header("Adversarial Attacks")
    
    # Before vs. After Attack Comparison
    st.subheader("Before vs. After Attack")
    st.write("Model accuracy before attack: ", clean_acc)
    adv_acc, adv_precision, adv_recall, adv_f1 = get_model_performance(model, X_adv, y_adv)
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

# Explainability Section
elif section == "Explainability":
    st.header("Explainability with SHAP")
    
    # Feature importance plot
    st.subheader("Feature Importance Plot (SHAP)")
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot()
    
    # Per-transaction explanation
    st.subheader("Per-Transaction Explanation")
    idx = st.slider("Select Transaction Index", 0, len(X_test)-1)
    st.write(f"Transaction: {X_test[idx]}")
    shap.force_plot(explainer.expected_value, shap_values[idx], X_test[idx], matplotlib=True)
    st.pyplot()

# Interactive Prediction Tool Section
elif section == "Interactive Prediction Tool":
    st.header("Interactive Prediction Tool")
    
    # Input features for new transaction
    st.subheader("Input Transaction Features")
    transaction_input = []
    for i in range(X_test.shape[1]):
        feature_val = st.number_input(f"Feature {i+1}", value=float(X_test[0, i]))
        transaction_input.append(feature_val)
    
    # Predict fraud/not fraud
    transaction_input = np.array(transaction_input).reshape(1, -1)
    pred = (model.predict(transaction_input) > 0.5).astype(int)[0][0]
    st.write(f"Prediction: {'Fraud' if pred == 1 else 'Not Fraud'}")
    
    # Show SHAP explanations for the prediction
    st.subheader("Explanation for the Prediction")
    shap_values_input = explainer(transaction_input)
    shap.force_plot(explainer.expected_value, shap_values_input, transaction_input, matplotlib=True)
    st.pyplot()
