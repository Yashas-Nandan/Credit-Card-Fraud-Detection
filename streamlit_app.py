import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv('creditcard.csv')  # Adjust with your dataset path
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target (fraud/not fraud)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Function to build a neural network model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_resampled.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(X_train, y_train):
    model = build_model()
    class_weight = {0: 1, 1: 10}  # Give more weight to fraud cases
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, class_weight=class_weight, validation_split=0.2)
    return model, history

# Train Random Forest model for comparison
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(class_weight={0: 1, 1: 10})  # Higher weight for fraud class
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to calculate model performance
def get_model_performance(model, X, y):
    if hasattr(model, 'predict'):
        y_pred = (model.predict(X) > 0.5).astype("int32")
    else:
        y_pred = model.predict(X)
        
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return acc, precision, recall, f1

# Build and train both models
nn_model, nn_history = train_model(X_train_resampled, y_train_resampled)
rf_model = train_random_forest(X_train_resampled, y_train_resampled)

# Calculate initial performance metrics on clean data
clean_metrics_nn = get_model_performance(nn_model, X_test, y_test)
clean_metrics_rf = get_model_performance(rf_model, X_test, y_test)

# Function to create adversarial examples
def generate_adversarial_examples(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)  # Generate Gaussian noise
    X_adv = X + noise  # Add noise to create adversarial examples
    X_adv = np.clip(X_adv, 0, None)  # Ensure no negative values
    return X_adv

# Generate adversarial examples
X_adv = generate_adversarial_examples(X_test)
y_adv = y_test  # Assuming labels remain the same for this example

# Calculate performance metrics on adversarial data
adv_metrics_nn = get_model_performance(nn_model, X_adv, y_adv)
adv_metrics_rf = get_model_performance(rf_model, X_adv, y_adv)

# Main Streamlit app
st.title("Fraud Detection Model Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model Overview", "Interactive Prediction Tool", "Adversarial Attacks"])

# Model Overview Section
if section == "Model Overview":
    st.header("Model Overview")
    
    # Display metrics for Neural Network
    st.subheader("Performance on Clean Data (Neural Network)")
    st.write(f"Accuracy: {clean_metrics_nn[0]:.4f}")
    st.write(f"Precision: {clean_metrics_nn[1]:.4f}")
    st.write(f"Recall: {clean_metrics_nn[2]:.4f}")
    st.write(f"F1-Score: {clean_metrics_nn[3]:.4f}")

    # Display metrics for Random Forest
    st.subheader("Performance on Clean Data (Random Forest)")
    st.write(f"Accuracy: {clean_metrics_rf[0]:.4f}")
    st.write(f"Precision: {clean_metrics_rf[1]:.4f}")
    st.write(f"Recall: {clean_metrics_rf[2]:.4f}")
    st.write(f"F1-Score: {clean_metrics_rf[3]:.4f}")

    # Visualize confusion matrix for clean data
    st.subheader("Confusion Matrix for Clean Data (Neural Network)")
    cm = confusion_matrix(y_test, (nn_model.predict(X_test) > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Clean Data - NN)")
    st.pyplot()

    # Visualize confusion matrix for adversarial data
    st.subheader("Confusion Matrix for Adversarial Data (Neural Network)")
    cm_adv_nn = confusion_matrix(y_adv, (nn_model.predict(X_adv) > 0.5).astype(int))
    sns.heatmap(cm_adv_nn, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Adversarial Data - NN)")
    st.pyplot()

# Adversarial Attacks Section
elif section == "Adversarial Attacks":
    st.header("Adversarial Attacks")
    
    # Before vs. After Attack Comparison
    st.subheader("Before vs. After Attack")
    st.write("NN Model accuracy before attack: ", round(clean_metrics_nn[0], 2))
    st.write("NN Model accuracy after attack: ", round(adv_metrics_nn[0], 2))
    
    # Generate adversarial example
    st.subheader("Adversarial Example")
    idx = st.slider("Select Transaction Index", 0, len(X_adv)-1)
    st.write(f"Original Transaction: {X_test[idx]}")
    st.write(f"Adversarial Transaction: {X_adv[idx]}")
    
    # Reshape input for prediction
    original_input = X_test[idx:idx+1]  # Ensure correct shape for model input
    adversarial_input = X_adv[idx:idx+1]  # Ensure correct shape for model input
    
    # Get predictions
    original_pred = (nn_model.predict(original_input) > 0.5).astype(int)[0][0]  # Reshape input
    adv_pred = (nn_model.predict(adversarial_input) > 0.5).astype(int)[0][0]  # Reshape input
    
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

elif section == "Explainability":
    st.header("Explainability with SHAP")
    
    # Create a SHAP explainer
    explainer = shap.KernelExplainer(model.predict, X_train_resampled[:100])  # Limit to 100 samples for faster SHAP calculations
    
    # Feature importance plot
    st.subheader("Feature Importance Plot (SHAP)")
    shap_values = explainer.shap_values(X_test[:100])  # Limit X_test for faster visualization
    shap.summary_plot(shap_values, X_test[:100], show=False)
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
    
    # Input features for a new transaction
    st.subheader("Input Transaction Features")
    transaction_input = []
    for i in range(X_test.shape[1]):
        feature_val = st.number_input(f"Feature {i+1}", value=float(X_test[0, i]))
        transaction_input.append(feature_val)

    # Predict fraud/not fraud
    transaction_input = np.array(transaction_input).reshape(1, -1)
    transaction_input_scaled = scaler.transform(transaction_input)  # Scale the input
    pred_prob = nn_model.predict(transaction_input_scaled)[0][0]
    pred_label = "Fraud" if pred_prob > 0.5 else "Not Fraud"
    
    st.write(f"Prediction Probability: {pred_prob:.4f}")
    st.write(f"Prediction: {pred_label}")
