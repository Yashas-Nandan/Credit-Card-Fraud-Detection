# Explainability Section
elif section == "Explainability":
    st.header("Explainability with Seaborn")
    
    # Calculate feature importances using basic correlations
    st.subheader("Feature Importance Plot (Correlation with Target)")
    feature_importance = pd.DataFrame({
        'Feature': data.columns[:-1],
        'Importance': np.abs(np.corrcoef(X_train_resampled.T, y_train_resampled)[-1, :-1])  # Absolute correlation between features and target
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title("Feature Importance Based on Correlation with Target")
    st.pyplot()

    # Per-Transaction Explanation: Show feature values for a selected transaction
    st.subheader("Per-Transaction Feature Values")
    idx = st.slider("Select Transaction Index", 0, len(X_test) - 1)
    selected_transaction = pd.DataFrame(X_test[idx], index=data.columns[:-1], columns=["Feature Value"])
    
    st.write(f"Transaction {idx}: Feature Values")
    st.dataframe(selected_transaction.T)
