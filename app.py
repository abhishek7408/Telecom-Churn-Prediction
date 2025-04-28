
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('dt_model.pkl')

# Feature list
features = ['Count', 'Latitude', 'Longitude', 'Gender', 'Senior_Citizen', 'Partner', 'Dependents', 'Tenure_Months', 'Phone_Service',
            'Multiple_Lines', 'Internet_Service', 'Online_Security', 'Online_Backup', 'Device_Protection', 'Tech_Support', 'Streaming_TV',
            'Streaming_Movies', 'Contract', 'Paperless_Billing', 'Payment_Method', 'Monthly_Charges', 'Total_Charges', 'Churn_Value', 'Churn_Score', 'CLTV']

# Sidebar option
option = st.sidebar.selectbox("Select Prediction Type:", ("Single Input", "Bulk Upload"))

st.title("Customer Churn Prediction App")

if option == "Single Input":
    st.header("Enter Customer Details:")
    input_data = {}
    for feature in features:
        input_data[feature] = st.selectbox(f"{feature}:", [0, 1])
    
    input_df = pd.DataFrame([input_data])
    
    if st.button('Predict'):
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")

elif option == "Bulk Upload":
    st.header("Upload CSV File:")
    file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    if file is not None:
        data = pd.read_csv(file)
        predictions = model.predict(data)
        data['Prediction'] = ['Churn' if pred == 1 else 'No Churn' for pred in predictions]
        st.write(data)
        
        # Downloadable csv
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv',
        )
