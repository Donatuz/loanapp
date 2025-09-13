# --- Import libraries ---
import numpy as np
import pandas as pd 
import streamlit as st 
import plotly.express as px 
import pickle

# --- Load model and dataset ---
loaded_model = pickle.load(open('loan_classifier', 'rb'))
load = pd.read_csv('bankloan.csv')

# --- Prediction function ---
def loan_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return 'Your loan has not been approved' if prediction == 0 else 'Congratulations, your loan has been approved'

# --- Chart page ---
def chart_page():
    st.title('Loan Applicants by Gender')
    count_gender = px.histogram(
        load, 
        x='Gender', 
        color='Loan_Status',
        title='Gender Status of Loan Applicants',
        labels={'Loan_Status': 'Loan Status'}
    )
    st.plotly_chart(count_gender)

    # Insights
    st.subheader('Insights')
    st.markdown('Men apply for loans more than women.')

# --- Dashboard (Form) page ---
def dashboard_page():
    st.title('Loan Application Form')
    st.markdown('Fill in your details below')

    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox('Gender (0=Female, 1=Male)', options=[0,1])
        Married = st.selectbox('Married (0=No, 1=Yes)', options=[0,1])
        Dependents = st.selectbox('Dependents (0 or 1)', options=[0,1])
        Education = st.selectbox('Education (0=Grads, 1=Not Grads)', options=[0,1])

    with col2:
        Self_Employed = st.selectbox('Self Employed (0=No, 1=Yes)', options=[0,1])
        ApplicantIncome = st.number_input('Applicant Income', value=0)
        CoapplicantIncome = st.number_input('Coapplicant Income', value=0)
        Loan_Amount = st.number_input('Loan Amount', value=0)

    with col3:
        Loan_Amount_Term = st.number_input('Loan Amount Term', value=0)
        Credit_History = st.selectbox('Credit History (0 or 1)', options=[0,1])
        Property_Area_Rural = st.selectbox('Property Area Rural (0 or 1)', options=[0,1])
        Property_Area_Urban = st.selectbox('Property Area Urban (0 or 1)', options=[0,1])

    # Prediction button
    if st.button('Bank Loan Application System'):
        try:
            input_data = [
                int(Gender),
                int(Married),
                int(Dependents),
                int(Education),
                int(Self_Employed),
                int(ApplicantIncome),
                float(CoapplicantIncome),
                float(Loan_Amount),
                int(Loan_Amount_Term),
                float(Credit_History),
                int(Property_Area_Rural),
                int(Property_Area_Urban)
            ]
            result = loan_prediction(input_data)
            st.success(result) 
        except ValueError:
            st.error('Please enter valid input values.')

# --- Sidebar navigation ---
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select Page', ['Chart', 'Form Inputs'])

    if page == 'Chart':
        chart_page()
    elif page == 'Form Inputs':
        dashboard_page()

# --- Run app ---
if __name__ == '__main__':
    main()
