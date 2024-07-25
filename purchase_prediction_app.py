import streamlit as st
import pickle
import numpy as np

# Load the trained pipeline
with open('Customer_Purchase_model1.pkl', 'rb') as file:
    pipeline = pickle.load(file)

    st.title("Customer Purchase Prediction")

    # Input fields
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
    annual_income = st.number_input('Annual Income', min_value=0, value=50000)
    number_of_purchases = st.number_input('Number of Purchases', min_value=0, value=1)
    product_category = st.selectbox('Product Category', options=[0, 1, 2, 3, 4], format_func=lambda x:
                                    ['Electronics', 'Clothing', 'Home Goods', 'Beauty', 'Sports'][x])
    time_spent_on_website = st.number_input('Time Spent on Website (minutes)', min_value=0, value=10)
    loyalty_program = st.selectbox('Loyalty Program', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    discounts_availed = st.number_input('Discounts Availed', min_value=0, max_value=5, value=0)

    # Create input data
    input_data = np.array([[age, gender, annual_income, number_of_purchases, product_category,
                            time_spent_on_website, loyalty_program, discounts_availed]])

    # Prediction
    if st.button('Predict'):
        prediction = pipeline.predict(input_data)
        st.write('Prediction:', 'Yes' if prediction[0] == 1 else 'No')



