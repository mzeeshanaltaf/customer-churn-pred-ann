import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Load the encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Initialize streamlit app
page_title = "Customer Churn Prediction "
page_icon = "👨‍💼"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")

# Streamlit app title
st.title('Bank Customer Churn Prediction')
st.write(':blue[***Preventing Bank Customer Churn using Deep Learning Techniques***]')
st.write('*Given the bank customer data 📊, this application predicts the probability of a customer churn '
         'by applying artificial neural network techniques 🧠. This allows the bank to improve their customer '
         'service 👨🏻‍💻'
         'by targeting low hanging fruits and thus preventing churn.*')
st.info('Select the customer data from sidebar')

# User inputs
st.sidebar.header('Configuration')
st.sidebar.subheader('Demographic Information')
geography = st.sidebar.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 100)

st.sidebar.subheader('Financial Information')
balance = st.sidebar.number_input('Balance', min_value=0, value=1000, step=100, format='%d')
credit_score = st.sidebar.number_input('Credit Score', min_value=0, value=100, step=10, format='%d')
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0, value=1000, step=50, format='%d')

st.sidebar.subheader('Account Information')
tenure = st.sidebar.slider('Tenure', 0, 10)
num_of_products = st.sidebar.slider('Number of Products', 1, 4)
has_credit_card = st.sidebar.selectbox('Has Credit Card', ['No', 'Yes'])
has_credit_card = 1 if has_credit_card == 'Yes' else 0
is_active_member = st.sidebar.selectbox('Is Active Member', ['No', 'Yes'])
is_active_member = 1 if is_active_member == 'Yes' else 0

predict_button = st.button('Predict', type='primary')

if predict_button:
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
    })

    # One-hot encode 'Geography'
    geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

    # Combine One-hot encoded columns with input data
    input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_df)

    # Predict Churn
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]

    st.subheader('Prediction:')

    if prediction_prob > 0.5:
        st.write(":red[***The customer is likely to churn.***]")
    else:
        st.write(':green[***The customer is not likely to churn***]')

    st.write(f'Churn Probability: {prediction_prob:.2f}')
