import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Title of the app
st.title('Welcome to Car Price Predictor')

# Load the dataset
df = pd.read_csv('Car_data_Cleaned.csv')

# Unique sorted values for dropdowns
companies = sorted(df['company'].unique())
car_models = sorted(df['name'].unique())
years = sorted(df['year'].unique(), reverse=True)
fuel = df['fuel_type'].unique()

# Description of the app
st.write('This app predicts the price of a car you want to sell. Try filling in the details below:')

# Select the car company
company = st.selectbox('Select the company:', companies)

# Select the car model
model = st.selectbox('Select the model:', car_models)

# Select the year of purchase
year = st.selectbox('Select Year of Purchase:', years)

# Select the fuel type
fuel_type = st.selectbox('Select the Fuel Type:', fuel)

# Enter the number of kilometers driven
kilometers_driven = st.text_input('Enter the Number of Kilometers that the car has travelled:')

# Ensure the input for kilometers driven is a number (default to 0 if not provided)
if kilometers_driven == "":
    kilometers_driven = 0
else:
    kilometers_driven = int(kilometers_driven)

# Try to load the pre-trained model, if not found, train it and save the model
try:
    with open('car_price_model.pkl', 'rb') as file:
        pipe = pickle.load(file)

except FileNotFoundError:
    st.write("Training the model since no pre-trained model was found.")

    # Defining features and target
    x = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = df['Price']

    # Train-test split with a fixed random seed for consistency
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # OneHotEncoder for categorical columns
    ohe = OneHotEncoder()
    ohe.fit(x[['name', 'company', 'fuel_type']])

    # Column transformer for applying one-hot encoding
    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
        remainder='passthrough'
    )

    # Linear regression model pipeline
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)

    # Train the model
    pipe.fit(x_train, y_train)

    # Save the trained model to a file
    with open('car_price_model.pkl', 'wb') as file:
        pickle.dump(pipe, file)



# Predicting price
if st.button('Predict Price'):
    # Create a DataFrame with the values entered by the user
    input_data = pd.DataFrame([[model, company, year, kilometers_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Make the prediction using the model
    predicted_price = pipe.predict(input_data)

    # Display the predicted price
    st.write(f"The estimated price of the car is: {predicted_price[0]:.2f}")

