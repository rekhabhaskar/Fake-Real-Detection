
import streamlit as st
import joblib
import warnings

warnings.filterwarnings('ignore')

 
import numpy as np  # linear algebra   
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv) 

# df1 = pd.read_csv("True.csv", sep=',', encoding='ISO-8859-1', skiprows=lambda x: x in [11067])
# df2 = pd.read_csv("Fake.csv", sep=',', encoding='ISO-8859-1', skiprows=lambda x: x in [11067]) 

# try:
#     df1 = pd.read_csv("True.csv", sep=',', encoding='ISO-8859-1', skiprows=lambda x: x in [11067])
#     df2 = pd.read_csv("Fake.csv", sep=',', encoding='ISO-8859-1', skiprows=lambda x: x in [11067])
# except pd.errors.ParserError as e:
#     print(f"Error reading CSV: {e}")
#     # Handle the error as needed

# df1 = pd.read_csv("True.csv", sep=',', encoding='ISO-8859-1', skiprows=lambda x: x in [11067])
# df2 = pd.read_csv("Fake.csv", sep=',', encoding='ISO-8859-1', skiprows=lambda x: x in [11067]) 
# data = pd.concat([df1, df2], ignore_index=True, sort=False)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl') 


def predict_news(input_text, vectorizer, model):
    input_tfidf = vectorizer.transform([input_text])
    prediction = model.predict(input_tfidf)[0]
    return prediction


# Streamlit app
st.title("📰 Fake News Detection App")
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stButton button {
            background-color: #4caf50;
            color: white;
        }
        .stSelectbox {
            width: 80%;
        }
        .stText {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Select a text from the loaded data
# selected_text = st.selectbox("Select a text:", data['text'])
selected_text = st.text_input('')

# Display the selected text
st.markdown("<p class='stText'>Selected Text:</p>", unsafe_allow_html=True)
st.write(selected_text)

# Button to make predictions
if st.button("Predict"):
    # Make a prediction
    prediction = predict_news(selected_text, vectorizer, model)

    # Display the prediction result
    st.markdown(f"<p class='stText'>Prediction: {'Real' if prediction == 1 else 'Fake'}</p>", unsafe_allow_html=True)



    