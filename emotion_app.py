import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# --- Download NLTK resources (if first time) ---
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# --- Text preprocessing function ---
def preprocess_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = ''.join([c for c in txt if not c.isdigit()])
    txt = ' '.join([word for word in txt.split()
                    if not (word.startswith("http") or word.startswith("www"))])
    txt = ''.join([c for c in txt if c.isascii()])
    words = word_tokenize(txt)
    txt = ' '.join([w for w in words if w not in stop_words])
    return txt

# --- Load saved model and vectorizer ---
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
logistic_model = pickle.load(open("logistic_model.pkl", "rb"))

# --- Streamlit App UI ---
st.title("Emotion Prediction from Text")
st.write("Enter text and the app will predict the emotion.")

user_input = st.text_area("Type your text here:")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed_text = preprocess_text(user_input)
        vectorized_text = tfidf_vectorizer.transform([processed_text])
        prediction = logistic_model.predict(vectorized_text)

        # Correct emotion mapping (based on your dataset)
        emotion_mapping = {
            0: 'sadness',
            1: 'anger',
            2: 'love',
            3: 'surprise',
            4: 'fear',
            5: 'joy'
        }

        st.success(f"Predicted Emotion: {emotion_mapping[prediction[0]]}")


