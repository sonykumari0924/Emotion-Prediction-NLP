# Emotion Prediction from Text using NLP

This project demonstrates **text data preprocessing, analysis, and emotion prediction** using Python, NLP, and machine learning.

## Project Overview
The project performs the following tasks:
- Loads text data for training a model
- Cleans and preprocesses text (lowercasing, removing punctuation, numbers, links, emojis, stopwords)
- Converts text into numerical features using **TF-IDF vectorization**
- Trains a **Logistic Regression** classifier to predict emotions
- Builds a **Streamlit web interface** for real-time emotion prediction

The model classifies text into six emotion categories: **Sadness, Anger, Love, Surprise, Fear, and Joy**.

## Assignments Covered
### Assignment 1: Text Data Cleaning & Preprocessing
- Converted text to lowercase
- Removed punctuation, numbers, links, and emojis
- Removed stopwords using NLTK

### Assignment 2: Feature Engineering & Model Training
- Transformed text into numerical features using TF-IDF
- Trained a Logistic Regression model on processed text
- Saved the trained model and TF-IDF vectorizer using pickle

### Assignment 3: Real-Time Emotion Prediction
- Created a Streamlit web app for user input
- Loaded trained model and vectorizer for instant predictions
- Displayed predicted emotion in the app interface

## Files in the Repository
- `Emotion_Prediction_NLP.ipynb` – Jupyter Notebook with preprocessing, model training, and evaluation
- `emotion_app.py` – Streamlit web app for real-time emotion prediction
- `logistic_model.pkl` – Saved trained Logistic Regression model
- `tfidf_vectorizer.pkl` – Saved TF-IDF vectorizer
- `train.txt` – Sample training dataset
- `README.md` – Project documentation

## Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- Streamlit
- Pickle

