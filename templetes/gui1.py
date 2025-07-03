import gradio as gr
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load models and vectorizer
model_rf = pickle.load(open(r'../models/model_rf.pkl', 'rb'))
model_xgb = pickle.load(open(r'../models/model_xgb.pkl', 'rb'))
model_dt = pickle.load(open(r'../models/model_dt.pkl', 'rb'))
scaler = pickle.load(open(r'../models/scaler.pkl', 'rb'))
cv = pickle.load(open(r'../models/countVectorizer.pkl', 'rb'))

# Define preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    STOPWORDS = set(stopwords.words('english'))
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = ' '.join(review)
    return review

# Define prediction function for single text input
def predict_text(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = cv.transform([preprocessed_text]).toarray()
    scaled_text = scaler.transform(vectorized_text)
    prediction = model_xgb.predict(scaled_text)[0]
    prediction_proba = model_xgb.predict_proba(scaled_text)[0]
    result = "Positive" if prediction == 1 else "Negative"

    # Create a bar chart
    labels = ['Negative', 'Positive']
    fig, ax = plt.subplots()
    ax.bar(labels, prediction_proba, color=['red', 'green'])
    ax.set_ylim(0, 1)
    for i, v in enumerate(prediction_proba):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.title("Prediction Confidence")
    plt.ylabel("Probability")

    return result, fig

# Define prediction function for CSV input
def predict_csv(file):
    data = pd.read_csv(file.name)
    data['preprocessed'] = data['verified_reviews'].apply(preprocess_text)
    vectorized_data = cv.transform(data['preprocessed']).toarray()
    scaled_data = scaler.transform(vectorized_data)
    predictions = model_xgb.predict(scaled_data)
    data['prediction'] = predictions
    data['prediction'] = data['prediction'].apply(lambda x: "Positive" if x == 1 else "Negative")
    return data

# Create Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.Markdown("## Text Sentiment Prediction")
    with gr.Tab("Single Text Input"):
        text_input = gr.Textbox(lines=2, placeholder="Enter text for sentiment prediction")
        text_output = gr.Textbox(label="Predicted Sentiment")
        plot_output = gr.Plot(label="Prediction Confidence")
        text_button = gr.Button("Predict")
        text_button.click(fn=predict_text, inputs=text_input, outputs=[text_output, plot_output])
    
    with gr.Tab("CSV Input"):
        csv_input = gr.File(label="Upload CSV file for sentiment prediction")
        csv_output = gr.Dataframe(label="Predictions")
        csv_button = gr.Button("Predict")
        csv_button.click(fn=predict_csv, inputs=csv_input, outputs=csv_output)

demo.launch()
