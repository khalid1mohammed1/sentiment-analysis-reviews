import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Load models and vectorizer
model_rf = pickle.load(open(r'C:\Users\DUBAU STORE\AppData\Local\Programs\Python\Python312\sentiment_analysis\models\model_rf.pkl', 'rb'))
model_xgb = pickle.load(open(r'C:\Users\DUBAU STORE\AppData\Local\Programs\Python\Python312\sentiment_analysis\models\model_xgb.pkl', 'rb'))
model_dt = pickle.load(open(r'C:\Users\DUBAU STORE\AppData\Local\Programs\Python\Python312\sentiment_analysis\models\model_dt.pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\DUBAU STORE\AppData\Local\Programs\Python\Python312\sentiment_analysis\models\scaler.pkl', 'rb'))
cv = pickle.load(open(r'C:\Users\DUBAU STORE\AppData\Local\Programs\Python\Python312\sentiment_analysis\models\countVectorizer.pkl', 'rb'))

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
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Negative', 'Positive'], prediction_proba, color=['red', 'green'])
    ax.set_ylim(0, 1)
    for i, v in enumerate(prediction_proba):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    ax.set_title("Prediction Confidence", fontsize=16)
    ax.set_ylabel("Probability", fontsize=14)
    ax.set_xlabel("Sentiment", fontsize=14)
    return result, fig

# Define prediction function for CSV input
def predict_csv(file_path):
    data = pd.read_csv(file_path)
    data['preprocessed'] = data['verified_reviews'].apply(preprocess_text)
    vectorized_data = cv.transform(data['preprocessed']).toarray()
    scaled_data = scaler.transform(vectorized_data)
    predictions = model_xgb.predict(scaled_data)
    data['prediction'] = predictions
    data['prediction'] = data['prediction'].apply(lambda x: "Positive" if x == 1 else "Negative")
    return data

# Function to open file dialog and get CSV file path
def browse_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = predict_csv(file_path)
        display_predictions(df)

# Function to display predictions in a new window
def display_predictions(df):
    predictions_window = tk.Toplevel(root)
    predictions_window.title("Predictions")
    predictions_text = tk.Text(predictions_window, wrap='word', height=10, width=50)
    predictions_text.insert(tk.END, df.to_string())
    predictions_text.pack(expand=True, fill='both')
    predictions_window.mainloop()

# Function to handle text prediction
def predict_from_text():
    text = text_input.get("1.0", "end-1c")
    if text:
        result, fig = predict_text(text)
        predicted_sentiment_label.config(text="Predicted Sentiment: " + result)
        canvas = FigureCanvasTkAgg(fig, master=predictions_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
    else:
        messagebox.showerror("Error", "Please enter some text for prediction.")

# GUI setup
root = tk.Tk()
root.title("Text Sentiment Prediction")
root.geometry("900x700")
root.configure(bg='#0A2032')

# Frame for title
title_frame = tk.Frame(root, bg='#0A2032')
title_frame.pack(pady=20)
title_label = tk.Label(title_frame, text="Understand the emotions behind the words.😊", font=("Helvetica", 24), bg='#0A2032', fg='white')
title_label.pack()

# Frame for text prediction
text_frame = tk.Frame(root, bg='#0A2032')
text_frame.pack(pady=20)

text_label = tk.Label(text_frame, text="Enter text for sentiment prediction:", font=("Helvetica", 14), bg='#0A2032', fg='white')
text_label.grid(row=0, column=0, padx=5, pady=5)

text_input = tk.Text(text_frame, height=5, width=50, bg='white', fg='black', font=("Helvetica", 12))
text_input.grid(row=1, column=0, padx=5, pady=5)

predict_button = tk.Button(text_frame, text="Predict", command=predict_from_text, bg='#11DFA9', fg='white', font=("Helvetica", 12))
predict_button.grid(row=2, column=0, padx=5, pady=5)

predicted_sentiment_label = tk.Label(text_frame, text="", font=("Helvetica", 14), bg='#0A2032', fg='white')
predicted_sentiment_label.grid(row=3, column=0, padx=5, pady=5)

# Frame for CSV prediction
csv_frame = tk.Frame(root, bg='#0A2032')
csv_frame.pack(pady=20)

browse_button = tk.Button(csv_frame, text="Browse CSV", command=browse_csv, bg='#11DFA9', fg='white', font=("Helvetica", 12))
browse_button.pack(padx=5, pady=5)

# Frame to display predictions
predictions_frame = tk.Frame(root, bg='#0A2032')
predictions_frame.pack(pady=20, fill=tk.BOTH, expand=True)

root.mainloop()
