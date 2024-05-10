import tkinter as tk
from tkinter import filedialog, Text, Button, Label, DISABLED, NORMAL
import threading
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
 
 
def load_data_thread():
   thread = threading.Thread(target=load_and_preprocess_data)
   thread.start()
 
 
def load_and_preprocess_data():
   file_path = filedialog.askopenfilename()
   if file_path:
       status_label.config(text="Loading data, please wait...")
       try:
           # Loading only the necessary columns can significantly reduce memory usage
           data = pd.read_csv(file_path, usecols=['text', 'label'], on_bad_lines='skip')
           if 'label' in data.columns:
               data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
               data.drop('label', axis=1, inplace=True)
               train_model(data)
               status_label.config(text="Data loaded and model trained. You can now predict.")
               predict_button.config(state=NORMAL)
           else:
               status_label.config(text="Necessary column 'label' not found in the dataset.")
       except Exception as e:
           status_label.config(text=f"Failed to load data: {str(e)}")
 
 
def train_model(data):
   x_train, x_test, y_train, y_test = train_test_split(data['text'], data['fake'], test_size=0.2)
   global vectorizer, model
   vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
   x_train_vectorized = vectorizer.fit_transform(x_train)
   model = SVC(kernel="linear")
   model.fit(x_train_vectorized, y_train)
   accuracy = model.score(vectorizer.transform(x_test), y_test)
   status_label.config(text=f"Model Accuracy: {accuracy:.2f}")
 
 
def predict_article():
   if vectorizer is None or model is None:
       prediction_label.config(text="Model not ready. Please load data and train the model first.")
       return
   article_text = article_input.get("1.0", "end-1c")
   article_vectorized = vectorizer.transform([article_text])
   prediction = model.predict(article_vectorized)
   label = "FAKE" if prediction[0] == 1 else "REAL"
   prediction_label.config(text=f"Predicted Label: {label}")
 
 
root = tk.Tk()
root.title("News Article Classifier - The Pycodes")
 
 
# Status label
status_label = Label(root, text="Load data to start", fg="blue")
status_label.pack()
 
 
# Load data button
load_button = Button(root, text="Load Data", command=load_data_thread)
load_button.pack()
 
 
# User input for predictions
article_input = Text(root, height=5, width=50)
article_input.pack()
 
 
# Prediction button
predict_button = Button(root, text="Predict Article", command=predict_article, state=DISABLED)
predict_button.pack()
 
 
# Prediction result display
prediction_label = Label(root, text="")
prediction_label.pack()
 
 
# Global variables for model and vectorizer
model = None
vectorizer = None
 
 
root.mainloop()
