import tkinter as tk
from tkinter import messagebox
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_toxicity():
    input_text = text_entry.get("1.0", tk.END).strip()

    if not input_text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return

    input_tfidf = vectorizer.transform([input_text])

    prediction = model.predict(input_tfidf)[0]
    
    if prediction == 1:
        result = "Toxic"
    else:
        result = "Normal"
    
    result_label.config(text=f"Result: {result}")


root = tk.Tk()
root.title("Toxicity Prediction")


text_entry = tk.Text(root, height=10, width=50)
text_entry.pack(pady=10)

predict_button = tk.Button(root, text="Predict", command=predict_toxicity)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="Result: ")
result_label.pack(pady=10)

root.mainloop()
