# Install required libraries (if not installed)
# !pip install datasets scikit-learn nltk matplotlib seaborn

# Step 1: Import libraries
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
import string
import tkinter as tk
from tkinter import messagebox

# Step 2: Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Step 3: Load IMDB dataset
dataset = load_dataset("imdb")

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# Step 4: Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

train_df['clean_review'] = train_df['text'].apply(clean_text)
test_df['clean_review'] = test_df['text'].apply(clean_text)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_review'])
y_train = train_df['label']

X_test = vectorizer.transform(test_df['clean_review'])
y_test = test_df['label']

# Step 6: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluate Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = round(accuracy * 100, 2)

# Step 8: GUI Setup
def predict_sentiment():
    review = entry.get("1.0", "end-1c")
    if not review.strip():
        messagebox.showwarning("⚠️ Warning", "Please enter a review first.")
        return

    review_clean = clean_text(review)
    review_vector = vectorizer.transform([review_clean])
    prediction = model.predict(review_vector)[0]

    sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
    result_label.config(
        text=f"Predicted Sentiment: {sentiment}",
        fg="green" if prediction == 1 else "red"
    )

def clear_text():
    entry.delete("1.0", "end")
    result_label.config(text="")

# Create main window
root = tk.Tk()
root.title("🎬 IMDB Sentiment Analyzer")
root.geometry("520x450")
root.configure(bg="#f4f6f8")

# Title
title_label = tk.Label(root, text="🎬 Movie Review Sentiment Analyzer", font=("Arial", 16, "bold"), bg="#f4f6f8")
title_label.pack(pady=10)

# Accuracy Display
accuracy_label = tk.Label(root, text=f"Model Accuracy: {accuracy_percent}%", font=("Arial", 12, "bold"), bg="#f4f6f8", fg="blue")
accuracy_label.pack(pady=5)

# Input
entry_label = tk.Label(root, text="Enter your movie review below:", bg="#f4f6f8", font=("Arial", 12))
entry_label.pack(pady=5)

entry = tk.Text(root, height=8, width=55, wrap="word", font=("Arial", 11))
entry.pack(pady=10)

# Buttons
button_frame = tk.Frame(root, bg="#f4f6f8")
button_frame.pack(pady=5)

predict_button = tk.Button(button_frame, text="Predict Sentiment", command=predict_sentiment,
                           font=("Arial", 12, "bold"), bg="#007BFF", fg="white", padx=10, pady=5)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="Clear Review", command=clear_text,
                         font=("Arial", 12, "bold"), bg="#FF4B4B", fg="white", padx=10, pady=5)
clear_button.grid(row=0, column=1, padx=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f4f6f8")
result_label.pack(pady=20)

# Run GUI
root.mainloop()
