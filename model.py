import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import re

# Function for basic text cleaning
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = text.lower()                  # Convert to lowercase
    return text

# Load dataset
df = pd.read_csv("/Users/marwahfaraj/Desktop/ms_degree_application_and_doc/final_projects/502_final_project/BBC_new_topic_classification/data/bbc_news_text_complexity_summarization.csv")  # Replace with your dataset path
if "text" not in df.columns or "labels" not in df.columns:
    raise ValueError("The dataset must have 'text' and 'labels' columns.")

# Clean text data
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["labels"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose the model: Random Forest or MLP
# Option 1: Random Forest
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# Option 2: Uncomment the following line to use MLP instead
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred, average="weighted") * 100
recall = recall_score(y_test, y_pred, average="weighted") * 100
f1 = f1_score(y_test, y_pred, average="weighted") * 100

# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1:.2f}%")

# Save the model, vectorizer, and best metric (F1)
joblib.dump(model, "saved_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump({"f1_score": f1}, "metrics.pkl")
print("Model, vectorizer, and metrics saved successfully.")
