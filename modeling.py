import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("/Users/marwahfaraj/Desktop/ms_degree_application_and_doc/final_projects/502_final_project/BBC_new_topic_classification/BBC_new_topic_classification/data/bbc_news_text_complexity_summarization.csv")

# Split into train, validation, and test sets
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['labels'])

# Save the test set separately (untouched)
test_df.to_csv("test_set.csv", index=False)
print("Test set saved successfully.")


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from nltk.corpus import stopwords

# Preprocessing function
def clean_text_column(df, column_name):
    stop_words = set(stopwords.words('english'))
    cleaned_texts = []

    for text in df[column_name]:
        if not isinstance(text, str):
            text = ""  # Handle non-string or NaN values
        text = text.replace('\n', '')  # Remove newlines
        words = re.findall(r'\b\w+\b', text.lower())  # Tokenize and lowercase
        filtered_words = [word for word in words if word not in stop_words]  # Remove stop words
        cleaned_texts.append(' '.join(filtered_words))  # Join back into a string

    return cleaned_texts

# Clean train and validation data
train_df['cleaned_text'] = clean_text_column(train_df, 'text')
val_df['cleaned_text'] = clean_text_column(val_df, 'text')

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000, max_df=0.9, ngram_range=(1, 3))
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
X_val = vectorizer.transform(val_df['cleaned_text'])

# PCA Transformation
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train.toarray())
X_val_pca = pca.transform(X_val.toarray())

# Train the Model
svc = SVC(kernel="linear", C=1, random_state=42)
svc.fit(X_train_pca, train_df['labels'])

# Save preprocessing artifacts and model
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(svc, "svm_model.pkl")
train_df.to_csv("cleaned_train_set.csv", index=False)
val_df.to_csv("cleaned_val_set.csv", index=False)

print("Train and validation preprocessing artifacts and model saved.")
