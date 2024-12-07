import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

# Load saved preprocessing artifacts and model
vectorizer = joblib.load("vectorizer.pkl")
pca = joblib.load("pca.pkl")
svc = joblib.load("svm_model.pkl")
# test_set = pd.read_csv("test_set.csv")  # Pre-saved test set for consistent accuracy

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

# Header
st.title("News Aggregation and Filtering Tool")

# Display the image and team credit
st.image(
    "image/ML2.jpg",  # Replace with your image path
    use_column_width=True,
)

st.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        <strong>Built by:</strong> Team 2 - Kirsten Drennen, Taylor Kirk, Marwah Faraj
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar: Static elements
st.sidebar.title("Model Performance")

# Display static accuracy in the sidebar
st.sidebar.markdown("**Spot-on predictions for 96% of the topics—see the results for yourself!**")

# Sidebar: Topic selection
st.sidebar.markdown("### Filter by Topic")
topic_placeholder = st.sidebar.empty()  # Placeholder for dropdown
selected_topic = None

# File uploader for new test data
uploaded_file = st.file_uploader("Upload your test data CSV file", type=["csv"])

if uploaded_file:
    # Load uploaded test file
    user_test_df = pd.read_csv(uploaded_file)

    if "text" in user_test_df.columns:
        st.success("File uploaded successfully!")

        # Clean text data
        user_test_df["cleaned_text"] = clean_text_column(user_test_df, "text")

        # Apply preprocessing steps
        X_user_test = vectorizer.transform(user_test_df["cleaned_text"])
        X_user_test_pca = pca.transform(X_user_test.toarray())

        # Predict labels
        user_test_df["predicted_labels"] = svc.predict(X_user_test_pca)

        # Add icons for display
        label_icons = {
            "business": "💼",
            "sport": "⚽",
            "politics": "🏛️",
            "tech": "💻",
            "entertainment": "🎬"
        }
        user_test_df["labeled_with_icon"] = user_test_df["predicted_labels"].apply(
            lambda label: f"{label_icons.get(label.lower(), '📰')} {label}"
        )

        # Sidebar: Update topic filtering
        unique_topics_with_icons = [
            f"{label_icons.get(topic.lower(), '📰')} {topic}" for topic in user_test_df["predicted_labels"].unique()
        ]
        selected_topic_with_icon = topic_placeholder.selectbox(
            "Select a topic:", options=unique_topics_with_icons
        )
        selected_topic = selected_topic_with_icon.split(" ", 1)[1]  # Extract raw topic name

        # Filter and display data
        filtered_df = user_test_df[user_test_df["predicted_labels"] == selected_topic]
        st.subheader(f"Articles for Topic: {selected_topic_with_icon}")
        st.dataframe(filtered_df[["text", "predicted_labels"]])

        # Allow downloading filtered results
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label=f"Download {selected_topic_with_icon} News as CSV",
            data=csv,
            file_name=f"{selected_topic}_news.csv",
            mime="text/csv",
        )
    else:
        st.error("The uploaded file must contain a 'text' column.")
else:
    # Sidebar: Initial placeholder before uploading
    topic_placeholder.markdown("Upload a dataset to filter by topic.")
