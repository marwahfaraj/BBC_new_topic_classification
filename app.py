import streamlit as st
import pandas as pd
import joblib
import re

# Function for basic text cleaning
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = text.lower()                  # Convert to lowercase
    return text

# Updated label-to-icon mapping based on actual labels
label_icons = {
    "business": "💼",
    "sport": "⚽",
    "politics": "🏛️",
    "tech": "💻",
    "entertainment": "🎬"
}

# Load the saved model, vectorizer, and metrics
model = joblib.load("saved_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
metrics = joblib.load("metrics.pkl")

# Header Image
st.image(
    "/Users/marwahfaraj/Desktop/ms_degree_application_and_doc/final_projects/502_final_project/BBC_new_topic_classification/ML2.jpg",
    use_column_width=True,
    caption="News Aggregation and Filtering App"
)

# Team Credit Below Image
st.markdown(
    """
    <div style='text-align: center; font-size: 16px;'>
        <strong>Built by:</strong> Team 2 - Kirsten Emily Drennen, Taylor Kirk, Marwah Faraj
    </div>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("News Aggregation and Filtering Tool")

# Display Model Performance
st.sidebar.title("Model Performance")
st.sidebar.markdown(
    f"""
    **This model predicts topics with an F1-Score of {metrics['f1_score']:.2f}%**  
    *(Weighted average across all classes)*
    """
)

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV file with news articles", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)

    # Check for required columns
    if "text" in df.columns:
        st.success("File uploaded successfully!")

        # Preprocess text data
        df["cleaned_text"] = df["text"].apply(preprocess_text)

        # Predict labels
        X = vectorizer.transform(df["cleaned_text"])
        df["labels"] = model.predict(X)

        # Convert labels to uppercase for display
        df["labels"] = df["labels"].str.upper()

        # Add icons to labels for display
        df["labeled_with_icon"] = df["labels"].apply(
            lambda label: f"{label_icons.get(label.lower(), '📰')} {label}"
        )

        # Sidebar for filtering
        st.sidebar.title("Filter News by Topics")
        unique_labels_with_icons = [
            f"{label_icons.get(label.lower(), '📰')} {label}" for label in df["labels"].unique()
        ]
        selected_label_with_icon = st.sidebar.selectbox(
            "Choose a topic:", options=unique_labels_with_icons
        )

        # Map the selected icon-label combination back to the raw label
        selected_label = selected_label_with_icon.split(" ", 1)[1]

        # Filter and display articles
        if selected_label:
            filtered_df = df[df["labels"] == selected_label]
            st.subheader(f"Showing articles for: {selected_label_with_icon}")
            st.dataframe(filtered_df[["text", "labels"]])  # Only show uppercase labels

            # Allow downloading the filtered data without the icon column
            csv = filtered_df[["text", "labels"]].to_csv(index=False)  # Exclude 'labeled_with_icon'
            st.download_button(
                label=f"Download {selected_label_with_icon} News as CSV",
                data=csv,
                file_name=f"{selected_label}_news.csv",
                mime="text/csv"
            )
    else:
        st.error("The uploaded file must contain a 'text' column!")
else:
    st.info("Please upload a CSV file to get started.")
