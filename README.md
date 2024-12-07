# Aggregating News Articles Using Text Classification

![Project Banner](image/ML2.jpg "Aggregating News Articles Banner")

## Overview
This repository contains the implementation of a project aimed at **classifying and organizing news articles** using advanced machine learning and natural language processing (NLP) techniques. The project leverages models like Support Vector Machines (SVM), Random Forest, and Logistic Regression to categorize news articles into five predefined topics: Politics, Entertainment, Technology, Sports, and Business.

---

## Dataset
![Dataset Overview](images/dataset_overview.png "Dataset Overview")

We used the **BBC News Classification** dataset from Kaggle:
- **Number of Records**: 2,127 articles.
- **Features**: 7 columns.
- **Topics**: Politics, Entertainment, Technology, Sports, Business.

### Data Preprocessing
- **Missing Values**: Replaced with median values.
- **Outliers**: Addressed via summarization features.
- **Duplicates**: Removed 69 duplicate records.
- **Cleaned Dataset**: Enhanced for better classification performance.

---

## Key Insights
### Complexity and Readability by Topic
![Insights Chart](images/complexity_vs_readability.png "Complexity vs Readability")

- Articles in **Politics** and **Business** are more complex, while **Sports** articles are simpler and score higher on readability.
- Shorter articles are often more readable, while longer articles require more complex summarization.

---

## Machine Learning Models
### Accuracy Comparison
![Model Accuracy](images/model_accuracy.png "Model Accuracy")

| Model                      | Accuracy   | Key Features                                                                 |
|----------------------------|------------|------------------------------------------------------------------------------|
| **Logistic Regression**    | 97.25%     | Generalizes well; signs of continued improvement beyond training data.       |
| **Random Forest**          | 97.90%     | Low bias and variance; good generalization to unseen data.                   |
| **Support Vector Machine** | 97.25%     | Low bias and variance; generalizes well and remains stable.                  |
| **SVM with PCA**           | 96.28%     | Hyperparameter tuning; lowest variance and bias; stable performance.         |

---

## Deployment
### Streamlit Application
![Streamlit App](images/streamlit_ui.png "Streamlit UI")

The trained model and application are deployed using **Streamlit** for:
- Instant classification of newly uploaded data.
- A user-friendly interface for users.

---

## Limitations
1. Limited dataset size and one source of data (BBC News).
2. Constraints in equipment for large-scale processing.
3. Restricted to five predefined topic labels.

---

## Future Enhancements
- Expand dataset to include diverse news sources.
- Explore differences in precision and recall among various topic labels.
- Investigate words with high predictive power to refine model performance.

---

## Installation and Usage

### Prerequisites
- Python 3.8 or later
- Required libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/aggregating-news-articles.git
