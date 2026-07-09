# Emotion Detection from Text using Natural Language Processing (NLP)

##  Project Overview

This project presents a complete Natural Language Processing (NLP) pipeline for detecting emotions from text using traditional Machine Learning algorithms.

The objective is to classify a given sentence into one of six emotion categories by applying text preprocessing, feature extraction using TF-IDF, and multiple classification algorithms.

This project demonstrates the complete workflow of an NLP classification task, from data preparation to model evaluation and prediction.

---

# Problem Statement

Understanding emotions expressed in text is an important task in Natural Language Processing. Emotion detection has many real-world applications, including:

* Chatbots
* Customer feedback analysis
* Mental health monitoring
* Social media analysis
* Recommendation systems

The goal of this project is to build a machine learning model capable of accurately predicting the emotion contained in a text message.

---

#  Dataset

Dataset used:

**DAIR-AI Emotion Dataset (Hugging Face)**

https://huggingface.co/datasets/dair-ai/emotion

The dataset contains English text samples labeled with six different emotions.

### Emotion Labels

| Label | Emotion  |
| ----- | -------- |
| 0     | Sadness  |
| 1     | Joy      |
| 2     | Love     |
| 3     | Anger    |
| 4     | Fear     |
| 5     | Surprise |

### Dataset Size

| Split      | Samples    |
| ---------- | ---------- |
| Train      | 16,000     |
| Validation | 2,000      |
| Test       | 2,000      |
| **Total**  | **20,000** |

---

#  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Seaborn
* Google Colab

---

#  Project Workflow

```
Dataset
        ↓
Data Exploration
        ↓
Data Cleaning
        ↓
Dataset Concatenation
        ↓
Train-Test Split
        ↓
TF-IDF Vectorization
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Prediction
```

---

#  Data Preprocessing

The following preprocessing steps were performed:

* Checked dataset structure
* Checked missing values
* Removed duplicate records
* Combined train, validation, and test datasets
* Split the dataset into training and testing sets
* Used stratified sampling to preserve the class distribution

---

#  Feature Extraction
Machine learning algorithms cannot process raw text directly.

Therefore, TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert text into numerical feature vectors.

The TF-IDF vectorizer was fitted only on the training data to prevent data leakage.

---

#  Machine Learning Models

Three classification algorithms were trained and compared.

## Logistic Regression

A linear classifier that learns the relationship between TF-IDF features and emotion labels.

---

## Multinomial Naive Bayes

A probabilistic classifier commonly used for text classification tasks.

---

## Linear Support Vector Machine (SVM)

A powerful linear classifier that performs exceptionally well on high-dimensional text data.

---

#  Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

### Performance Comparison

| Model                   | Accuracy   | Precision  | Recall     | F1-score   |
| ----------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression     | **86.60%** | **86.78%** | **86.60%** | **86.20%** |
| Multinomial Naive Bayes | 63.80%     | 71.79%     | 63.80%     | 54.51%     |
| Linear SVM              | 85.78%     | 86.17%     | 85.78%     | 85.31%     |

---

#  Best Model

Among the three models, **Logistic Regression** achieved the highest overall performance.

Final Results:

* Accuracy: **86.60%**
* Precision: **86.78%**
* Recall: **86.60%**
* F1-score: **86.20%**

Therefore, Logistic Regression was selected as the final model for emotion prediction.

---

#  Project Structure

```
Emotion-Detection-NLP/
│
├── Emotion_Detection.ipynb
├── emotion_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
├── requirements.txt
└── images/
    ├── confusion_matrix.png
    └── model_comparison.png
```

---

#  How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Emotion-Detection-NLP.git
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Load the trained model

```python
import joblib

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
```

### 4. Predict the emotion of new text

```python
text = ["I am so excited because I passed my exam."]

text_vector = vectorizer.transform(text)

prediction = model.predict(text_vector)

label_mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

print(label_mapping[prediction[0]])
```

---

# Future Improvements

Possible future enhancements include:

* Applying text normalization techniques
* Comparing CountVectorizer with TF-IDF
* Using Word2Vec or GloVe embeddings
* Building an LSTM-based emotion detection model
* Fine-tuning a BERT transformer model
* Deploying the model as a web application using Streamlit or Flask


#  Author

**Arouba Samardali**

Data Science & Artificial Intelligence Student

GitHub: https://github.com/arouba05


#  Conclusion

This project demonstrates a complete end-to-end NLP workflow for emotion classification. Starting from raw text data, the project applies preprocessing, TF-IDF feature extraction, and multiple machine learning algorithms to classify emotions effectively.

The comparison of Logistic Regression, Multinomial Naive Bayes, and Linear SVM shows that **Logistic Regression achieved the best overall performance**, making it the final selected model for this task.

This project strengthened practical skills in Natural Language Processing, feature engineering, machine learning model evaluation, and building reproducible NLP pipelines.
