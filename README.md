# 🎯 Sentiment Analysis — Machine Learning Project

A production-quality, end-to-end sentiment analysis system built with **Python**, **scikit-learn**, and **Streamlit**. This project classifies Twitter tweets into four sentiment categories: **Positive**, **Negative**, **Neutral**, and **Irrelevant**.

---

## 📋 Problem Statement

Social media platforms generate massive volumes of text data every day. Understanding the sentiment behind these messages is essential for businesses, researchers, and policymakers. Manual classification is infeasible at scale — hence the need for automated **NLP-based sentiment analysis**.

## 🎯 Objective

Build a robust, multi-model sentiment classification system that:

- Accurately classifies tweets into **4 sentiment classes**
- Compares multiple ML algorithms using rigorous cross-validation
- Provides an **interactive web interface** for real-time predictions

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏠 Home Dashboard | Project overview with key metrics |
| 📤 Dataset Upload | Upload custom CSV or use the default dataset |
| 🧠 Model Training | Train 3 models with GridSearchCV + 5-fold CV |
| 🔮 Prediction | Real-time sentiment prediction with confidence |
| 📊 Performance | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 📈 Visualizations | Word clouds, bar charts, pie charts, comparisons |
| ℹ️ About | Project architecture and documentation |

### Machine Learning Pipeline

- **Preprocessing**: Lowercasing, URL/mention/hashtag removal, lemmatization
- **Vectorization**: TF-IDF (uni + bigrams, 20K features)
- **Models**: Naive Bayes · Logistic Regression · Linear SVM
- **Tuning**: GridSearchCV with Stratified K-Fold cross-validation
- **Selection**: Automatic best model selection by F1 score

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Frontend | Streamlit |
| Machine Learning | scikit-learn |
| NLP | NLTK |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Serialization | Joblib |
| Testing | pytest |

---

## 📁 Project Structure

```
sentiment-analysis-project/
├── app.py                          # Streamlit application (entry point)
├── config.py                       # Centralized configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
│
├── data/
│   ├── raw/                        # Original dataset
│   │   └── twitter_training.csv
│   └── processed/                  # Cleaned/transformed data
│
├── models/                         # Trained model artifacts (.pkl)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── data_loader.py          # CSV loading utilities
│   │   └── data_preprocessing.py   # Text cleaning & NLP pipeline
│   ├── features/
│   │   └── feature_engineering.py  # TF-IDF, label encoding, splitting
│   ├── models/
│   │   ├── train_model.py          # Pipeline training & GridSearchCV
│   │   ├── evaluate_model.py       # Metrics computation
│   │   └── predict_model.py        # Inference module
│   ├── visualization/
│   │   └── plots.py                # All chart generators
│   └── utils/
│       ├── logger.py               # Logging configuration
│       └── helpers.py              # Reusable utility functions
│
├── tests/
│   ├── test_preprocessing.py       # Preprocessing unit tests
│   └── test_model.py               # Model/feature tests
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis
│
├── assets/images/                  # Static assets
└── logs/                           # Application logs
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-project.git
   cd sentiment-analysis-project
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (happens automatically on first run)

---

## ▶️ How to Run

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Workflow

1. Navigate to **Upload Dataset** → load the default dataset or upload your own CSV
2. Navigate to **Train Model** → click **Start Training** to train all 3 models
3. Navigate to **Predict Sentiment** → enter text and get real-time predictions
4. Explore **Model Performance** and **Visualizations** for detailed analysis

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Dataset

The project uses the **Twitter Sentiment Dataset** containing ~74,000 tweets labeled as:

| Sentiment | Description |
|-----------|-------------|
| Positive | Tweets expressing positive sentiment |
| Negative | Tweets expressing negative sentiment |
| Neutral | Factual or neutral tweets |
| Irrelevant | Off-topic tweets |

---

## 📄 License

This project is developed as a **Final Year College Project** for academic and portfolio purposes.

---

## 👨‍💻 Author

Built with ❤️ as a Final Year Project.
