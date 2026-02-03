import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise

def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer: %s', e)
        raise

def main():
    try:
        root_dir = get_root_directory()
        
        # Load model and vectorizer
        model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
        vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

        # Load test data
        test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

        # Prepare test data
        X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
        y_test = test_data['category'].values

        # Predict
        y_pred = model.predict(X_test_tfidf)

        # ---------------- TERMINAL OUTPUT ----------------
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # 1. Classification Report
        print("\nClassification Report:")
        # We set output_dict=False to get the human-readable table string
        print(classification_report(y_test, y_pred))

        # 2. Confusion Matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("\n" + "="*50)

        

    except Exception as e:
        logger.error(f"Failed to complete model evaluation: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()