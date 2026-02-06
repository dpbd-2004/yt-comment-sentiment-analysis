# app.py
import os
import io
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import mlflow
from mlflow.tracking import MlflowClient
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Use non-interactive backend
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

# --- NLTK Setup ---
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except Exception as e:
    print(f"NLTK Download Warning: {e}")

# --- Utility Functions ---

def preprocess_comment(comment):
    try:
        if not comment or not isinstance(comment, str):
            return ""
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        return comment

def load_model_and_vectorizer(model_name, model_version, vectorizer_filename):
    """Load model from MLflow and vectorizer from one level above the script."""
    mlflow.set_tracking_uri("http://ec2-13-48-27-69.eu-north-1.compute.amazonaws.com:5000")
    
    model_uri = f"models:/{model_name}/{model_version}"
    
    # --- LOGIC TO FIND FILE IN ROOT ---
    # Goes from /flask_app/app.py -> /tfidf_vectorizer.pkl
    base_path = os.path.dirname(os.path.abspath(__file__)) # This is the flask_app folder
    root_path = os.path.dirname(base_path)                 # This is the project root
    vectorizer_path = os.path.join(root_path, vectorizer_filename)
    
    print(f"[*] Looking for vectorizer at: {vectorizer_path}")
    
    model = mlflow.pyfunc.load_model(model_uri)
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"❌ Could not find {vectorizer_filename} in the root folder!")
        
    vectorizer = joblib.load(vectorizer_path)
    print(f"✅ Successfully loaded model and vectorizer.")
    return model, vectorizer

# --- Initialize ---
try:
    model, vectorizer = load_model_and_vectorizer(
        "yt_chrome_plugin_model", 
        "1", 
        "tfidf_vectorizer.pkl"
    )
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    model, vectorizer = None, None

# --- Routes ---

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    if not model or not vectorizer:
        return jsonify({"error": "Model/Vectorizer not loaded"}), 500

    data = request.json
    comments_data = data.get('comments', [])
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item.get('text', '') for item in comments_data]
        timestamps = [item.get('timestamp', '') for item in comments_data]

        # 1. Preprocess
        preprocessed = [preprocess_comment(c) for c in comments]
        
        # 2. Vectorize
        transformed = vectorizer.transform(preprocessed)
        
        # 3. FIX: Convert to DataFrame with feature names to satisfy MLflow schema
        # get_feature_names_out() gives the list of words (columns) in order
        feature_names = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(transformed.toarray(), columns=feature_names)
        
        # 4. Predict using the DataFrame
        predictions = model.predict(df_input).tolist()
        
        response = [
            {"comment": c, "sentiment": str(s), "timestamp": t} 
            for c, s, t in zip(comments, predictions, timestamps)
        ]
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc() 
        return jsonify({"error": f"Schema Mismatch or Prediction Error: {str(e)}"}), 500

# (Keep your other routes like /generate_chart the same)
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        counts = data.get('sentiment_counts', {})
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [int(counts.get('1', 0)), int(counts.get('0', 0)), int(counts.get('-1', 0))]
        if sum(sizes) == 0: return jsonify({"error": "No data"}), 400
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=['#36A2EB', '#C9CBCF', '#FF6384'], autopct='%1.1f%%', startangle=140)
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments', [])
        text = ' '.join([preprocess_comment(c) for c in comments])
        wc = WordCloud(width=800, height=400, background_color='black', colormap='Blues').generate(text)
        img_io = io.BytesIO()
        wc.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        df = pd.DataFrame(data.get('sentiment_data', []))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        # Handle case where resample 'M' might fail on small datasets
        monthly = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        for val in [-1, 0, 1]:
            if val not in monthly.columns: monthly[val] = 0
        monthly = monthly[[-1, 0, 1]]
        pct = (monthly.T / monthly.sum(axis=1)).T * 100
        plt.figure(figsize=(10, 5))
        for val, color, lbl in [(-1, 'red', 'Negative'), (0, 'gray', 'Neutral'), (1, 'green', 'Positive')]:
            plt.plot(pct.index, pct[val], marker='o', label=lbl, color=color)
        plt.title('Monthly Sentiment Trend')
        plt.legend()
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)