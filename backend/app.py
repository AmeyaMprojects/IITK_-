import logging
import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all requests (for Vercel frontend)

# Global model and scaler
model = None
scaler = None

# Function to calculate popularity metric
def popularity_metric(friends_count: int, followers_count: int):
    return np.round(np.log(1 + friends_count) * np.log(1 + followers_count), 3)

# Function to process and train models
def train_model(df):
    global model, scaler

    logger.info("Starting model training...")

    # Preprocess dataset
    df['account_type'] = df['account_type'].replace({'human': 1, 'bot': 0})
    df['default_profile'] = df['default_profile'].astype(int)
    df['default_profile_image'] = df['default_profile_image'].astype(int)
    df['geo_enabled'] = df['geo_enabled'].astype(int)
    df['verified'] = df['verified'].astype(int)

    # Drop unnecessary columns
    df.drop(columns=['location', 'profile_background_image_url', 'profile_image_url', 
                     'screen_name', 'lang', 'id', 'Unnamed: 0', 'created_at', 'description'], inplace=True)

    # Compute popularity metric
    df["popularity"] = df.apply(lambda row: popularity_metric(row["friends_count"], row["followers_count"]), axis=1)

    # Features and target
    target = df['account_type']
    features = df.drop(columns=['account_type'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    model_results = {}

    for name, m in models.items():
        logger.info(f"Training {name} model...")
        m.fit(X_train_scaled, y_train)
        y_pred = m.predict(X_test_scaled)

        # Compute metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, m.predict_proba(X_test_scaled)[:, 1])

        model_results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }

    # Select best model (XGBoost in this case)
    model = models['XGBoost']

    # Generate performance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(model_results.keys())
    metrics = ['precision', 'recall', 'f1', 'auc_roc']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bar_width = 0.2
    index = range(len(model_names))

    for i, metric in enumerate(metrics):
        ax.bar(
            [x + bar_width * i for x in index],
            [model_results[name][metric] for name in model_names],
            bar_width,
            label=metric.capitalize(),
            color=colors[i]
        )

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison: Precision, Recall, F1, AUC-ROC')
    ax.set_xticks([x + bar_width * 1.5 for x in index])
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the plot to a BytesIO object and encode as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    return model_results, img_base64

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        logger.info(f"CSV file uploaded: {file.filename}")

        model_results, plot_img = train_model(df)

        return jsonify({
            'model_results': model_results,
            'roc_auc_plot': plot_img
        })

    except Exception as e:
        logger.error(f"Error during training: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not trained. Please train the model first.'}), 400

        data = request.json
        logger.info(f"Received prediction data: {data}")

        # Convert inputs
        input_data = np.array([[
            int(data['default_profile']),
            int(data['default_profile_image']),
            int(data['favourites_count']),
            int(data['followers_count']),
            int(data['friends_count']),
            int(data['geo_enabled']),
            int(data['statuses_count']),
            int(data['verified']),
            float(data['average_tweets_per_day']),
            int(data['account_age_days']),
            popularity_metric(int(data['friends_count']), int(data['followers_count']))
        ]])

        # Standardize input
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction = int(model.predict(input_data_scaled)[0])

        return jsonify({'prediction': prediction})

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=10000)
