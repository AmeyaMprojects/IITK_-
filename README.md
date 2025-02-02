# Bot Detection Model

This project is a web application for training and predicting bot detection using various machine learning models. The frontend is built with React, and the backend is built with Flask. The application allows users to upload a CSV file for training the model and then use the trained model to predict whether a user is a bot or not.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js and npm
- `pip` for Python package management

### Backend Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/bot-detection.git
    cd bot-detection/backend
    ```

2. Install the required Python packages:
    ```sh
    sudo apt update
    sudo apt install python3-flask python3-pandas python3-sklearn python3-matplotlib python3-numpy gunicorn
    pip install Flask-CORS xgboost
    ```

3. Run the Flask application:
    ```sh
    cd backend
    python app.py
    ```

### Frontend Setup


1. Go to the root directory of the project:
    ```sh
    cd ..
    ```

2. Install the required npm packages:
    ```sh
    npm install
    ```

3. Start the React application:
    ```sh
    npm run dev
    ```

## Usage

1. Open your browser and navigate to `http://localhost:80`.
2. Upload a CSV file containing the training data.(the csv file should contain the following columns: default_profile, default_profile_image, favourites_count, followers_count, friends_count, screen_name, statuses_count, verified, geo_enabled, average_tweets_per_day, account_age_days) the csv file that we used is given in the zip file.
3. Click on the "Train Model" button to train the model.
4. After the model is trained, enter user information to predict whether the user is a bot or not.

## API Endpoints

### `/train` (POST)

- Description: Train the model with the uploaded CSV file.
- Request: Multipart form data with a CSV file.
- Response: JSON object containing the training results and performance metrics.

### `/predict` (POST)

- Description: Predict whether a user is a bot or not using the trained model.
- Request: JSON object containing user information.
- Response: JSON object containing the prediction result.

## Errors

- if facing any erros related to network or CORS please check if the backedn server is running.



