# LNA Performance Predictor

## Introduction

This project is a machine learning application designed to predict the performance of Low-Noise Amplifiers (LNAs) based on their material, architecture, and operating parameters. By leveraging a dataset of existing LNA specifications, this tool can estimate key performance metrics such as **Gain (dB)** and **Noise Figure (dB)**.

The project encompasses the entire machine learning pipeline, from exploratory data analysis (EDA) and data preprocessing to model training, evaluation, and deployment as a user-friendly web application built with Flask.

## Features

*   **Performance Prediction:** Predicts LNA Gain and Noise Figure based on user inputs for material, architecture, frequency, and bandwidth.
*   **Web-Based Interface:** A simple and intuitive web application built with Flask allows for easy interaction and prediction.
*   **Model Comparison:** The underlying model training process evaluates and compares three different regression models (Linear Regression, Random Forest, and Gradient Boosting) to select the best performer.
*   **Data Visualization:** The web app includes charts that visualize the accuracy (R² score) of the different models for both Gain and Noise Figure predictions.
*   **Detailed Performance Metrics:** The application also displays a table of performance metrics (MAE and R²) for each model, providing transparency into the model selection process.

## Technologies Used

*   **Programming Language:** Python
*   **Machine Learning Libraries:** Scikit-learn, Pandas, NumPy
*   **Web Framework:** Flask
*   **Data Visualization:** Matplotlib
*   **Development Environment:** JupyterLab

## Setup and Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LNA-Performance-Predictor.git
    cd LNA-Performance-Predictor
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the setup is complete, you can run the web application:

1.  **Start the Flask server:**
    ```bash
    python web_app.py
    ```

2.  **Open your web browser** and navigate to `http://127.0.0.1:5000`.

3.  **Enter the LNA parameters** into the form:
    *   **Material:** Select from the dropdown (e.g., GaN, GaAs).
    *   **Architecture:** Select the LNA architecture.
    *   **Frequency (GHz):** Enter the operating frequency.
    *   **Bandwidth (GHz):** Enter the required bandwidth.

4.  **Click "Predict Performance"** to see the estimated Gain and Noise Figure.

## Model Training and Evaluation

The machine learning model was trained using the `model.ipynb` notebook. The process included:

1.  **Exploratory Data Analysis (EDA):** The initial dataset (`dataset.csv`) was analyzed in `eda.ipynb` to understand feature distributions, correlations, and handle missing values.
2.  **Data Preprocessing:**
    *   Categorical features like `material` and `lna_arch` were encoded using Label Encoding and One-Hot Encoding, respectively.
    *   Numerical features were scaled using `StandardScaler` to ensure consistent performance across models.
    *   The cleaned and preprocessed dataset was saved as `processed_lna_dataset.csv`.
3.  **Model Training:**
    *   The preprocessed data was split into training and testing sets.
    *   Three regression models were trained to predict both Gain and Noise Figure:
        *   Linear Regression
        *   Random Forest Regressor
        *   Gradient Boosting Regressor
4.  **Model Evaluation:**
    *   The models were evaluated on the test set using **Mean Absolute Error (MAE)** and **R² Score**.
    *   **Gradient Boosting** was selected as the champion model due to its superior performance on both target variables (Noise R²: 0.8783, Gain R²: 0.6092).
5.  **Model Serialization:**
    *   The trained Gradient Boosting models for gain (`gb_model_gain.pkl`) and noise (`gb_model_noise.pkl`), along with the scaler and label encoder, were saved using `joblib` for use in the prediction application.

## Files in this Repository

*   `dataset.csv`: The raw dataset containing LNA specifications.
*   `eda.ipynb`: Jupyter notebook for exploratory data analysis and data cleaning.
*   `model.ipynb`: Jupyter notebook detailing the model training, evaluation, and comparison process.
*   `prediction_app.py`: A Python module containing the function to load the trained models and make predictions.
*   `web_app.py`: The Flask application that provides the web interface.
*   `requirements.txt`: A list of Python packages required to run the project.
*   `*.pkl`: Serialized `joblib` files for the trained models, scaler, and label encoder. (Will appear once you train the model)
*   `processed_lna_dataset.csv`: The cleaned and preprocessed dataset used for model training.
