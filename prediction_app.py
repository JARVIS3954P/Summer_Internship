# prediction_app.py

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

def predict_lna_performance(material, frequency, bandwidth, architecture):
    """
    Loads trained models and pre-processing objects to predict LNA performance.
    Returns (predicted_gain, predicted_noise, error_message)
    """
    try:
        # Load all required saved objects
        model_noise = joblib.load('gb_model_noise.pkl')
        model_gain = joblib.load('gb_model_gain.pkl')
        scaler = joblib.load('scaler.pkl')
        le_material = joblib.load('label_encoder.pkl')
        
        feature_columns = [
            'material', 'gLen_µm', 'freq_Ghz', 'bandwidth_GHz', 'lna_arch_3stage',
            'lna_arch_3stageCS', 'lna_arch_4stage', 'lna_arch_5stage', 'lna_arch_6stage',
            'lna_arch_Cascode', 'lna_arch_Distributed', 'lna_arch_Foldedcascode',
            'lna_arch_PowerAmplifier', 'lna_arch_Singlestage', 'lna_arch_UWB', 'lna_arch_Unknown'
        ]
    except FileNotFoundError as e:
        return None, None, f"Error loading a required .pkl file: {e}"

    # Create the input DataFrame
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0, :] = 0

    # **This now works because the loaded encoder expects uppercase**
    try:
        input_df['material'] = le_material.transform([material.upper()])[0]
    except ValueError:
        return None, None, f"Warning: Material '{material}' not recognized. Known materials are {le_material.classes_}."

    # One-Hot Encode architecture
    arch_col = 'lna_arch_' + architecture
    if arch_col in input_df.columns:
        input_df[arch_col] = 1
    else:
        return None, None, f"Warning: Architecture '{architecture}' is not recognized."

    # Prepare and scale numeric features
    scaler_features = ['gLen_µm', 'freq_Ghz', 'bandwidth_GHz']
    scale_temp_df = pd.DataFrame([[0, frequency, bandwidth]], columns=scaler_features)
    scaled_values = scaler.transform(scale_temp_df)
    
    input_df['gLen_µm'] = scaled_values[0, 0]
    input_df['freq_Ghz'] = scaled_values[0, 1]
    input_df['bandwidth_GHz'] = scaled_values[0, 2]

    # Make predictions
    predicted_gain = model_gain.predict(input_df)[0]
    predicted_noise = model_noise.predict(input_df)[0]
    
    return predicted_gain, predicted_noise, None


def get_valid_materials_and_architectures():
    # Materials: Based on demo and likely dataset, but ideally should be loaded from label_encoder
    materials = ['GaN', 'GaAs']
    # Architectures: Based on feature columns in the model
    architectures = [
        '3stage', '3stageCS', '4stage', '5stage', '6stage',
        'Cascode', 'Distributed', 'Foldedcascode', 'PowerAmplifier',
        'Singlestage', 'UWB', 'Unknown'
    ]
    return materials, architectures


# --- Main execution block ---
# if __name__ == '__main__':
#     print("LNA Performance Prediction Application")
#     print("="*40)
#     predict_lna_performance(material='GaN', frequency=94, bandwidth=8, architecture='4stage')
#     predict_lna_performance(material='GaAs', frequency=5.8, bandwidth=1, architecture='3stage')
#     predict_lna_performance(material='GaN', frequency=24, bandwidth=4, architecture='Unknown')