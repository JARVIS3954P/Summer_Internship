from flask import Flask, render_template_string, request, send_file
from prediction_app import predict_lna_performance, get_valid_materials_and_architectures
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64

app = Flask(__name__)

# Model performance metrics (from the notebook results)
MODEL_PERFORMANCE = {
    'Linear Regression': {
        'Noise MAE': 0.4141,
        'Noise R²': 0.8288,
        'Gain MAE': 2.7693,
        'Gain R²': 0.3945
    },
    'Random Forest': {
        'Noise MAE': 0.3811,
        'Noise R²': 0.8359,
        'Gain MAE': 2.2727,
        'Gain R²': 0.5837
    },
    'Gradient Boosting': {
        'Noise MAE': 0.3348,
        'Noise R²': 0.8783,
        'Gain MAE': 2.2761,
        'Gain R²': 0.6092
    }
}

def create_accuracy_chart():
    """Create accuracy comparison charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = list(MODEL_PERFORMANCE.keys())
    noise_r2 = [MODEL_PERFORMANCE[model]['Noise R²'] for model in models]
    gain_r2 = [MODEL_PERFORMANCE[model]['Gain R²'] for model in models]
    
    # Noise R² Chart
    bars1 = ax1.bar(models, noise_r2, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Model Accuracy - Noise Figure Prediction', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, noise_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Gain R² Chart
    bars2 = ax2.bar(models, gain_r2, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Model Accuracy - Gain Prediction', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, gain_r2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

FORM_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LNA Performance Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f4; margin: 0; padding: 20px; }
        .main-container { max-width: 1200px; margin: 0 auto; }
        .prediction-section { background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px #ccc; margin-bottom: 30px; }
        .accuracy-section { background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px #ccc; }
        h2 { text-align: center; color: #333; margin-bottom: 30px; }
        h3 { color: #1976d2; margin-bottom: 20px; }
        .form-row { display: flex; gap: 20px; margin-bottom: 15px; }
        .form-group { flex: 1; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input, select { width: 100%; padding: 10px; border-radius: 4px; border: 1px solid #ddd; font-size: 14px; }
        input:focus, select:focus { outline: none; border-color: #1976d2; box-shadow: 0 0 5px rgba(25, 118, 210, 0.3); }
        .result { margin-top: 25px; padding: 20px; background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border-radius: 8px; border-left: 4px solid #4caf50; }
        .error { margin-top: 25px; padding: 20px; background: linear-gradient(135deg, #ffebee, #ffcdd2); border-radius: 8px; border-left: 4px solid #f44336; color: #c00; }
        button { margin-top: 20px; width: 100%; padding: 12px; background: linear-gradient(135deg, #1976d2, #1565c0); color: #fff; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; transition: all 0.3s ease; }
        button:hover { background: linear-gradient(135deg, #1565c0, #0d47a1); transform: translateY(-2px); box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3); }
        .charts-container { display: flex; gap: 20px; margin-top: 20px; }
        .chart { flex: 1; text-align: center; }
        .chart img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .metrics-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .metrics-table th, .metrics-table td { padding: 12px; text-align: center; border: 1px solid #ddd; }
        .metrics-table th { background: #f5f5f5; font-weight: bold; color: #333; }
        .metrics-table tr:nth-child(even) { background: #f9f9f9; }
        .best-model { background: #e8f5e9 !important; font-weight: bold; }
        .model-info { background: #e3f2fd; padding: 20px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #2196f3; }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="prediction-section">
            <h2>🎯 LNA Performance Predictor</h2>
            <form method="post">
                <div class="form-row">
                    <div class="form-group">
                        <label for="material">Material:</label>
                        <select name="material" id="material" required>
                            {% for m in materials %}
                            <option value="{{m}}" {% if m == material %}selected{% endif %}>{{m}}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="architecture">Architecture:</label>
                        <select name="architecture" id="architecture" required>
                            {% for a in architectures %}
                            <option value="{{a}}" {% if a == architecture %}selected{% endif %}>{{a}}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="frequency">Frequency (GHz):</label>
                        <input type="number" step="any" name="frequency" id="frequency" value="{{frequency}}" required>
                    </div>
                    <div class="form-group">
                        <label for="bandwidth">Bandwidth (GHz):</label>
                        <input type="number" step="any" name="bandwidth" id="bandwidth" value="{{bandwidth}}" required>
                    </div>
                </div>
                <button type="submit">🚀 Predict Performance</button>
            </form>
            
            {% if result %}
            <div class="result">
                <h3>📊 Prediction Results</h3>
                <p><strong>Predicted Gain:</strong> <span style="color: #2e7d32; font-size: 18px;">{{result.gain}} dB</span></p>
                <p><strong>Predicted Noise Figure:</strong> <span style="color: #2e7d32; font-size: 18px;">{{result.noise}} dB</span></p>
            </div>
            {% endif %}
            
            {% if error %}
            <div class="error">
                <h3>⚠️ Error</h3>
                <p>{{error}}</p>
            </div>
            {% endif %}
        </div>

        <div class="accuracy-section">
            <h2>📈 Model Accuracy & Performance</h2>
            
            <div class="model-info">
                <h3>🏆 Best Performing Model: Gradient Boosting</h3>
                <p>Our models have been trained on a comprehensive LNA dataset and evaluated using R² score and Mean Absolute Error (MAE). The Gradient Boosting model shows the best performance for both noise figure and gain predictions.</p>
            </div>

            <div class="charts-container">
                <div class="chart">
                    <h3>Noise Figure Prediction Accuracy</h3>
                    <img src="data:image/png;base64,{{accuracy_chart}}" alt="Noise Figure Accuracy Chart">
                </div>
                <div class="chart">
                    <h3>Gain Prediction Accuracy</h3>
                    <img src="data:image/png;base64,{{accuracy_chart}}" alt="Gain Accuracy Chart">
                </div>
            </div>

            <h3>📋 Detailed Performance Metrics</h3>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Noise MAE (dB)</th>
                        <th>Noise R²</th>
                        <th>Gain MAE (dB)</th>
                        <th>Gain R²</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model, metrics in model_performance.items() %}
                    <tr {% if model == 'Gradient Boosting' %}class="best-model"{% endif %}>
                        <td><strong>{{model}}</strong></td>
                        <td>{{"%.4f"|format(metrics['Noise MAE'])}}</td>
                        <td>{{"%.4f"|format(metrics['Noise R²'])}}</td>
                        <td>{{"%.4f"|format(metrics['Gain MAE'])}}</td>
                        <td>{{"%.4f"|format(metrics['Gain R²'])}}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    materials, architectures = get_valid_materials_and_architectures()
    result = None
    error = None
    # Defaults
    form_data = {
        'material': materials[0],
        'frequency': '',
        'bandwidth': '',
        'architecture': architectures[0]
    }
    if request.method == 'POST':
        form_data['material'] = request.form.get('material', materials[0])
        form_data['frequency'] = request.form.get('frequency', '')
        form_data['bandwidth'] = request.form.get('bandwidth', '')
        form_data['architecture'] = request.form.get('architecture', architectures[0])
        try:
            freq = float(form_data['frequency'])
            bw = float(form_data['bandwidth'])
            gain, noise, err = predict_lna_performance(
                form_data['material'], freq, bw, form_data['architecture']
            )
            if err:
                error = err
            else:
                result = {'gain': f"{gain:.2f}", 'noise': f"{noise:.2f}"}
        except Exception as e:
            error = f"Invalid input: {e}"
    
    # Generate accuracy chart
    accuracy_chart = create_accuracy_chart()
    
    return render_template_string(
        FORM_TEMPLATE,
        materials=materials,
        architectures=architectures,
        result=result,
        error=error,
        accuracy_chart=accuracy_chart,
        model_performance=MODEL_PERFORMANCE,
        **form_data
    )



if __name__ == '__main__':
    app.run(debug=True) 