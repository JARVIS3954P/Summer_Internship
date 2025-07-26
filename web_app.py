from flask import Flask, render_template_string, request
from prediction_app import predict_lna_performance, get_valid_materials_and_architectures

app = Flask(__name__)

FORM_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LNA Performance Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f4; }
        .container { max-width: 500px; margin: 40px auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px #ccc; }
        h2 { text-align: center; }
        label { display: block; margin-top: 15px; }
        input, select { width: 100%; padding: 8px; margin-top: 5px; border-radius: 4px; border: 1px solid #ccc; }
        .result { margin-top: 25px; padding: 15px; background: #e8f5e9; border-radius: 6px; }
        .error { margin-top: 25px; padding: 15px; background: #ffebee; border-radius: 6px; color: #c00; }
        button { margin-top: 20px; width: 100%; padding: 10px; background: #1976d2; color: #fff; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background: #1565c0; }
    </style>
</head>
<body>
    <div class="container">
        <h2>LNA Performance Predictor</h2>
        <form method="post">
            <label for="material">Material:</label>
            <select name="material" id="material" required>
                {% for m in materials %}
                <option value="{{m}}" {% if m == material %}selected{% endif %}>{{m}}</option>
                {% endfor %}
            </select>
            <label for="frequency">Frequency (GHz):</label>
            <input type="number" step="any" name="frequency" id="frequency" value="{{frequency}}" required>
            <label for="bandwidth">Bandwidth (GHz):</label>
            <input type="number" step="any" name="bandwidth" id="bandwidth" value="{{bandwidth}}" required>
            <label for="architecture">Architecture:</label>
            <select name="architecture" id="architecture" required>
                {% for a in architectures %}
                <option value="{{a}}" {% if a == architecture %}selected{% endif %}>{{a}}</option>
                {% endfor %}
            </select>
            <button type="submit">Predict</button>
        </form>
        {% if result %}
        <div class="result">
            <strong>Predicted Gain:</strong> {{result.gain}} dB<br>
            <strong>Predicted Noise Figure:</strong> {{result.noise}} dB
        </div>
        {% endif %}
        {% if error %}
        <div class="error">{{error}}</div>
        {% endif %}
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
    return render_template_string(
        FORM_TEMPLATE,
        materials=materials,
        architectures=architectures,
        result=result,
        error=error,
        **form_data
    )

if __name__ == '__main__':
    app.run(debug=True) 