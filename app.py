import os
import json
import time
import numpy as np
import joblib
import gdown
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, 'static'),
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_url_path='/static'
)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

print(f"[OK] BASE_DIR: {BASE_DIR}")
print(f"[OK] STATIC_DIR: {os.path.join(BASE_DIR, 'static')}")
print(f"[OK] TEMPLATE_DIR: {os.path.join(BASE_DIR, 'templates')}")

model = None
scaler = None
label_encoder = None
feature_columns = None


def load_ml_artifacts():
    global model, scaler, label_encoder, feature_columns
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'ann_student_status.keras')

    # Force delete existing model file to ensure it's not an LFS pointer
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            print(f"Removed existing model file at {model_path}")
        except OSError as e:
            print(f"Error removing file {model_path}: {e}")

    # Always attempt to download the model on startup using gdown
    print("Attempting to download model...")
    model_url = "https://drive.google.com/uc?id=1EFn4QlEe9RGUZDZi4xRIx_TFKcxNBWOP"
    try:
        gdown.download(model_url, model_path, quiet=False)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(f"Could not download the model file: {e}")

    model = load_model(model_path)
    scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
    feature_columns = joblib.load(os.path.join(base_dir, 'feature_columns.pkl'))
    
    print("[OK] Model artifacts loaded successfully")


@app.route('/')
def index():
    return render_template('index.html', static_version=int(time.time()))


@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), filename, max_age=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_values = []
        for col in feature_columns:
            val = data.get(col)
            if val is None:
                return jsonify({'error': f'Missing feature: {col}'}), 400
            try:
                input_values.append(float(val))
            except ValueError:
                return jsonify({'error': f'Invalid value for {col}'}), 400
        
        X_input = np.array([input_values])
        X_scaled = scaler.transform(X_input)
        
        pred_prob = model.predict(X_scaled, verbose=0)
        pred_class_idx = np.argmax(pred_prob[0])
        pred_class = label_encoder.classes_[pred_class_idx]
        
        probabilities = {
            label_encoder.classes_[i]: float(pred_prob[0][i]) * 100 
            for i in range(len(label_encoder.classes_))
        }
        
        return jsonify({
            'prediction': pred_class,
            'probabilities': probabilities,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/_debug')
def debug_info():
    static_dir = os.path.join(BASE_DIR, 'static')
    template_dir = os.path.join(BASE_DIR, 'templates')
    try:
        static_files = sorted(os.listdir(static_dir))
    except Exception as e:
        static_files = [f"ERROR: {e}"]

    return jsonify({
        'base_dir': BASE_DIR,
        'static_dir': static_dir,
        'template_dir': template_dir,
        'static_files': static_files,
        'server_time': int(time.time())
    })


@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Load artifacts on application startup
load_ml_artifacts()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
