
import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, numpy as np, os

app = Flask(__name__)
CORS(app)  # allow the HTML file to call this server from any origin

BASE = os.path.dirname(os.path.abspath(__file__))

# Load models once at startup
models = {
    'knn': joblib.load(os.path.join(BASE, 'model_knn.pkl')),
    'dt':  joblib.load(os.path.join(BASE, 'model_dt.pkl')),
    'rf':  joblib.load(os.path.join(BASE, 'model_rf.pkl')),
    'xgb': joblib.load(os.path.join(BASE, 'model_xgb.pkl')),
}
scaler   = joblib.load(os.path.join(BASE, 'scaler.pkl'))
meta     = json.load(open(os.path.join(BASE, 'model_meta.json')))
features = meta['feature_order']

@app.route('/meta', methods=['GET'])
def get_meta():
    return jsonify(meta)

@app.route('/predict', methods=['POST'])
def predict():
    data       = request.json                        # { model_id, passenger }
    model_id   = data['model_id']
    passenger  = data['passenger']                   # dict of feature values
    model_info = next(m for m in meta['models'] if m['id'] == model_id)

    # Build feature vector in the correct order
    X = np.array([[passenger[f] for f in features]], dtype=float)

    if model_info['needs_scale']:
        X = scaler.transform(X)

    prediction = int(models[model_id].predict(X)[0])
    proba_arr  = models[model_id].predict_proba(X)[0]
    confidence = float(max(proba_arr))

    return jsonify({
        'model_id':   model_id,
        'survived':   prediction,
        'confidence': round(confidence * 100, 1),
        'prob_0':     round(float(proba_arr[0]) * 100, 1),
        'prob_1':     round(float(proba_arr[1]) * 100, 1),
    })

if __name__ == '__main__':
    print('\n  Titanic Model Server running at http://localhost:5050')
    print('   Open titanic_demo.html in your browser.\n')
    app.run(port=5050, debug=False)
