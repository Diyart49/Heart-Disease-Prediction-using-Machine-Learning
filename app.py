from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

def load_or_train_model():
    """Try loading saved model; if version mismatch, retrain from data.csv."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded from model.pkl")
        return model
    except Exception as e:
        print(f"Could not load model.pkl ({e}). Retraining from data.csv...")
        return retrain_model()

def retrain_model():
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('data.csv')
    features = df.drop(['target'], axis=1)
    target = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=0
    )
    model = GradientBoostingClassifier(random_state=1)
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    acc = model.score(X_test, y_test)
    print(f"Model retrained. Test accuracy: {acc:.3f}")
    return model

model = load_or_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]

        input_array = np.array([features])
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0]
        classes = model.classes_.tolist()

        # Target: 1 = no disease, 2 = disease
        has_disease = int(prediction) == 2

        # Get probability for class "2" (disease)
        disease_idx = classes.index(2)
        risk_score = round(float(probability[disease_idx]) * 100, 1)

        return jsonify({
            'prediction': int(prediction),
            'has_disease': has_disease,
            'risk_score': risk_score,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
