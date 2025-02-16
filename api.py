from flask import Flask, request, jsonify
import joblib
import pandas as pd
from geolocation_model import GeolocationModel
from nearby_services_model import NearbyServicesModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load models and data
symptom_model, symptom_vectorizer = joblib.load('naive_bayes_modelS.pkl.gz'), joblib.load('tfidf_vectorizerS.pkl')
remedies_df = pd.read_csv('REMEDIES.csv')
otc_df = pd.read_csv('Book1__OTC.csv')

geolocation_model = GeolocationModel()
services_model = NearbyServicesModel(api_key="vTUXwsGD6SPVv_cpCDTdI_TPs4IPwdxSbyB_yMjy3W4")

@app.route('/')
def home():
    return "Welcome to the Health Assistant API!", 200

@app.route('/predict_symptoms', methods=['POST', 'GET'])
def predict_symptoms():
    if request.method == 'GET':
        return jsonify({"error": "Use POST method"}), 405

    data = request.json
    user_input = data.get("symptoms")
    if not user_input:
        return jsonify({"error": "No symptoms provided."}), 400

    user_input_vectorized = symptom_vectorizer.transform([user_input])
    matched_symptoms = symptom_model.predict(user_input_vectorized)
    
    if matched_symptoms:
        detected_symptoms = [symptom for symptom in matched_symptoms if isinstance(symptom, str)]
        disease = detected_symptoms[0] if detected_symptoms else "Unknown Disease"
        return jsonify({"detected_symptoms": list(set(detected_symptoms)), "predicted_disease": disease})

    return jsonify({"error": "No matching symptoms found."}), 404


@app.route('/get_remedies', methods=['GET'])
def get_remedies():
    disease = request.args.get('disease')
    if not disease:
        return jsonify({"error": "No disease provided."}), 400
    
    remedies_row = remedies_df[remedies_df['DISEASE NAME'].str.lower().str.strip() == disease.lower().strip()]
    remedies = remedies_row.iloc[0, 1:7].dropna().tolist() if not remedies_row.empty else []
    
    return jsonify({"remedies": remedies})

@app.route('/get_otc', methods=['GET'])
def get_otc():
    disease = request.args.get('disease')
    if not disease:
        return jsonify({"error": "No disease provided."}), 400
    
    otc_row = otc_df[otc_df['Diseases'].str.lower().str.strip() == disease.lower().strip()]
    otc_medicines = otc_row.iloc[0, 1:5].dropna().tolist() if not otc_row.empty else []
    
    return jsonify({"otc_medicines": otc_medicines})

@app.route('/get_nearby_services', methods=['POST'])
def get_nearby_services():
    data = request.json
    location = data.get("location")
    if not location:
        return jsonify({"error": "No location provided."}), 400
    
    lat, lng = geolocation_model.get_geolocation(location)
    if lat and lng:
        services = services_model.get_nearby_services(lat, lng)
        if not services:
            return jsonify({"error": "No nearby services found."}), 404
        
        formatted_services = [{"name": s["name"], "address": s["address"]} for s in services]
        return jsonify({"nearby_services": formatted_services})
    
    return jsonify({"error": "Could not find location."}), 404

@app.route('/process_input', methods=['POST'])
def process_input():
    data = request.json
    symptoms = data.get("symptoms")
    location = data.get("location")
    if not symptoms or not location:
        return jsonify({"error": "Symptoms and location are required."}), 400
    
    # Predict symptoms
    user_input_vectorized = symptom_vectorizer.transform([symptoms])
    matched_symptoms = symptom_model.predict(user_input_vectorized)
    detected_symptoms = [symptom for symptom in matched_symptoms if isinstance(symptom, str)]
    disease = detected_symptoms[0] if detected_symptoms else "Unknown Disease"
    
    # Get remedies
    remedies_row = remedies_df[remedies_df['DISEASE NAME'].str.lower().str.strip() == disease.lower().strip()]
    remedies = remedies_row.iloc[0, 1:7].dropna().tolist() if not remedies_row.empty else []
    
    # Get OTC medicines
    otc_row = otc_df[otc_df['Diseases'].str.lower().str.strip() == disease.lower().strip()]
    otc_medicines = otc_row.iloc[0, 1:5].dropna().tolist() if not otc_row.empty else []
    
    # Get nearby services
    lat, lng = geolocation_model.get_geolocation(location)
    services = services_model.get_nearby_services(lat, lng) if lat and lng else []
    
    # ✅ Fix: Handle None case properly
    formatted_services = [{"name": s["name"], "address": s["address"]} for s in services] if services else []
    
    return jsonify({
        "detected_symptoms": list(set(detected_symptoms)),
        "predicted_disease": disease,
        "remedies": remedies,
        "otc_medicines": otc_medicines,
        "nearby_services": formatted_services
    })

if __name__ == '__main__':
    app.run(debug=True)