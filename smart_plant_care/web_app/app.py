import os
import pandas as pd
import json
from flask import Flask, render_template, jsonify, request
import random

app = Flask(__name__)

# --- DATA LOADING ---
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'rollouts.csv')

def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    
    data_dict = {}
    policies = df['policy'].unique()
    order = {'fixed': 0, 'threshold': 1, 'ppo': 2}
    policies = sorted(policies, key=lambda x: order.get(x, 99))
    DEMO_LEN = 1000 # Longer for the app
    
    for policy in policies:
        policy_df = df[df['policy'] == policy]
        subset = policy_df.iloc[:DEMO_LEN]
        data_dict[policy] = {
            't': subset['t'].tolist(),
            'health': subset['plant_health'].round(1).tolist(),
            'moisture': subset['soil_moisture'].round(2).tolist(),
            'temperature': subset['temperature'].round(1).tolist(),
            'light': subset['light_level'].round(1).tolist(),
            'water': subset['water_amount'].tolist(),
            'lamp': subset['lamp_on'].tolist(),
            'hour': subset['hour_of_day'].tolist()
        }
    return data_dict

CACHED_DATA = load_data()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    if CACHED_DATA is None:
        return jsonify({"error": "No data found. Please run evaluation first."}), 404
    return jsonify(CACHED_DATA)

@app.route('/api/action/water', methods=['POST'])
def action_water():
    # Simulate processing time
    amount = random.randint(20, 100)
    return jsonify({
        "status": "success",
        "message": f"Manual injection: {amount}ml",
        "amount": amount,
        "new_moisture": 0.65 # Mock update
    })

@app.route('/api/action/light', methods=['POST'])
def action_light():
    return jsonify({
        "status": "success",
        "message": "Manual light override: ON",
        "duration": "10s"
    })

@app.route('/api/config/update', methods=['POST'])
def update_config():
    data = request.json
    print(f"Received config update: {data}")
    return jsonify({
        "status": "success",
        "message": "Configuration updated. Retraining initiated (Simulation)."
    })

if __name__ == '__main__':
    print("🌿 LeafMind Server Starting...")
    print("🚀 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)

