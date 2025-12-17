import pandas as pd
import json
import os

def generate_cool_html():
    # 1. Load and Process Data
    print("Loading data...")
    try:
        df = pd.read_csv('data/rollouts.csv')
    except FileNotFoundError:
        print("Error: data/rollouts.csv not found. Please run evaluation first.")
        return

    data_dict = {}
    policies = df['policy'].unique()
    order = {'fixed': 0, 'threshold': 1, 'ppo': 2}
    policies = sorted(policies, key=lambda x: order.get(x, 99))
    DEMO_LEN = 600
    
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

    json_data = json.dumps(data_dict)

    # 2. HTML Template - PRO APP STYLE
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S.P.C. PRO | NEURAL COMMAND</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;500;600;700&family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    
    <style>
        :root {{
            --bg-dark: #050505;
            --sidebar-width: 280px;
            --primary: #00f3ff;
            --accent: #00ff9d;
            --danger: #ff0055;
            --warning: #ffd700;
            --surface: #0f1115;
            --surface-hover: #1a1d24;
            --border: 1px solid #2a2d35;
            --text-main: #e0f7ff;
            --text-dim: #6b7280;
        }}

        * {{ box-sizing: border-box; }}
        
        body {{
            margin: 0; padding: 0;
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            height: 100vh;
            display: flex;
            overflow: hidden;
        }}

        /* SCROLLBAR */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-dark); }}
        ::-webkit-scrollbar-thumb {{ background: #333; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--primary); }}

        /* --- SIDEBAR --- */
        .sidebar {{
            width: var(--sidebar-width);
            background: #0a0a0a;
            border-right: var(--border);
            display: flex; flex-direction: column; padding: 20px;
            z-index: 100;
        }}

        .logo-area {{
            margin-bottom: 40px; display: flex; align-items: center; gap: 12px;
            color: white; font-family: 'Orbitron'; font-weight: 800; font-size: 18px; letter-spacing: 1px;
        }}
        .logo-icon {{
            width: 36px; height: 36px; background: var(--primary); color: black;
            display: flex; align-items: center; justify-content: center; border-radius: 8px;
        }}

        .nav-group {{ margin-bottom: 25px; }}
        .nav-label {{ color: var(--text-dim); font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; padding-left: 10px; letter-spacing: 1px; }}
        
        .nav-item {{
            display: flex; align-items: center; gap: 12px; padding: 12px 15px;
            color: #9ca3af; text-decoration: none; border-radius: 8px;
            transition: all 0.2s; cursor: pointer; margin-bottom: 4px;
            font-family: 'Inter'; font-weight: 500; font-size: 14px;
        }}
        .nav-item:hover {{ background: var(--surface-hover); color: white; }}
        .nav-item.active {{ background: rgba(0, 243, 255, 0.1); color: var(--primary); border: 1px solid rgba(0, 243, 255, 0.2); }}
        .nav-item i {{ font-size: 20px; }}

        /* --- MAIN CONTENT --- */
        .main-content {{
            flex: 1; display: flex; flex-direction: column; position: relative;
            background: radial-gradient(circle at 10% 10%, rgba(0, 243, 255, 0.02) 0%, transparent 40%);
        }}

        .top-bar {{
            height: 64px; border-bottom: var(--border);
            display: flex; align-items: center; justify-content: space-between; padding: 0 30px;
            background: rgba(5, 5, 5, 0.8); backdrop-filter: blur(10px);
        }}

        .view-section {{ display: none; height: 100%; overflow-y: auto; padding: 30px; opacity: 0; transition: opacity 0.3s; }}
        .view-section.active {{ display: block; opacity: 1; }}

        /* --- DASHBOARD GRID --- */
        .dash-grid {{
            display: grid;
            grid-template-columns: 320px 1fr 340px;
            grid-template-rows: 150px 1fr 240px;
            gap: 20px; height: calc(100vh - 124px);
        }}

        .card {{
            background: var(--surface); border: var(--border); border-radius: 12px;
            padding: 20px; position: relative; display: flex; flex-direction: column;
        }}
        .card-header {{
            font-family: 'JetBrains Mono'; font-weight: 700; color: var(--text-dim);
            font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
            margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;
        }}

        /* Metric Cards */
        .metric-val {{ font-size: 36px; font-family: 'Orbitron'; font-weight: 700; color: white; }}
        .metric-unit {{ font-size: 14px; color: var(--text-dim); margin-left: 5px; }}
        .metric-chart {{ height: 40px; width: 100%; margin-top: auto; }}

        /* Neural Viz (Dashboard) */
        .neural-viz {{
            display: flex; justify-content: space-between; align-items: center;
            height: 100%; position: relative;
        }}
        .layer {{ display: flex; flex-direction: column; justify-content: space-around; height: 80%; }}
        .node {{
            width: 12px; height: 12px; border-radius: 50%; background: #333;
            border: 1px solid #555; transition: all 0.1s; position: relative;
            z-index: 2;
        }}
        .node.active {{ background: var(--primary); box-shadow: 0 0 10px var(--primary); border-color: white; transform: scale(1.2); }}
        .connection {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1;
        }}

        /* Plant Viz */
        .viz-card {{
            grid-row: 1 / -1; grid-column: 2 / 3;
            background: #000; border: 1px solid #333;
            display: flex; align-items: center; justify-content: center; position: relative;
            overflow: hidden;
        }}
        /* (Reuse Plant CSS from before but simplified) */
        #plant-wrapper {{ perspective: 1000px; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }}
        #plant-container {{ width: 20px; height: 300px; transform-style: preserve-3d; animation: spinPlant 20s infinite linear; }}
        @keyframes spinPlant {{ 0% {{ transform: rotateY(0deg); }} 100% {{ transform: rotateY(360deg); }} }}
        .stem {{ position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); width: 10px; height: 100%; background: var(--accent); }}
        .leaf {{ position: absolute; width: 70px; height: 35px; background: var(--accent); border-radius: 50px 0; opacity: 0.9; transform-origin: 0 50%; transition: all 0.5s; }}
        
        /* --- PRO CONFIG PAGE --- */
        .config-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        
        .model-card {{
            background: var(--surface); border: var(--border); border-radius: 12px; padding: 20px;
            cursor: pointer; transition: 0.2s; border: 1px solid transparent;
        }}
        .model-card:hover {{ border-color: #444; background: var(--surface-hover); }}
        .model-card.selected {{ border-color: var(--primary); background: rgba(0, 243, 255, 0.05); }}
        
        .badge {{
            padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: bold;
            background: #222; color: #888; display: inline-block; margin-right: 5px;
        }}
        .badge.pro {{ background: var(--primary); color: black; }}

        .slider-group {{ margin-bottom: 20px; }}
        .slider-label {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; color: #ccc; }}
        .slider-val {{ color: var(--primary); font-family: 'JetBrains Mono'; }}
        input[type="range"] {{
            width: 100%; height: 4px; background: #333; border-radius: 2px; appearance: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            appearance: none; width: 16px; height: 16px; background: white; border-radius: 50%; cursor: pointer; transition: 0.2s;
        }}
        input[type="range"]::-webkit-slider-thumb:hover {{ background: var(--primary); transform: scale(1.2); }}

        /* Inference Log */
        .inf-log {{ font-family: 'JetBrains Mono'; font-size: 11px; color: #666; height: 150px; overflow: hidden; position: relative; }}
        .inf-line {{ margin-bottom: 4px; display: flex; gap: 10px; }}
        .inf-time {{ color: #444; }}
        .inf-data {{ color: #aaa; }}
        .inf-tensor {{ color: var(--accent); }}

    </style>
</head>
<body>

    <!-- SIDEBAR -->
    <div class="sidebar">
        <div class="logo-area">
            <div class="logo-icon"><i class="material-icons">hub</i></div>
            <div>
                <div>NEURAL.PLANT</div>
                <div style="font-size: 10px; color: var(--text-dim); font-weight: 400;">PRO EDITION v2.4</div>
            </div>
        </div>

        <div class="nav-group">
            <div class="nav-label">Platform</div>
            <div class="nav-item active" onclick="nav('dashboard', this)"><i class="material-icons">dashboard</i>Dashboard</div>
            <div class="nav-item" onclick="nav('analytics', this)"><i class="material-icons">analytics</i>Analytics</div>
        </div>

        <div class="nav-group">
            <div class="nav-label">Intelligence</div>
            <div class="nav-item" onclick="nav('config', this)"><i class="material-icons">tune</i>Model Config</div>
            <div class="nav-item" onclick="nav('training', this)"><i class="material-icons">model_training</i>Training</div>
        </div>
        
        <div style="margin-top: auto; border: 1px solid #333; border-radius: 8px; padding: 15px; background: #0f0f0f;">
            <div style="font-size: 11px; color: #888; margin-bottom: 5px;">GPU STATUS</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; color: white;">RTX 5090</span>
                <span style="color: var(--accent); font-size: 10px;">● ACTIVE</span>
            </div>
            <div style="width: 100%; height: 4px; background: #333; margin-top: 8px; border-radius: 2px;">
                <div style="width: 45%; height: 100%; background: var(--primary);"></div>
            </div>
            <div style="font-size: 10px; color: #666; margin-top: 4px;">45% Load • 42°C</div>
        </div>
    </div>

    <!-- MAIN -->
    <div class="main-content">
        <div class="top-bar">
            <div id="page-title" style="font-family: 'Orbitron'; font-weight: 700; font-size: 16px;">MISSION CONTROL</div>
            <div style="display: flex; gap: 15px;">
                <button onclick="togglePlay()" id="playBtn" style="background: var(--primary); color: black; border: none; padding: 8px 20px; border-radius: 6px; font-weight: 700; cursor: pointer;">START STREAM</button>
            </div>
        </div>

        <!-- VIEW: DASHBOARD -->
        <div id="dashboard" class="view-section active">
            <div class="dash-grid">
                
                <!-- Metrics -->
                <div class="card">
                    <div class="card-header">ENVIRONMENT SENSORS <i class="material-icons">sensors</i></div>
                    <div style="display: flex; flex-direction: column; gap: 15px;">
                        <div>
                            <div style="font-size: 12px; color: #888;">SOIL MOISTURE</div>
                            <div style="display: flex; align-items: baseline;"><span class="metric-val" id="val-moisture">0</span><span class="metric-unit">%</span></div>
                            <div style="width:100%; height:4px; background:#333;"><div id="bar-moisture" style="width:0%; height:100%; background:var(--primary);"></div></div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #888;">TEMPERATURE</div>
                            <div style="display: flex; align-items: baseline;"><span class="metric-val" id="val-temp">0</span><span class="metric-unit">°C</span></div>
                            <div style="width:100%; height:4px; background:#333;"><div id="bar-temp" style="width:0%; height:100%; background:#ff9d00;"></div></div>
                        </div>
                    </div>
                </div>

                <!-- Neural Viz -->
                <div class="card">
                    <div class="card-header">
                        LIVE INFERENCE
                        <span style="color: var(--accent); font-size: 10px;">● 12ms</span>
                    </div>
                    <div class="neural-viz" id="neural-net">
                        <!-- Generated by JS -->
                    </div>
                </div>

                <!-- Live Logs -->
                <div class="card">
                    <div class="card-header">DECISION STREAM</div>
                    <div class="inf-log" id="inf-log">
                        <div class="inf-line"><span class="inf-time">00:00:01</span><span class="inf-data">INIT SYSTEM...</span></div>
                    </div>
                </div>

                <!-- Main Viz -->
                <div class="viz-card">
                    <div style="position: absolute; top: 20px; left: 20px; font-family: 'Orbitron'; color: #444; font-size: 40px; font-weight: 900; z-index: 0;">PLANT.OS</div>
                    
                    <div id="plant-wrapper">
                        <div id="plant-container">
                            <div class="stem"></div>
                            <div class="leaf L1" style="bottom: 30%; left: 50%; transform: rotateY(0deg) rotateZ(-45deg);"></div>
                            <div class="leaf R1" style="bottom: 25%; left: 50%; transform: rotateY(180deg) rotateZ(-45deg);"></div>
                            <div class="leaf L2" style="bottom: 50%; left: 50%; transform: rotateY(90deg) rotateZ(-30deg);"></div>
                            <div class="leaf R2" style="bottom: 45%; left: 50%; transform: rotateY(270deg) rotateZ(-30deg);"></div>
                        </div>
                    </div>
                    
                    <!-- Stats Overlay -->
                    <div style="position: absolute; bottom: 30px; width: 100%; text-align: center;">
                        <div style="font-size: 10px; color: #666; letter-spacing: 2px; margin-bottom: 5px;">PREDICTED HEALTH</div>
                        <div id="val-health" style="font-size: 60px; font-weight: 900; color: var(--accent); text-shadow: 0 0 30px rgba(0,255,157,0.3);">98%</div>
                    </div>
                    
                    <!-- Lamp Overlay -->
                    <div id="lamp-overlay" style="position: absolute; top:0; left:0; width:100%; height:100%; background: linear-gradient(to bottom, rgba(255,255,255,0.1), transparent); pointer-events: none; opacity: 0; transition: 0.3s;"></div>
                </div>

                <!-- Bottom Stats -->
                <div class="card" style="grid-column: 1 / 4; grid-template-columns: 1fr 1fr 1fr; display: grid; gap: 20px;">
                     <div>
                         <div class="card-header">WATER USAGE</div>
                         <div style="font-size: 24px; color: var(--primary);" id="stats-water">0.0 L</div>
                     </div>
                     <div>
                         <div class="card-header">ENERGY USAGE</div>
                         <div style="font-size: 24px; color: var(--warning);" id="stats-energy">0.0 kWh</div>
                     </div>
                     <div>
                         <div class="card-header">MODEL CONFIDENCE</div>
                         <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="font-size: 24px; color: white;">99.2%</div>
                            <div style="height: 4px; width: 50px; background: #333;"><div style="width: 99%; height: 100%; background: var(--accent);"></div></div>
                         </div>
                     </div>
                </div>
            </div>
        </div>

        <!-- VIEW: CONFIG -->
        <div id="config" class="view-section">
            <h2 style="font-family: 'Orbitron'; margin-bottom: 30px;">MODEL ARCHITECTURE</h2>
            
            <div class="config-grid" style="margin-bottom: 40px;">
                <div class="model-card selected" onclick="selectModel(this, 'ppo')">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="font-weight: bold; font-family: 'Orbitron';">PPO-RESNET</span>
                        <span class="badge pro">RECOMMENDED</span>
                    </div>
                    <div style="font-size: 12px; color: #888; margin-bottom: 15px; line-height: 1.5;">
                        Proximal Policy Optimization with Residual connections. Best balance of stability and sample efficiency for continuous control.
                    </div>
                    <div style="display: flex; gap: 5px;">
                        <span class="badge">Continuous</span>
                        <span class="badge">On-Policy</span>
                        <span class="badge">Actor-Critic</span>
                    </div>
                </div>

                <div class="model-card" onclick="selectModel(this, 'dqn')">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="font-weight: bold; font-family: 'Orbitron';">DQN-DUELING</span>
                        <span class="badge">LEGACY</span>
                    </div>
                    <div style="font-size: 12px; color: #888; margin-bottom: 15px; line-height: 1.5;">
                        Deep Q-Network with Dueling architecture. Good for discrete action spaces, can be unstable in dynamic environments.
                    </div>
                    <div style="display: flex; gap: 5px;">
                        <span class="badge">Discrete</span>
                        <span class="badge">Off-Policy</span>
                    </div>
                </div>
                
                <div class="model-card" onclick="selectModel(this, 'sac')">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="font-weight: bold; font-family: 'Orbitron';">SAC-ENTROPY</span>
                        <span class="badge pro">EXPERIMENTAL</span>
                    </div>
                    <div style="font-size: 12px; color: #888; margin-bottom: 15px; line-height: 1.5;">
                        Soft Actor-Critic. Maximizes expected reward and entropy. Highest exploration capability but computationally expensive.
                    </div>
                    <div style="display: flex; gap: 5px;">
                        <span class="badge">Max Entropy</span>
                        <span class="badge">Off-Policy</span>
                    </div>
                </div>
            </div>

            <h2 style="font-family: 'Orbitron'; margin-bottom: 30px;">HYPERPARAMETERS</h2>
            <div class="config-grid">
                <div class="card">
                    <div class="card-header">TRAINING DYNAMICS</div>
                    
                    <div class="slider-group">
                        <div class="slider-label"><span>Learning Rate (Alpha)</span> <span class="slider-val">3e-4</span></div>
                        <input type="range" min="1" max="100" value="30">
                    </div>
                    
                    <div class="slider-group">
                        <div class="slider-label"><span>Discount Factor (Gamma)</span> <span class="slider-val">0.99</span></div>
                        <input type="range" min="90" max="99" value="99">
                    </div>

                    <div class="slider-group">
                        <div class="slider-label"><span>GAE Lambda</span> <span class="slider-val">0.95</span></div>
                        <input type="range" min="90" max="99" value="95">
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">POLICY CONSTRAINTS</div>
                    
                    <div class="slider-group">
                        <div class="slider-label"><span>Clip Range (Epsilon)</span> <span class="slider-val">0.2</span></div>
                        <input type="range" min="1" max="5" value="2">
                    </div>
                    
                    <div class="slider-group">
                        <div class="slider-label"><span>Entropy Coefficient</span> <span class="slider-val">0.01</span></div>
                        <input type="range" min="0" max="10" value="1">
                    </div>
                    
                    <div class="slider-group">
                        <div class="slider-label"><span>Max Gradient Norm</span> <span class="slider-val">0.5</span></div>
                        <input type="range" min="1" max="10" value="5">
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 30px; text-align: right;">
                <button style="background: transparent; border: 1px solid #444; color: white; padding: 10px 20px; border-radius: 6px; cursor: pointer; margin-right: 10px;">RESET DEFAULTS</button>
                <button style="background: var(--primary); border: none; color: black; padding: 10px 30px; border-radius: 6px; font-weight: bold; cursor: pointer;">APPLY & RETRAIN</button>
            </div>
        </div>

    </div>

    <script>
        const RAW_DATA = {json_data};
        let step = 0;
        let isPlaying = false;
        let intervalId = null;
        let currentPolicy = 'ppo';
        let totalWater = 0;
        
        // --- NAVIGATION ---
        function nav(id, el) {{
            document.querySelectorAll('.view-section').forEach(v => v.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            el.classList.add('active');
            
            const titles = {{'dashboard': 'MISSION CONTROL', 'config': 'MODEL CONFIGURATION', 'analytics': 'ANALYTICS & METRICS'}};
            document.getElementById('page-title').innerText = titles[id] || 'SYSTEM';
        }}
        
        function selectModel(el, model) {{
            document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
            el.classList.add('selected');
        }}

        // --- NEURAL VIZ GENERATOR ---
        function initNeuralViz() {{
            const container = document.getElementById('neural-net');
            const layers = [3, 5, 4, 2]; // Input, Hidden, Output
            
            layers.forEach((count, lIdx) => {{
                const layerDiv = document.createElement('div');
                layerDiv.className = 'layer';
                
                for(let i=0; i<count; i++) {{
                    const node = document.createElement('div');
                    node.className = 'node';
                    node.id = `node-${{lIdx}}-${{i}}`;
                    layerDiv.appendChild(node);
                }}
                container.appendChild(layerDiv);
            }});
        }}

        // --- SIMULATION LOOP ---
        function animateNet() {{
            // Randomly activate nodes to simulate inference
            document.querySelectorAll('.node').forEach(n => n.classList.remove('active'));
            
            // Activate input based on real values
            const d = RAW_DATA[currentPolicy];
            if(d.moisture[step] < 0.3) document.getElementById('node-0-0').classList.add('active'); // Moisture input
            if(d.temperature[step] > 30) document.getElementById('node-0-1').classList.add('active'); // Temp input
            
            // Random hidden
            for(let i=0; i<3; i++) {{
                const rL = Math.floor(Math.random() * 2) + 1; // layers 1,2
                const rN = Math.floor(Math.random() * 4);
                const el = document.getElementById(`node-${{rL}}-${{rN}}`);
                if(el) el.classList.add('active');
            }}
            
            // Output
            if(d.water[step] > 0) document.getElementById('node-3-0').classList.add('active'); // Water Output
            if(d.lamp[step] > 0) document.getElementById('node-3-1').classList.add('active'); // Lamp Output
        }}

        function updateSim(idx) {{
            const d = RAW_DATA[currentPolicy];
            if(!d || idx >= d.t.length) return;
            
            // Metrics
            document.getElementById('val-moisture').innerText = (d.moisture[idx]*100).toFixed(0);
            document.getElementById('bar-moisture').style.width = (d.moisture[idx]*100) + '%';
            document.getElementById('val-temp').innerText = d.temperature[idx].toFixed(1);
            document.getElementById('bar-temp').style.width = ((d.temperature[idx]/40)*100) + '%';
            document.getElementById('val-health').innerText = d.health[idx].toFixed(0) + '%';
            
            // Stats
            if(d.water[idx] > 0) totalWater += d.water[idx];
            document.getElementById('stats-water').innerText = (totalWater/1000).toFixed(1) + ' L';
            
            // Logs
            if(d.water[idx] > 0) log(`Inference: ACTION_WATER [Prob: ${{(0.8 + Math.random()*0.2).toFixed(2)}}]`);
            if(d.lamp[idx] > 0) {{
                document.getElementById('lamp-overlay').style.opacity = 1;
                log(`Inference: ACTION_LIGHT [Prob: ${{(0.9 + Math.random()*0.1).toFixed(2)}}]`);
            }} else {{
                document.getElementById('lamp-overlay').style.opacity = 0;
            }}
            
            animateNet();
        }}
        
        function log(msg) {{
            const c = document.getElementById('inf-log');
            const l = document.createElement('div');
            l.className = 'inf-line';
            l.innerHTML = `<span class="inf-time">${{new Date().toLocaleTimeString()}}</span><span class="inf-tensor">${{msg}}</span>`;
            c.prepend(l);
            if(c.children.length > 8) c.lastChild.remove();
        }}

        function togglePlay() {{
            if(isPlaying) {{
                clearInterval(intervalId); isPlaying=false;
                document.getElementById('playBtn').innerText = "RESUME";
            }} else {{
                isPlaying=true;
                document.getElementById('playBtn').innerText = "PAUSE";
                intervalId = setInterval(() => {{
                    step++;
                    if(step >= RAW_DATA[currentPolicy].t.length) step=0;
                    updateSim(step);
                }}, 100);
            }}
        }}

        initNeuralViz();
        updateSim(0);

    </script>
</body>
</html>
    """
    
    output_path = 'docs/cool_demo.html'
    os.makedirs("docs", exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"🚀 PRO DEMO GENERATED: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_cool_html()
