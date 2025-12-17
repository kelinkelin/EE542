
import pandas as pd
import json
import os

def generate_html():
    # Load data
    df = pd.read_csv('data/rollouts.csv')
    
    data_dict = {}
    policies = df['policy'].unique()
    
    # Sort policies to ensure consistent order: fixed, threshold, ppo
    # This order helps with color assignment
    order = {'fixed': 0, 'threshold': 1, 'ppo': 2}
    policies = sorted(policies, key=lambda x: order.get(x, 99))
    
    for policy in policies:
        policy_df = df[df['policy'] == policy]
        
        # Filter for the first continuous sequence of t (first episode)
        first_episode_mask = (policy_df['t'] - policy_df['t'].shift(1).fillna(-1)) > 0
        
        # Let's just grab the first 700 steps (approx 30 days)
        demo_len = 700
        subset = policy_df.iloc[:demo_len]
        
        data_dict[policy] = {
            't': subset['t'].tolist(),
            'health': subset['plant_health'].tolist(),
            'moisture': subset['soil_moisture'].tolist(),
            'water': subset['water_amount'].tolist(),
            'light': subset['lamp_on'].tolist(),
            'hour': subset['hour_of_day'].tolist()
        }

    json_data = json.dumps(data_dict)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Plant Care - Final PPO Results</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f2f5;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .controls {{
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
        }}
        button {{
            padding: 10px 25px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            transition: background-color 0.3s;
            font-weight: bold;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: #fff;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        #plots {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .plot-container {{
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            background: white;
        }}
        .legend-container {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 15px;
            font-size: 14px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌿 Smart Plant Care: Final Project Demo</h1>
        <div class="subtitle">Comparing Traditional Methods vs. PPO Reinforcement Learning</div>
        
        <div class="legend-container">
            <div class="legend-item"><div class="dot" style="background: #FF6B6B"></div> Fixed Schedule</div>
            <div class="legend-item"><div class="dot" style="background: #FFB366"></div> Threshold Rule</div>
            <div class="legend-item"><div class="dot" style="background: #2ECC71"></div> <b>PPO Agent (AI)</b></div>
        </div>

        <div class="controls">
            <button id="playBtn" onclick="toggleAnimation()">Start Simulation</button>
            <button onclick="resetAnimation()" style="background-color: #95a5a6;">Reset</button>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span>Speed:</span>
                <input type="range" id="speedRange" min="10" max="200" value="50">
            </div>
        </div>

        <div class="dashboard">
            <div class="metric-card">
                <div class="metric-value" id="timeStep">Day 0 - 00:00</div>
                <div class="metric-label">Time</div>
            </div>
            <div class="metric-card" style="border-top: 4px solid #2ECC71">
                <div class="metric-value" id="healthPPO" style="color: #27ae60">0</div>
                <div class="metric-label">PPO Health</div>
            </div>
            <div class="metric-card" style="border-top: 4px solid #2ECC71">
                <div class="metric-value" id="waterPPO" style="color: #27ae60">0 ml</div>
                <div class="metric-label">PPO Water Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="improvement" style="color: #e67e22">+0%</div>
                <div class="metric-label">Health Gain vs Baseline</div>
            </div>
        </div>

        <div id="plots">
            <div id="healthPlot" class="plot-container"></div>
            <div id="moisturePlot" class="plot-container"></div>
        </div>
    </div>

    <script>
        // Data from Python
        const data = {json_data};
        // Ensure specific order
        const policies = ['fixed', 'threshold', 'ppo'].filter(p => data[p]);
        
        const colors = {{
            'fixed': '#FF6B6B',
            'threshold': '#FFB366',
            'ppo': '#2ECC71'
        }};
        
        const names = {{
            'fixed': 'Fixed Schedule',
            'threshold': 'Threshold Rule',
            'ppo': 'PPO Agent (AI)'
        }};

        let currentIndex = 0;
        let isPlaying = false;
        let animationId = null;
        const maxIndex = data[policies[0]].t.length;

        // Initialize Plots
        function initPlots() {{
            // Health Plot
            const healthTraces = policies.map(p => ({{
                x: [0],
                y: [data[p].health[0]],
                name: names[p],
                mode: 'lines',
                line: {{ 
                    color: colors[p],
                    width: p === 'ppo' ? 4 : 2 
                }}
            }}));

            const healthLayout = {{
                title: 'Plant Health Over Time (Higher is Better)',
                yaxis: {{ title: 'Health Score (0-100)', range: [0, 100] }},
                height: 400,
                margin: {{ t: 40, r: 20, l: 60, b: 40 }},
                legend: {{ orientation: 'h', y: -0.2 }}
            }};

            Plotly.newPlot('healthPlot', healthTraces, healthLayout);

            // Moisture Plot
            const moistureTraces = policies.map(p => ({{
                x: [0],
                y: [data[p].moisture[0]],
                name: names[p],
                mode: 'lines',
                line: {{ 
                    color: colors[p],
                    width: p === 'ppo' ? 3 : 1.5,
                    dash: p === 'ppo' ? 'solid' : 'dot'
                }}
            }}));

            const moistureLayout = {{
                title: 'Soil Moisture (Target: 40-70%)',
                yaxis: {{ title: 'Moisture (%)', range: [0, 100] }},
                height: 300,
                margin: {{ t: 40, r: 20, l: 60, b: 40 }},
                shapes: [
                    {{
                        type: 'rect',
                        xref: 'paper', yref: 'y',
                        x0: 0, x1: 1,
                        y0: 40, y1: 70,
                        fillcolor: 'green',
                        opacity: 0.1,
                        line: {{ width: 0 }}
                    }}
                ],
                showlegend: false
            }};

            Plotly.newPlot('moisturePlot', moistureTraces, moistureLayout);
        }}

        function updateDashboard(idx) {{
            // Update Time
            const t = data[policies[0]].t[idx];
            const day = Math.floor(t / 24);
            const hour = t % 24;
            document.getElementById('timeStep').innerText = `Day ${{day}} - ${{hour.toString().padStart(2, '0')}}:00`;

            // Calculate stats for PPO
            const ppo = 'ppo';
            if (data[ppo]) {{
                const currentHealth = data[ppo].health[idx];
                const waterSlice = data[ppo].water.slice(0, idx+1);
                const totalWater = waterSlice.reduce((a, b) => a + b, 0);
                
                document.getElementById('healthPPO').innerText = currentHealth.toFixed(1);
                document.getElementById('waterPPO').innerText = `${{totalWater.toFixed(0)}} ml`;
                
                // Compare with Fixed
                if (data['fixed']) {{
                    const fixedHealth = data['fixed'].health[idx];
                    if (fixedHealth > 0) {{
                        const imp = ((currentHealth - fixedHealth) / fixedHealth) * 100;
                        const sign = imp >= 0 ? '+' : '';
                        document.getElementById('improvement').innerText = `${{sign}}${{imp.toFixed(0)}}%`;
                        document.getElementById('improvement').style.color = imp >= 0 ? '#27ae60' : '#c0392b';
                    }}
                }}
            }}
        }}

        function step() {{
            if (currentIndex >= maxIndex - 1) {{
                isPlaying = false;
                document.getElementById('playBtn').innerText = 'Restart Simulation';
                return;
            }}

            currentIndex++;
            
            // Extend traces for all policies
            const traceIndices = Array.from(Array(policies.length).keys());
            
            Plotly.extendTraces('healthPlot', {{
                x: policies.map(p => [data[p].t[currentIndex]]),
                y: policies.map(p => [data[p].health[currentIndex]])
            }}, traceIndices);

            Plotly.extendTraces('moisturePlot', {{
                x: policies.map(p => [data[p].t[currentIndex]]),
                y: policies.map(p => [data[p].moisture[currentIndex]])
            }}, traceIndices);

            updateDashboard(currentIndex);

            if (isPlaying) {{
                const rangeVal = document.getElementById('speedRange').value;
                const speed = 210 - rangeVal; // Invert range
                setTimeout(step, speed);
            }}
        }}

        function toggleAnimation() {{
            if (currentIndex >= maxIndex - 1) {{
                resetAnimation();
            }}
            
            isPlaying = !isPlaying;
            document.getElementById('playBtn').innerText = isPlaying ? 'Pause' : 'Resume';
            
            if (isPlaying) {{
                step();
            }}
        }}

        function resetAnimation() {{
            isPlaying = false;
            currentIndex = 0;
            document.getElementById('playBtn').innerText = 'Start Simulation';
            initPlots();
            updateDashboard(0);
        }}

        // Init
        initPlots();
        updateDashboard(0);

    </script>
</body>
</html>
    """
    
    os.makedirs("docs", exist_ok=True)
    with open('docs/demo.html', 'w') as f:
        f.write(html_content)
    
    print(f"✅ Generated Final HTML demo at: {os.path.abspath('docs/demo.html')}")

if __name__ == "__main__":
    generate_html()
