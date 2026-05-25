from flask import Flask, request, jsonify, render_template_string
import numpy as np
import json, os
import tensorflow as tf

# ── Load model & scaler at startup ────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model("lstm_weather.keras")
scaler_min   = np.load("scaler_min.npy")
scaler_range = np.load("scaler_scale.npy")
print("Model loaded successfully!")

app = Flask(__name__)

def scale_val(val, idx):
    return (val - scaler_min[idx]) / scaler_range[idx]

def unscale_temp(v):
    return v * scaler_range[0] + scaler_min[0]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WeatherLSTM · TensorFlow</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f0f4f8;--surface:#ffffff;--card:#ffffff;
  --border:#e2e8f0;--accent:#2563eb;--accent-light:#dbeafe;
  --green:#16a34a;--green-light:#dcfce7;
  --text:#0f172a;--muted:#64748b;--subtle:#94a3b8;
  --shadow:0 1px 3px rgba(0,0,0,.08),0 4px 16px rgba(0,0,0,.06);
  --font:'Space Grotesk',sans-serif;
  --mono:'JetBrains Mono',monospace;
}
body{background:var(--bg);color:var(--text);font-family:var(--font);
     min-height:100vh;padding:2rem 1rem;}
.header{max-width:1000px;margin:0 auto 2.5rem;}
.badge{display:inline-flex;align-items:center;gap:.4rem;
       background:var(--accent-light);color:var(--accent);
       border-radius:999px;padding:.25rem .75rem;font-size:.72rem;
       font-weight:600;letter-spacing:.04em;margin-bottom:1rem;}
h1{font-size:clamp(1.8rem,4vw,2.8rem);font-weight:700;
   letter-spacing:-.04em;line-height:1.1;color:var(--text);}
h1 span{color:var(--accent);}
.subtitle{color:var(--muted);font-size:.9rem;margin-top:.5rem;}
.tags{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem;}
.tag{background:var(--surface);border:1px solid var(--border);
     border-radius:6px;padding:.2rem .6rem;font-size:.72rem;
     font-family:var(--mono);color:var(--muted);}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;
      max-width:1000px;margin:0 auto;}
@media(max-width:660px){.grid{grid-template-columns:1fr;}}
.full{grid-column:1/-1;}
.card{background:var(--surface);border:1px solid var(--border);
      border-radius:16px;padding:1.75rem;box-shadow:var(--shadow);}
.card-title{font-size:.7rem;font-weight:600;letter-spacing:.08em;
            text-transform:uppercase;color:var(--muted);margin-bottom:1.25rem;}
label{display:block;font-size:.78rem;font-weight:500;
      color:var(--muted);margin-bottom:.35rem;margin-top:1rem;}
label:first-of-type{margin-top:0;}
input[type=number]{
  width:100%;background:var(--bg);border:1.5px solid var(--border);
  border-radius:10px;padding:.65rem .9rem;color:var(--text);
  font-family:var(--mono);font-size:.9rem;transition:border-color .2s;}
input[type=number]:focus{outline:none;border-color:var(--accent);
  box-shadow:0 0 0 3px rgba(37,99,235,.12);}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.75rem;margin-top:1rem;}
button{width:100%;margin-top:1.5rem;padding:.8rem;
  background:var(--accent);color:#fff;border:none;border-radius:10px;
  font-family:var(--font);font-weight:600;font-size:.95rem;cursor:pointer;
  transition:background .2s,transform .1s;}
button:hover{background:#1d4ed8;}
button:active{transform:scale(.98);}
.result{margin-top:1.5rem;background:var(--green-light);border:1px solid #86efac;
        border-radius:12px;padding:1.25rem;display:none;}
.temp-big{font-size:3.2rem;font-weight:700;letter-spacing:-.05em;
          color:var(--green);font-family:var(--mono);}
.temp-label{font-size:.78rem;color:#166534;margin-top:.2rem;}
.metrics{display:flex;gap:.75rem;margin-top:1rem;flex-wrap:wrap;}
.metric{background:#fff;border:1px solid #bbf7d0;border-radius:8px;
        padding:.6rem 1rem;flex:1;min-width:80px;}
.metric .val{font-size:1rem;font-weight:600;color:var(--green);font-family:var(--mono);}
.metric .key{font-size:.68rem;color:var(--muted);margin-top:.15rem;}
.error{color:#dc2626;font-size:.82rem;margin-top:.75rem;
       background:#fef2f2;border:1px solid #fecaca;border-radius:8px;
       padding:.6rem .9rem;display:none;}
.arch{display:flex;flex-direction:column;gap:.5rem;margin-top:.5rem;}
.arch-box{background:var(--accent-light);border:1.5px solid var(--accent);
          border-radius:8px;padding:.45rem .9rem;font-size:.75rem;
          font-family:var(--mono);color:var(--accent);font-weight:500;
          text-align:center;}
.arch-box.gray{background:var(--bg);border-color:var(--border);color:var(--muted);}
.arch-box.green{background:var(--green-light);border-color:#86efac;color:var(--green);}
</style>
</head>
<body>

<div class="header">
  <div class="badge">⚡ TensorFlow · Keras</div>
  <h1>Weather<span>LSTM</span></h1>
  <p class="subtitle">Stacked LSTM · Delhi Climate Dataset · 30-day lookback · Next-day temperature forecast</p>
  <div class="tags">
    <span class="tag">TF 2.x / Keras</span>
    <span class="tag">LSTM × 2</span>
    <span class="tag">Dropout 0.2</span>
    <span class="tag">MinMaxScaler</span>
    <span class="tag">Sliding Window SEQ=30</span>
  </div>
</div>

<div class="grid">

  <div class="card">
    <div class="card-title">📥 Today's Weather Input</div>
    <label>Mean Temperature (°C)</label>
    <input type="number" id="temp" value="25" step="0.1" min="-10" max="55">
    <div class="row3">
      <div>
        <label>Humidity (%)</label>
        <input type="number" id="humidity" value="60" step="1" min="0" max="100">
      </div>
      <div>
        <label>Wind (km/h)</label>
        <input type="number" id="wind" value="10" step="0.5" min="0" max="100">
      </div>
      <div>
        <label>Pressure (hPa)</label>
        <input type="number" id="pressure" value="1015" step="1" min="950" max="1080">
      </div>
    </div>
    <button onclick="predict()">Predict Tomorrow →</button>
    <div class="error" id="err">⚠ Prediction failed. Check terminal for errors.</div>
    <div class="result" id="result">
      <div class="temp-big" id="pred-val">—</div>
      <div class="temp-label">Predicted mean temperature tomorrow</div>
      <div class="metrics">
        <div class="metric">
          <div class="val" id="m-mae">—</div>
          <div class="key">Validation MAE</div>
        </div>
        <div class="metric">
          <div class="val" id="m-rmse">—</div>
          <div class="key">Validation RMSE</div>
        </div>
        <div class="metric">
          <div class="val">30</div>
          <div class="key">Lookback days</div>
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">🧠 Model Architecture</div>
    <div class="arch">
      <div class="arch-box gray">Input → (30 timesteps × 4 features)</div>
      <div style="text-align:center;color:#94a3b8">↓</div>
      <div class="arch-box">LSTM Layer 1 — 64 units, return_sequences=True</div>
      <div style="text-align:center;color:#94a3b8">↓</div>
      <div class="arch-box gray">Dropout (0.2)</div>
      <div style="text-align:center;color:#94a3b8">↓</div>
      <div class="arch-box">LSTM Layer 2 — 64 units, return_sequences=False</div>
      <div style="text-align:center;color:#94a3b8">↓</div>
      <div class="arch-box gray">Dropout (0.2)</div>
      <div style="text-align:center;color:#94a3b8">↓</div>
      <div class="arch-box">Dense (32, ReLU)</div>
      <div style="text-align:center;color:#94a3b8">↓</div>
      <div class="arch-box green">Dense (1) → Temperature °C</div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">📉 Training Loss (MSE)</div>
    <canvas id="lossChart" height="200"></canvas>
  </div>

  <div class="card">
    <div class="card-title">📐 Training MAE</div>
    <canvas id="maeChart" height="200"></canvas>
  </div>

  <div class="card full">
    <div class="card-title">📈 Validation: Predicted vs Actual (first 60 days)</div>
    <canvas id="predChart" height="100"></canvas>
  </div>

</div>

<script>
const C={
  responsive:true,
  plugins:{legend:{labels:{color:'#64748b',font:{family:'Space Grotesk',size:11}}}},
  scales:{
    x:{ticks:{color:'#94a3b8',maxTicksLimit:8,font:{size:10}},grid:{color:'#f1f5f9'}},
    y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#f1f5f9'}}
  }
};

async function loadCharts(){
  try{
    const r=await fetch('/results');
    if(!r.ok) return;
    const d=await r.json();
    const ep=d.history.train_loss.map((_,i)=>i+1);
    new Chart(document.getElementById('lossChart'),{type:'line',data:{labels:ep,datasets:[
      {label:'Train',data:d.history.train_loss,borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,.08)',borderWidth:2,pointRadius:0,tension:.4},
      {label:'Val',  data:d.history.val_loss,  borderColor:'#f59e0b',backgroundColor:'rgba(245,158,11,.08)',borderWidth:2,pointRadius:0,tension:.4}
    ]},options:{...C}});
    new Chart(document.getElementById('maeChart'),{type:'line',data:{labels:ep,datasets:[
      {label:'Train MAE',data:d.history.train_mae,borderColor:'#16a34a',backgroundColor:'rgba(22,163,74,.08)',borderWidth:2,pointRadius:0,tension:.4},
      {label:'Val MAE',  data:d.history.val_mae,  borderColor:'#dc2626',backgroundColor:'rgba(220,38,38,.08)',borderWidth:2,pointRadius:0,tension:.4}
    ]},options:{...C}});
    const days=d.sample_true.map((_,i)=>`Day ${i+1}`);
    new Chart(document.getElementById('predChart'),{type:'line',data:{labels:days,datasets:[
      {label:'Actual',   data:d.sample_true, borderColor:'#0f172a',borderWidth:1.5,pointRadius:0,tension:.3},
      {label:'Predicted',data:d.sample_preds,borderColor:'#2563eb',borderWidth:2,  pointRadius:0,tension:.3,borderDash:[5,3]}
    ]},options:{...C,maintainAspectRatio:true}});
    document.getElementById('m-mae').textContent =d.mae+' °C';
    document.getElementById('m-rmse').textContent=d.rmse+' °C';
  }catch(e){console.warn('No results yet.');}
}

async function predict(){
  const body={
    temperature:+document.getElementById('temp').value,
    humidity:   +document.getElementById('humidity').value,
    wind_speed: +document.getElementById('wind').value,
    pressure:   +document.getElementById('pressure').value,
  };
  try{
    const r=await fetch('/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error) throw new Error(d.error);
    document.getElementById('pred-val').textContent=d.predicted_temp_c+' °C';
    document.getElementById('result').style.display='block';
    document.getElementById('err').style.display='none';
  }catch(e){
    document.getElementById('err').style.display='block';
  }
}

loadCharts();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    t = float(data.get("temperature", 25))
    h = float(data.get("humidity",    60))
    w = float(data.get("wind_speed",  10))
    p = float(data.get("pressure",  1013))

    seq = []
    for i in range(30):
        alpha = i / 29.0
        row = [
            scale_val(t * alpha + 15 * (1 - alpha), 0),
            scale_val(h * alpha + 55 * (1 - alpha), 1),
            scale_val(w * alpha + 8  * (1 - alpha), 2),
            scale_val(p * alpha + 1013*(1-alpha),   3),
        ]
        seq.append(row)

    x = np.array([seq], dtype=np.float32)
    pred_scaled = float(model.predict(x, verbose=0)[0][0])
    pred_temp   = round(unscale_temp(pred_scaled), 1)
    return jsonify({"predicted_temp_c": pred_temp})

@app.route("/results")
def results():
    if os.path.exists("results.json"):
        with open("results.json") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "results.json not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
