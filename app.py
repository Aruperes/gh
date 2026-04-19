import os, cv2, torch, datetime, functools, time
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
from torch.serialization import add_safe_globals
try:
    from numpy.dtypes import Float64DType, Int64DType
    add_safe_globals([Float64DType, Int64DType])
except ImportError:
    pass

add_safe_globals([
    np._core.multiarray._reconstruct, np.ndarray, np.dtype,
    np.core.multiarray._reconstruct, np.float64, np.int64
])

torch.load = functools.partial(torch.load, weights_only=False)
import torch.serialization
torch.serialization.safe_globals = lambda *args, **kwargs: None 

if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

req_path = os.path.join(os.getcwd(), 'yolov7', 'requirements.txt')
if os.path.exists(req_path):
    with open(req_path, 'w') as f: f.write("")

app = Flask(__name__)
app.secret_key = "asrama_gh_key_secret"

users = {
    "admin": {"password": "123", "role": "admin"},
    "op1": {"password": "123", "role": "operator"}
}

mahasiswa_db = [
    {"id": "1", "label": "budi", "nama": "Budi Setiawan", "status": "asrama"},
    {"id": "2", "label": "ani", "nama": "Anisa Rahmawati", "status": "asrama"},
    {"id": "3", "label": "陌生人", "nama": "Orang Asing", "status": "bukan_asrama"}
]

attendance_history = [] 

try:
    model = torch.hub.load('./yolov7', 'custom', 'best_model_asrama.pt', 
                           source='local', trust_repo=True)
    print("✅ BERHASIL: Model YOLOv7 siap digunakan!")
except Exception as e:
    print(f"❌ ERROR SAAT LOAD MODEL: {e}")

def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        success, frame = cap.read()
        if not success: break
        
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)
            detections = results.pandas().xyxy[0]
            
            for _, row in detections.iterrows():
                if row['confidence'] > 0.5:
                    label_model = row['name']
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    
                    # Cari data di dummy database
                    mhs = next((item for item in mahasiswa_db if item["label"] == label_model), None)
                    
                    if mhs and mhs['status'] == 'asrama':
                        color = (0, 255, 0) # Hijau
                        display_name = mhs['nama']
                    else:
                        color = (0, 0, 255) # Merah
                        display_name = "STRANGER / BUKAN ASRAMA"
                    
                    # Simpan ke history (dengan cooldown 10 detik per orang agar tidak penuh)
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%H:%M:%S")
                    if not any(h['nama'] == display_name and (now - h['raw_time']).seconds < 10 for h in attendance_history):
                        attendance_history.append({'nama': display_name, 'waktu': timestamp, 'raw_time': now})

                    # Gambar di frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, display_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
        except Exception as e:
            print(f"⚠️ Deteksi Error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    if 'user' in session: return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pw = request.form.get('password')
        if user in users and users[user]['password'] == pw:
            session['user'] = user
            session['role'] = users[user]['role']
            return redirect(url_for('dashboard'))
        flash("Username atau password salah!")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('dashboard.html', role=session['role'], user=session['user'])

@app.route('/admin/mahasiswa')
def admin_mahasiswa():
    if session.get('role') != 'admin': return "Akses Ditolak"
    return render_template('admin_data.html', mahasiswa=mahasiswa_db)

@app.route('/admin/history')
def admin_history():
    if session.get('role') != 'admin': return "Akses Ditolak"
    return render_template('history.html', history=attendance_history)

@app.route('/operator/monitor')
def operator_monitor():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('operator_monitor.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)