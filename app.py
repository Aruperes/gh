import os, cv2, torch, datetime, functools, time, base64
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from torch.serialization import add_safe_globals
from PIL import Image
import io

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

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

camera_active = True

ASRAMA_LABELS = ['Arlan', 'Christiano', 'Fael', 'gafe', 'Jonathan', 'Kenneth', 'Kevin', 'Kia', 'Matthew', 'Natan', 'Reins', 'Sdyney']

LABEL_FIX = {
    'Reins': 'Sdyney',
    'Natan': 'Reins',
    'Kia': 'Matthew',
    'Kevin': 'Kia',
    'Kenneth': 'Kevin',
    'gafe': 'Jonathan',
    'Sdyney': 'gafe',
}

users = {
    "admin": {"password": "123", "role": "admin"},
    "op1": {"password": "123", "role": "operator"}
}

try:
    model = torch.hub.load('./yolov7', 'custom', 'best_model_asrama.pt', 
                           source='local', trust_repo=True)
    print("✅ BERHASIL: Model YOLOv7 siap digunakan!")
except Exception as e:
    print(f"❌ ERROR SAAT LOAD MODEL: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def gen_frames():
    global camera_active
    cap = cv2.VideoCapture(0)
    while True:
        if not camera_active:
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "OFF", (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', black_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success: break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = results.pandas().xyxy[0]
        
        frame, _ = process_detections(frame, detections)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def process_detections(frame, detections):
    global last_detection_status
    detected_info = []
    unknown_found = False
    
    for _, row in detections.iterrows():
        confidence = row['confidence']
        if confidence > 0.40: 
            label_model = row['name']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            if label_model in LABEL_FIX:
                label_model = LABEL_FIX[label_model]
            
            if label_model in ASRAMA_LABELS and confidence > 0.75:
                color = (0, 255, 0)
                status = "ASRAMA"
                display_name = f"{label_model.upper()} {int(confidence*100)}%"
            else:
                color = (0, 0, 255)
                status = "STRANGER"
                display_name = "UNKNOWN"
                unknown_found = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            cv2.putText(frame, display_name, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 4)
            
            detected_info.append({'nama': display_name, 'status': status, 'confidence': float(confidence)})

    if unknown_found:
        last_detection_status = {"unknown_detected": True, "timestamp": time.time()}
    
    return frame, detected_info

@app.route('/')
def index():
    if 'user' not in session: return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

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

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('dashboard.html', role=session['role'], user=session['user'])

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return jsonify({'active': camera_active})

@app.route('/check_alert')
def check_alert():
    global last_detection_status
    is_active = (time.time() - last_detection_status['timestamp']) < 2
    return jsonify({'alert': is_active})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/operator/monitor')
def operator_monitor():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('operator_monitor.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = results.pandas().xyxy[0]
        
        img_processed, info = process_detections(img, detections)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'res_' + filename)
        cv2.imwrite(output_path, img_processed)
        
        with open(output_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()

        return jsonify({'success': True, 'image': 'data:image/jpeg;base64,' + img_base64, 'detections': info})
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)