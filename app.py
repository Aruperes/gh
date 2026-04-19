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

# Konfigurasi upload file
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

users = {
    "admin": {"password": "123", "role": "admin"},
    "op1": {"password": "123", "role": "operator"}
}

mahasiswa_db = [
    {"id": "1", "label": "Arlan", "nama": "Arlan", "status": "asrama"},
    {"id": "2", "label": "Christiano", "nama": "Christiano", "status": "asrama"},
    {"id": "3", "label": "Fael", "nama": "Fael", "status": "asrama"},
    {"id": "4", "label": "gafe", "nama": "Gafe", "status": "asrama"},
    {"id": "5", "label": "Jonathan", "nama": "Jonathan", "status": "asrama"},
    {"id": "6", "label": "Kenneth", "nama": "Kenneth", "status": "asrama"},
    {"id": "7", "label": "Kevin", "nama": "Kevin", "status": "asrama"},
    {"id": "8", "label": "Kia", "nama": "Kia", "status": "asrama"},
    {"id": "9", "label": "Matthew", "nama": "Matthew", "status": "asrama"},
    {"id": "10", "label": "Natan", "nama": "Natan", "status": "asrama"},
    {"id": "11", "label": "Reins", "nama": "Reins", "status": "asrama"},
    {"id": "12", "label": "Sdyney", "nama": "Sdyney", "status": "asrama"}
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
    if not cap.isOpened():
        print("❌ ERROR: Webcam tidak dapat diakses!")
        yield (b'--frame\r\n' b'Content-Type: text/plain\r\n\r\n' + b'Webcam not available' + b'\r\n')
        return
    print("✅ Webcam berhasil diakses")
    while True:
        success, frame = cap.read()
        if not success: 
            print("❌ Gagal membaca frame dari webcam")
            break
        
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)
            detections = results.pandas().xyxy[0]
            
            for _, row in detections.iterrows():
                if row['confidence'] > 0.65:
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

def detect_on_image(image_path):
    """Deteksi pada gambar statis"""
    # Label dari model training di Kaggle
    ASRAMA_LABELS = ['Arlan', 'Christiano', 'Fael', 'gafe', 'Jonathan', 'Kenneth', 'Kevin', 'Kia', 'Matthew', 'Natan', 'Reins', 'Sdyney']
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "Gambar tidak dapat dibaca"
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = results.pandas().xyxy[0]
        
        output_img = img.copy()
        detected_people = []
        
        print(f"\n📊 DETECTION DEBUG:")
        print(f"Total detections: {len(detections)}")
        print(f"ASRAMA labels available: {ASRAMA_LABELS}")
        
        # Debug: Tampilkan SEMUA deteksi dengan confidence score
        if len(detections) > 0:
            print(f"\n📋 ALL DETECTIONS (sorted by confidence):")
            sorted_detections = detections.sort_values('confidence', ascending=False)
            for _, row in sorted_detections.iterrows():
                threshold_marker = "✅" if row['confidence'] > 0.65 else "❌"
                print(f"{threshold_marker} {row['name']:15} - Confidence: {row['confidence']:.2%}")
        
        for _, row in detections.iterrows():
            if row['confidence'] > 0.65:
                label_model = row['name']
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                
                print(f"\n✅ ACCEPTED Detection:")
                print(f"  Model detected: '{label_model}'")
                print(f"  Confidence: {row['confidence']:.2%}")
                
                # Check apakah label ada di ASRAMA list
                if label_model in ASRAMA_LABELS:
                    color = (0, 255, 0)  # Hijau
                    display_name = label_model
                    status = "ASRAMA"
                    print(f"  ✅ Status: ASRAMA")
                else:
                    color = (0, 0, 255)  # Merah
                    display_name = "STRANGER / BUKAN ASRAMA"
                    status = "BUKAN ASRAMA"
                    print(f"  ⚠️  Status: BUKAN ASRAMA")
                
                detected_people.append({
                    'nama': display_name,
                    'status': status,
                    'confidence': float(row['confidence']),
                    'model_label': label_model
                })
                
                # Gambar di frame
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output_img, display_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if len(detections) == 0:
            print("  ⚠️ No people detected in image")
        
        # Simpan hasil deteksi
        output_path = os.path.join(UPLOAD_FOLDER, 'detected_' + os.path.basename(image_path))
        cv2.imwrite(output_path, output_img)
        
        return output_path, detected_people
    except Exception as e:
        print(f"Error detecting on image: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

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

@app.route('/test_auth', methods=['GET'])
def test_auth():
    """Test endpoint to check authentication"""
    return jsonify({
        'authenticated': 'user' in session,
        'user': session.get('user', 'None'),
        'role': session.get('role', 'None')
    })

@app.route('/upload_image', methods=['POST'])
def upload_image():
    print(f"\n{'='*50}")
    print("UPLOAD ENDPOINT HIT")
    print(f"{'='*50}\n")
    try:
        # Debug logging
        print(f"Session data: {dict(session)}")
        print(f"User in session: {'user' in session}")
        print(f"Request content-type: {request.content_type}")
        print(f"Request files: {request.files}")
        
        if 'user' not in session:
            print("❌ No user in session - redirecting to login")
            return redirect(url_for('login'))
        
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            print(f"❌ File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed. Use: PNG, JPG, JPEG, GIF'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✅ File saved to: {filepath}")
        
        # Deteksi pada gambar
        output_path, detection_results = detect_on_image(filepath)
        
        if output_path is None:
            print(f"❌ Detection error: {detection_results}")
            return jsonify({'error': detection_results}), 400
        
        # Baca gambar hasil deteksi
        with open(output_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        print(f"✅ Detection complete. Found {len(detection_results)} people")
        print(f"{'='*50}\n")
        
        return jsonify({
            'success': True,
            'image': 'data:image/jpeg;base64,' + img_base64,
            'detections': detection_results,
            'message': f'Terdeteksi {len(detection_results)} orang'
        })
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*50}\n")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)