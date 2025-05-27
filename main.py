import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# --- Konfigurasi Penting ---
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Pastikan folder upload ada saat aplikasi dimulai
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi untuk memeriksa ekstensi file yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_traffic(image_obj, model_path, labels_path):
    np.set_printoptions(suppress=True)

    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model: {e}")
    
    try:
        class_names = open(labels_path, "r").readlines()
    except Exception as e:
        raise RuntimeError(f"Gagal membaca file label: {e}")

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image_obj = image_obj.convert("RGB")
    size = (224, 224)
    image_obj = ImageOps.fit(image_obj, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_obj)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Memprediksi model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    # Ambil nama dan gambar dari formulir
    name = request.form.get('name')
    
    if 'traffic_image' not in request.files:
        return render_template('error.html', message="Tidak ada file gambar yang diunggah!")

    traffic_image_file = request.files['traffic_image']

    if traffic_image_file.filename == '':
        return render_template('error.html', message="Tidak ada file yang dipilih!")

    if name and traffic_image_file and allowed_file(traffic_image_file.filename):
        filename = f"{uuid.uuid4().hex}_{secure_filename(traffic_image_file.filename)}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        traffic_image_file.save(save_path)

        try:
            img_to_process = Image.open(save_path)
            detected_result, confidence = detect_traffic(img_to_process, "keras_model.h5", "labels.txt")

            return render_template('index.html', 
                               name=name, 
                               detected_result=detected_result, 
                               confidence=f"{confidence*100:.2f}%",
                               image_url=url_for('static', filename='images/' + filename))

        except Exception as e:
            return render_template('error.html', message=f"Terjadi kesalahan saat memproses gambar: {e}")
        
    # else:
    #     return "Formulir tidak lengkap atau ekstensi file tidak diizinkan!"
    return render_template('error.html', message="Formulir tidak lengkap atau ekstensi file tidak diizinkan!")


if __name__ == '__main__':
    app.run(debug=True)
