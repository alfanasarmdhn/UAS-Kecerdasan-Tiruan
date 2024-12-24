from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
import base64

# Ini WM punya ALFAN
# Fungsi untuk menambahkan latar belakang
def set_background(image_path):
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image;base64,{base64_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Function untuk memproses hasil deteksi
def display_results(image, results):
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results.boxes.conf.cpu().numpy()  # Confidence scores
    labels = results.boxes.cls.cpu().numpy()  # Class indices
    names = results.names  # Class names
    
    detected_objects = []
    
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(labels[i])]
            score = scores[i]
            detected_objects.append(label)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, detected_objects

# Fungsi utama aplikasi
def main():

    # Menambahkan teks "Real-time Object Detection with YOLO"
    st.markdown("<h1 style='text-align: center;'>Real-time Object Detection with YOLO - ALFAN</h1>", unsafe_allow_html=True)

    # Menambahkan judul pada sidebar
    st.sidebar.title("SELAMAT DATANG DI REAL TIME OBJECT DETECTION")

    # Menambahkan subheader pada side bar
    st.sidebar.subheader("Lengkapi Data Diri")


    # Menambahkan teks input atau mengetikkan teks
    st.sidebar.text_input('Nama Lengkap')
    
    # Menambahkan selectbox sebagai pilihan
    option = st.sidebar.selectbox(
        'Jenis Kelamin',
        ('Laki-Laki','Perempuan',)
    )
    
    # Menambahkan fungsi umur dalam bilangan bulat
    umur = st.sidebar.number_input('Umur', min_value=0, max_value=120, step=1, format="%d")

    # Menambahkan button atau tombol
    st.sidebar.button('simpan')

    # Menambahkan subheader di sidebar
    st.sidebar.subheader("Klik Tombol Dibawah Untuk Memulai")
    
    # Atur background
    set_background("Background.jpg")  # Path ke gambar latar belakang
    
    # Load model YOLO
    model_path = "yolo11n.pt"  # Path to your YOLO model
    model = load_model(model_path)

    # Tambahkan dua tombol untuk Start dan Stop
    start_button = st.sidebar.button("Start Object Detection")
    stop_button = st.sidebar.button("Stop Object Detection")

    # Menggunakan flag untuk mengontrol deteksi
    detection_active = False

    if start_button:
        detection_active = True
    
    if stop_button:
        detection_active = False
    
    if detection_active:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()  # Placeholder for video frames
        st_detection_info = st.empty()  # Placeholder for detection information

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture image.")
                break

            # Run YOLO detection
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
            results = model.predict(frame, imgsz=640)  # Perform detection
            
            # Draw results and collect detected objects
            frame, detected_objects = display_results(frame, results[0])
            
            # Display video feed
            st_frame.image(frame, channels="RGB", use_column_width=True)
            
            # Display detection information
            if detected_objects:
                object_counts = Counter(detected_objects)
                detection_info = "\n".join([f"{obj}: {count}" for obj, count in object_counts.items()])
            else:
                detection_info = "No objects detected."

            st_detection_info.text(detection_info)  # Update detection info text

            # Break the loop if detection is turned off
            if not detection_active:
                break
        
        cap.release()

if __name__ == "__main__":
    main()
