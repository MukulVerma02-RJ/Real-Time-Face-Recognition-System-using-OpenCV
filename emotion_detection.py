import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import tensorflow as tf

# ========== GPU CHECK ==========
if tf.config.list_physical_devices('GPU'):
    print("\U0001F680 GPU detected and will be used.")
else:
    print("\u26A0\ufe0f No GPU detected. Running on CPU.")

# ========== LOAD MODEL ==========
model_path = 'emotion_model.h5'  # Ensure this file exists in the directory
if os.path.exists(model_path):
    print("\u2705 Model loaded successfully.")
    model = load_model(model_path)
else:
    print("\u274C Model not found. Please ensure 'emotion_model.h5' is in the folder.")
    sys.exit(1)

# ========== EMOTION DETECTOR GUI ==========
class EmotionDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Detector")
        self.setFixedSize(800, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        # Required files check
        if not os.path.exists("haarcascade_frontalface_default.xml"):
            QMessageBox.critical(self, "Error", "Missing 'haarcascade_frontalface_default.xml' file.")
            sys.exit(1)

        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = model
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.highlight_color = (0, 255, 255)

        self.label = QLabel("Camera feed will appear here.")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.label.setStyleSheet("border-radius: 20px; background-color: #333333; border: 1px solid #444444; padding: 10px;")

        self.button_start = QPushButton("Start Camera")
        self.button_start.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.button_start.setStyleSheet("QPushButton { background-color: #2f80ed; color: white; border: none; border-radius: 10px; padding: 10px 20px; } QPushButton:hover { background-color: #1c6dd0; }")

        self.button_stop = QPushButton("Stop Camera")
        self.button_stop.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.button_stop.setStyleSheet("QPushButton { background-color: #eb5757; color: white; border: none; border-radius: 10px; padding: 10px 20px; } QPushButton:hover { background-color: #cc4c4c; }")

        self.button_start.clicked.connect(self.start_camera)
        self.button_stop.clicked.connect(self.stop_camera)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button_start)
        layout.addWidget(self.button_stop)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not access the webcam.")
            return
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.label.setText("Camera stopped.")
        self.label.setPixmap(QPixmap())

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

            if np.sum([roi_gray]) != 0:
                roi = roi_rgb.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = self.model.predict(roi, verbose=0)[0]
                label = self.emotion_labels[prediction.argmax()]
                label_position = (x, y - 10)

                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.highlight_color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.highlight_color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDetector()
    window.show()
    sys.exit(app.exec_())
# This code is a PyQt5 application that uses OpenCV to detect emotions from webcam feed.
# It loads a pre-trained emotion detection model and uses a Haar Cascade classifier for face detection.
# The application displays the webcam feed, detects faces, predicts emotions, and overlays the results on the video stream.
# The user can start and stop the camera feed using buttons in the GUI.
# Ensure that the required files like 'haarcascade_frontalface_default.xml' and 'emotion_model.h5' are present in the same directory as this script.
# The application is designed to run on systems with a webcam and requires the PyQt5 and OpenCV libraries.
# It also checks for GPU availability to utilize hardware acceleration if available.
# The emotion labels are defined as a list, and the application uses a timer to update the video feed at regular intervals.
# The GUI is styled with a modern look using CSS-like styles for buttons and labels.
# The application handles errors gracefully, displaying error messages if the webcam cannot be accessed or if required files are missing.
# The application is structured to be user-friendly, with clear labels and buttons for interaction.
# It is a complete solution for real-time emotion detection using a webcam feed, suitable for educational and practical applications in computer    