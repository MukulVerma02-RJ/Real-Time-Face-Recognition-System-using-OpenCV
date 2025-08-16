import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import os

# Define file paths
face_cascade_path = r'C:\Users\mukul\emotion final\haarcascade_frontalface_default.xml'
model_path = r'C:\Users\mukul\emotion final\emotion_model.h5'

# Check if files exist
if not os.path.exists(face_cascade_path) or not os.path.exists(model_path):
    QMessageBox.critical(None, "Missing Files", "Make sure 'emotion_model.h5' and 'haarcascade_frontalface_default.xml' exist in the specified path.")
    sys.exit(1)

# Load models
face_classifier = cv2.CascadeClassifier(face_cascade_path)
classifier = load_model(model_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Detection - PyQt5")
        self.setFixedSize(800, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        self.image_label = QLabel("Camera feed will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.image_label.setStyleSheet("""
            border-radius: 20px;
            background-color: #333333;
            border: 1px solid #444444;
            padding: 10px;
        """)

        self.start_button = QPushButton("Start Camera")
        self.start_button.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2f80ed;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #1c6dd0;
            }
        """)
        self.start_button.clicked.connect(self.start_camera)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Error", "Could not access the webcam.")
            return
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
                roi_rgb = roi_rgb.astype("float") / 255.0
                roi_rgb = img_to_array(roi_rgb)
                roi_rgb = np.expand_dims(roi_rgb, axis=0)

                prediction = classifier.predict(roi_rgb, verbose=0)[0]
                label = emotion_labels[prediction.argmax()]
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        step = channel * width
        q_image = QImage(rgb_image.data, width, height, step, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.timer.stop()
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())
# This code is a PyQt5 application for emotion detection using a webcam feed.
# It uses OpenCV for face detection and a pre-trained Keras model for emotion classification.                           