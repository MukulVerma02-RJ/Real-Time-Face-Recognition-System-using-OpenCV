# Real-Time Face Recognition System using OpenCV  

A real-time face recognition system built with **Python, OpenCV, and machine learning models**.  
The project detects and recognizes human faces from a webcam feed by comparing them against a trained dataset.  

---

## 🚀 Features  
- Real-time **face detection & recognition**  
- Supports **multiple users**  
- Easy dataset creation and training  
- Lightweight and runs on most machines  
- Displays live video with **bounding boxes & labels**  

---

## 🛠️ Tech Stack  
- **Python 3.x**  
- **OpenCV** – for video capture and image processing  
- **NumPy** – for array operations  
- **Dlib / Face-Recognition / TensorFlow-Keras** (optional for deep learning)  

---

## 📂 Project Structure  
📦 Real-Time-Face-Recognition-System-using-OpenCV
┣ 📂 dataset/ # Collected face images (not included in repo)
┣ 📂 trained_model/ # Saved embeddings/model (not included in repo)
┣ 📜 face_collect.py # Script to collect images for new users
┣ 📜 face_train.py # Script to train dataset and save model
┣ 📜 face_recog.py # Main real-time recognition script
┣ 📜 requirements.txt # Dependencies

▶️ Usage
1. Collect dataset for a new user
python face_collect.py
➡ Captures images and stores them in the dataset/ folder.

2. Train the model
python face_train.py
➡ Generates embeddings and saves them in trained_model/.

3. Run real-time face recognition
python face_recog.py
➡ Starts webcam and recognizes faces in real-time.

📂 Dataset & Models

The full dataset and trained models are not included in this repository due to size limitations.

👉 Download them here: https://drive.google.com/drive/folders/1GvVNil9ddYM89N4kH_j2c0NbTLFXEJFl?usp=sharing
