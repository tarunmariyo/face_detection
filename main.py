import os
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

# Initialize the face recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model_trained = False  # Flag to check if the model is trained or loaded

# Function to get the images and labels
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        try:
            id = int(os.path.split(imagePath)[-1].split(".")[0])
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        except ValueError:
            # Skip files that do not match the expected format
            continue
    return faceSamples, ids

# Function to train the recognizer
def trainRecognizer():
    global model_trained
    faces, ids = getImagesAndLabels('TrainingImages')
    if faces and ids:
        recognizer.train(faces, np.array(ids))
        recognizer.save('face_recognizer_model.yml')
        model_trained = True
        messagebox.showinfo("Info", "Model trained and saved successfully")
    else:
        messagebox.showwarning("Warning", "No valid training data found.")

# Function to load the recognizer
def loadRecognizer():
    global model_trained
    if os.path.exists('face_recognizer_model.yml'):
        recognizer.read('face_recognizer_model.yml')
        model_trained = True
        messagebox.showinfo("Info", "Model loaded successfully")
    else:
        messagebox.showwarning("Warning", "No trained model found. Please train the model first.")

# Function to recognize faces in an image
def recognizeFace():
    if not model_trained:
        messagebox.showwarning("Warning", "Model is not trained or loaded. Please train or load the model first.")
        return

    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            try:
                roi_gray = gray[y:y + h, x:x + w]
                id, confidence = recognizer.predict(roi_gray)
                confidence_text = f"Confidence: {round(100 - confidence, 2)}%"
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, f"ID: {id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(img, confidence_text, (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during recognition: {e}")
        
        cv2.imshow('Recognized Faces', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to capture and recognize face from webcam
def captureAndRecognize():
    if not model_trained:
        messagebox.showwarning("Warning", "Model is not trained or loaded. Please train or load the model first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                roi_gray = gray[y:y + h, x:x + w]
                id, confidence = recognizer.predict(roi_gray)
                confidence_text = f"Confidence: {round(100 - confidence, 2)}%"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, confidence_text, (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during recognition: {e}")

        cv2.imshow('Webcam Face Recognition', frame)

        # Press 'q' to exit the webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the main application window
app = tk.Tk()
app.title("Face Recognition System")
app.geometry("400x300")

# Add buttons to the GUI
train_button = tk.Button(app, text="Train Model", command=trainRecognizer)
train_button.pack(pady=10)

load_button = tk.Button(app, text="Load Model", command=loadRecognizer)
load_button.pack(pady=10)

recognize_button = tk.Button(app, text="Recognize Face from File", command=recognizeFace)
recognize_button.pack(pady=10)

webcam_button = tk.Button(app, text="Capture and Recognize Face from Webcam", command=captureAndRecognize)
webcam_button.pack(pady=10)

# Start the GUI event loop
app.mainloop()
