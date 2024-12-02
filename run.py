import cv2
import os
import time
from keras import models
import numpy as np
from pygame import mixer
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import sys

# Khởi tạo âm thanh cảnh báo
mixer.init()
sound = mixer.Sound('alarm.wav')

# Tải các file cascade để phát hiện khuôn mặt và mắt
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Tải mô hình dự đoán trạng thái mắt
model = models.load_model('models/CNN.h5')

# Helper function to predict the state of the eye (Open/Closed)
def predict_eye_state(eye_image):
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    eye_image = cv2.resize(eye_image, (24, 24)) / 255.0
    eye_image = np.expand_dims(eye_image.reshape(24, 24, 1), axis=0)
    prediction = np.argmax(model.predict(eye_image), axis=-1)
    return 'Open' if prediction[0] == 1 else 'Closed'


# Process video feed
def process_video(video_source, frame_skip=2):
    cap = cv2.VideoCapture(video_source)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    face_scores = {}
    last_sound_play_time = 0
    thicc = 2
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from video.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip some frames to improve performance

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))

        for i, (x, y, w, h) in enumerate(faces):
            face_frame = frame[y:y + h, x:x + w]

            # Detect and predict right eye state
            right_eye = reye_cascade.detectMultiScale(gray[y:y + h, x:x + w])
            rpred = 'Open'
            for (ex, ey, ew, eh) in right_eye:
                r_eye = face_frame[ey:ey + eh, ex:ex + ew]
                rpred = predict_eye_state(r_eye)
                break

            # Detect and predict left eye state
            left_eye = leye_cascade.detectMultiScale(gray[y:y + h, x:x + w])
            lpred = 'Open'
            for (ex, ey, ew, eh) in left_eye:
                l_eye = face_frame[ey:ey + eh, ex:ex + ew]
                lpred = predict_eye_state(l_eye)
                break

            # Initialize face score if not already
            face_id = i
            if face_id not in face_scores:
                face_scores[face_id] = 0

            # Update score based on eye states
            if rpred == 'Closed' and lpred == 'Closed':
                face_scores[face_id] += 1
            else:
                face_scores[face_id] = max(0, face_scores[face_id] - 1)

            # Cap the score between 0 and 5
            face_scores[face_id] = min(face_scores[face_id], 7)

            # Draw rectangles and display score
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
            cv2.putText(frame, f'Score: {face_scores[face_id]}', (x, y + h + 20), font, 1, (255, 255, 255), 1,
                        cv2.LINE_AA)

            # Trigger alarm if score exceeds threshold
            if face_scores[face_id] >= 7:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thicc)
                if sound and time.time() - last_sound_play_time > 3:
                    sound.play()
                    last_sound_play_time = time.time()
                thicc = min(thicc + 2, 16)
            else:
                thicc = max(2, thicc - 2)

        # Display frame
        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Use the camera feed
def use_camera():
    process_video(0)


# Use video file as input
def use_video():
    video_path = filedialog.askopenfilename()
    if video_path:
        process_video(video_path)


# Exit the program
def quit_program():
    root.quit()


# Setup Tkinter GUI
root = tk.Tk()
root.title("Drowsiness Detection System")

# Set window size and layout
root.geometry("800x600")
root.resizable(False, False)

# Create left control panel
control_frame = tk.Frame(root, width=200, bg='#f0f0f0')
control_frame.pack(side="left", fill="y")

# Create right video display area
video_frame = tk.Frame(root, width=600, height=600, bg='white')
video_frame.pack(side="right", fill="both", expand=True)

# Add control buttons
btn_camera = ttk.Button(control_frame, text="Camera", command=use_camera)
btn_camera.pack(pady=20, padx=10, fill="x")

btn_video = ttk.Button(control_frame, text="File", command=use_video)
btn_video.pack(pady=20, padx=10, fill="x")

btn_exit = ttk.Button(control_frame, text="Exit", command=quit_program)
btn_exit.pack(pady=20, padx=10, fill="x")

# Set background color
root.configure(bg='#e6f2ff')

# Start the Tkinter event loop
root.mainloop()
