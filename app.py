from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import pickle
import pandas as pd
from pydub import AudioSegment
from playsound import playsound
from gtts import gTTS
import pygame
import pyttsx3
import time
import os
from pydub.playback import play as pydub_play

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

with open('hand_sign.pkl', 'rb') as f:
    model = pickle.load(f)

detected_letters = []
current_keyword = ""
current_letter_duration = 0
min_continuous_duration = 30

engine = pyttsx3.init()

audio_thread = None
is_audio_playing = False

# Initialize pygame mixer
pygame.mixer.init()

whole_word_text = ""

def play_continuous_audio(detected_letters):
    try:
        # Join detected letters without spaces
        whole_word_text = "".join(detected_letters)

        print("Generating audio data...")
        audio_data = generate_audio_data(whole_word_text)

        # Save the AudioSegment to a temporary WAV file
        temp_wav_path = 'temp_output.wav'
        audio_data.export(temp_wav_path, format='wav')

        print("Playing audio...")
        pygame.mixer.Sound(temp_wav_path).play()

        print("Audio playback initiated.")
    except Exception as e:
        print(f"Error during audio playback: {e}")
    finally:
        # Clean up: Remove the temporary WAV file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

def generate_audio_data(whole_word_text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Save the speech to a temporary WAV file
    temp_file_path = 'temp_output.wav'
    engine.save_to_file(whole_word_text, temp_file_path)
    engine.runAndWait()

    # Load the saved audio file
    audio_data = AudioSegment.from_file(temp_file_path, format='wav')

    # Clean up: Remove the temporary file
    os.remove(temp_file_path)

    return audio_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_frames', methods=['GET', 'POST'])
def generate_frames():
    global detected_letters, last_detected_letter, whole_word_text

    global current_keyword

    predicted_values = []
    detected_letters = []
    last_detected_letter = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return render_template("index.html")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from the camera.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                       )

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )

            body_language_class = "None"

            try:
                if results.left_hand_landmarks:
                    pose = results.left_hand_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    row = pose_row

                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    print(body_language_class)
                    speak = gTTS(text=body_language_class)
                    speak.save("show.mp3")
                    playsound('show.mp3')
                    os.remove('show.mp3')



            except Exception as e:
                print(e)

            

            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Camera Feed with Predictions', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_detected')
def clear_detected():
    global detected_letters, current_keyword, whole_word_text, last_detected_letter

    detected_letters = []
    current_keyword = ""
    whole_word_text = ""
    last_detected_letter = ""

    return "Detected letters and last detected letter cleared successfully."

    

    
if __name__ == '__main__':
    app.run(port=200)
