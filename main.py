import os

import cv2
import mediapipe as mp
import playsound
from dotenv import load_dotenv
from gtts import gTTS

from live_video_processing import preprocess_live
from model import classify_live, load_model
from utils import load_class_names, update_word_buffer

load_dotenv()

def continuous_capture_and_classify():
    cap = cv2.VideoCapture(0)
    model = load_model()
    idx2label = load_class_names()
    word_buffer = []

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    hands_detected = False  # Flag to keep track of hand detection state

    print("Show your hands to the camera to start predictions.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw the hand annotations on the frame for debugging/visual feedback
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            hands_detected = True
        else:
            hands_detected = False

        cv2.imshow('Sign Language to Speech', frame)

        # Process frames only if hands are detected
        if hands_detected:
            input_tensor = preprocess_live(cap)
            if input_tensor is not None:
                predictions = classify_live(input_tensor, model, idx2label, word_buffer)
                print("Predictions:", predictions)
            else:
                print("No frames to process.")
        # else:
        #     print("No hands detected. Waiting for hands to be shown...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    continuous_capture_and_classify()

