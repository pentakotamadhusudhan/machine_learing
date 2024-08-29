import cv2
import numpy as np
import mediapipe as mp
from tensorflow.python.keras.models import load_model

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model_path = r'C:\Users\Admin\Downloads\hand-gesture-recognition-code\mp_hand_gesture'
try:
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Load class names
class_names_path = r'C:\Users\Admin\Downloads\hand-gesture-recognition-code\gesture.names'
try:
    with open(class_names_path, 'r') as f:
        classNames = f.read().split('\n')
    print(f"Class names loaded from {class_names_path}")
except FileNotFoundError as e:
    print(f"Class names file not found: {e}")
except Exception as e:
    print(f"Error loading class names: {e}")



# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x , y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # Show the final output
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
            break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()