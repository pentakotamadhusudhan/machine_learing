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


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    cv2.rectangle(vid, (200, 100), (200 + 20, 100 + 20), (0, 255, 255), 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x , y, c = frame.shape
    faces = detect_bounding_box(
        frame
    ) 
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # Show the final output
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Get hand landmark prediction
    result = hands.process(framergb)
    # print(result)
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
            landmarks = []
            # label = result.multi_handedness[handNumber].classification[0].label
           
            for handslms in result.multi_hand_landmarks:
                # print("index finger",handslms.)

                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])
                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                for i, lm in enumerate(handslms.landmark):
                    if i == 0:
                        finger_name = 'Wrist'
                        cv2.putText(frame, f"{i}", (lmx, lmy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,)
                        print(finger_name)
                    elif i in range(1, 5):
                        finger_name = 'Thumb'
                        

                    elif i in range(5, 9):
                        finger_name = 'Index Finger'
                        cv2.putText(frame, f"{i}", (lmx, lmy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,)
                        print(finger_name)
                    elif i in range(9, 13):
                        finger_name = 'Middle Finger'
                    elif i in range(13, 17):
                        finger_name = 'Ring Finger'
                    elif i in range(17, 21):
                        finger_name = 'Pinky Finger'
                    else:
                        finger_name = 'Unknown'
                    # Display finger names on the frame
                   
                
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
            break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()