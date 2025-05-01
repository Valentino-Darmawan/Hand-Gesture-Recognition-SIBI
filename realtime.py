import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# Load trained model
model = tf.keras.models.load_model('model_sibi.h5')

# List of class names (A-Y without J and Z)
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# Mediapipe hands detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_coords) * frame.shape[1]) - 20
            xmax = int(max(x_coords) * frame.shape[1]) + 20
            ymin = int(min(y_coords) * frame.shape[0]) - 20
            ymax = int(max(y_coords) * frame.shape[0]) + 20

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(frame.shape[1], xmax), min(frame.shape[0], ymax)

            # Crop ROI
            roi = frame[ymin:ymax, xmin:xmax]
            if roi.size == 0:
                continue

            # Preprocess ROI
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            # Prediction
            prediction = model.predict(roi, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            label_text = f"{class_names[predicted_class]} ({confidence*100:.2f}%)"

            # Draw bounding box and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(frame, label_text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()