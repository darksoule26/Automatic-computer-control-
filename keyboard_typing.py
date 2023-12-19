import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Create a text file for recording
output_file = open("output.txt", "w")

# Define the keyboard layout
keyboard = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']
]

# Store information about key press animation
key_pressed_frames = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks of the index and thumb fingers
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Check if the index finger is pointing and thumb finger is up
            if index_finger_tip.y < thumb_tip.y:
                # Get the normalized coordinates of the index finger tip
                index_x, index_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

                # Display a circle at the index finger tip
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

                # Map the normalized coordinates to the screen resolution
                x = int(np.interp(index_x, [0, frame.shape[1]], [0, 1366]))
                y = int(np.interp(index_y, [0, frame.shape[0]], [0, 768]))

                # Write the corresponding character to the text file
                char = chr(65 + (x // 50))  # A-Z
                output_file.write(char)

                # Mark the key as pressed and store the frame count
                key_pressed_frames[char] = 30  # Number of frames to display the key press animation

    # Display the on-screen keyboard with translucent pink squares
    for i, row in enumerate(keyboard):
        for j, key in enumerate(row):
            x_pos = 50 * j
            y_pos = 70 * i

            # Decrease the frame count for the pressed key animation
            if key in key_pressed_frames and key_pressed_frames[key] > 0:
                key_pressed_frames[key] -= 1
                alpha = 255 - int(255 * key_pressed_frames[key] / 30)  # Corrected alpha calculation
            else:
                alpha = 100

            # Draw a translucent pink square
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_pos, y_pos), (x_pos + 50, y_pos + 70), (255, 0, 255, alpha), -1)

            # Add the overlay to the original frame
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            # Display the key label
            cv2.putText(frame, key, (x_pos + 15, y_pos + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Virtual Keyboard', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output_file.close()
cap.release()
cv2.destroyAllWindows()
