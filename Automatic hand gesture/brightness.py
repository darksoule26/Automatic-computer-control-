import cv2
import mediapipe as mp
from screen_brightness_control import get_brightness, set_brightness
import pyautogui

# Function to set brightness using the screen-brightness-control library
def set_brightness_percentage(brightness_control, brightness):
    brightness_control.set_brightness(brightness)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

brightness_controls = get_brightness(display=0)  # Change the display index if needed
brightness_control = brightness_controls[0]

brightness = 50  # Initial brightness level (0-100)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

            if index_finger_y < thumb_y:
                hand_gesture = 'pointing up'
            elif index_finger_y > thumb_y:
                hand_gesture = 'pointing down'
            else:
                hand_gesture = 'other'

            if hand_gesture == 'pointing up':
                brightness = min(brightness + 5, 100)
            elif hand_gesture == 'pointing down':
                brightness = max(brightness - 5, 0)

    # Set brightness using the screen-brightness-control library
    set_brightness_percentage(brightness_control, brightness)

    cv2.imshow('Hand Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
