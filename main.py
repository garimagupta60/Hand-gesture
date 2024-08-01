import cv2
import mediapipe as mp
import time
import math
import streamlit as st

# Allow the camera to warm up
# time.sleep(2.0)

# Initialize MediaPipe Hand module and drawing utility
mp_draw = mp.solutions.drawing_utils    #to draw
mp_hand = mp.solutions.hands        #to capture hands

# Finger tip landmark indices
tipIds = [4, 8, 12, 16, 20]

st.title("Hand Gesture Detection")
st.write("This app detects hand gestures using MediaPipe and displays the number of fingers raised.")

start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")
video_placeholder = st.empty()

# Initialize video capture
video = None

# Calculate distance function
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

if start_button and video is None:
    video = cv2.VideoCapture(0)

if stop_button and video is not None:
    video.release()
    video = None
    st.write("Camera stopped")
    st.stop()

if video and video.isOpened():
    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while video.isOpened():
            ret, image = video.read()
            if not ret:
                st.write("Failed to grab frame.")
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            lmList = []
            handType = None
            if results.multi_hand_landmarks:
                for hand_landmark, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    for id, lm in enumerate(hand_landmark.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    handType = hand_handedness.classification[0].label  # 'Right' or 'Left'
                    mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
            
            fingers = []
            if lmList:
                # Thumb: Check if thumb tip is to the right of the thumb knuckle (for left hand, reverse for right hand)
                if (handType == 'Right' and lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]) or \
                   (handType == 'Left' and lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]):
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Fingers: Check if finger tip is above the middle knuckle
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total = fingers.count(1)

                # Calculate the distance between the index finger tip (8) and thumb tip (4)
                if len(lmList) > 8:
                    distance = calculate_distance(lmList[4][1], lmList[4][2], lmList[8][1], lmList[8][2])
                    
                    # Add a green rectangle background for the distance text
                    distance_text = f"Dist: {int(distance)}"
                    text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = 35
                    text_y = 400
                    rectangle_start = (text_x - 10, text_y - text_size[1] - 10)
                    rectangle_end = (text_x + text_size[0] + 10, text_y + 10)
                    cv2.rectangle(image, rectangle_start, rectangle_end, (0, 255, 0), cv2.FILLED)
                    cv2.putText(image, distance_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Make the rectangle smaller and the text smaller
                cv2.rectangle(image, (20, 300), (200, 350), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, f"{handType}: {total}", (30, 335), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            video_placeholder.image(image, channels="BGR")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if video:
    video.release()
    cv2.destroyAllWindows()
