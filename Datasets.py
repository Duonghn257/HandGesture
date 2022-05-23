import cv2
import mediapipe as mp
import pandas as pd

cap = cv2.VideoCapture(1)

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "HandBye"
no_of_frames = 600
def make_landmark_timestep(results):
    c_lm = []
    # print(results.multi_hand_landmarks[0])
    for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS)

    for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
        height, width, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * width), int(lm.y*height)
        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
    return img

while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if ret:
        # Hands Detect
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            # Get skeleton data
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            # Draw skeleton on Hand
            frame = draw_landmark_on_image(mpDraw, results, frame)

        cv2.imshow("img", frame)
        if cv2.waitKey(1) == ord('q'):
            break
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()