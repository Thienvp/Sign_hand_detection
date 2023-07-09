import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from generation_model import create_model

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q",
          "R", "S", "T", "U", "V", "W", "X", "Y"]

detector = HandDetector(maxHands=1)
camera = cv2.VideoCapture(0)
model = create_model()
model.load_weights(".\weights\weights_ANN.h5")
while(True):
    (t, img) = camera.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, h, w = hand["bbox"]
        hand_type = hand["type"]
        lmList = np.array(hand["lmList"])
        if(hand_type == "Right"):
            lmList[:,0] = lmList[:,0]*-1
        distances = []
        for i_p1 in range(21 - 1):
            for i_p2 in range(i_p1 + 1, 21):
                distances.append(lmList[i_p1] - lmList[i_p2])
        data = np.array(distances).reshape(-1)
        y_pre = model.predict(np.array([data]))
        label = np.argmax(y_pre, axis = 1)
        label = labels[label[0]]
        img = cv2.putText(img, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
    cv2.imshow("Video Feed", img)
    if(cv2.waitKey(1) == ord('e')):
        break

# free up memory
camera.release()
cv2.destroyAllWindows()