import cv2
import mediapipe as mp
import time
import HandTrackingModule as mpd

pTime = 0
cTime=0
cap = cv2.VideoCapture(0)
detector = mpd.HandDetector()

while True:
    success, img = cap.read()
    if not success:
        continue

    img = detector.findHands(img)
    lmList = detector.findPositions(img)

    if len(lmList)!=0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Captured Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()