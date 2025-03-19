import time
import cv2
import PoseModule as pm

desired_width = 600
desired_height = 600


cap = cv2.VideoCapture('Pose Videos/4.mp4')
detector=pm.poseDetector()

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img=detector.findPose(img)
    lmList=detector.findPostions(img)
    if len(lmList) != 0:
        print(lmList[14])
        # To generate the circle for which position we are tracking the list
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()