import time
import cv2
import os
import HandTrackingModule as htm

cameraWidth=640
cameraHeight=480

cap=cv2.VideoCapture(0)
cap.set(3,cameraWidth)
cap.set(4,cameraHeight)

pTime=0
cTime=0

folderPath="FingerImages"
myList=os.listdir(folderPath)
print(myList)
overLayList=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)

# print(len(overLayList))

detector=htm.HandDetector(detectionCon=0.7)
tipIds=[4,8,12,16,20]

while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPositions(img,draw=False)
    # print(lmList)
    if len(lmList)!=0:
        fingers=[]

        # For thumbs (right hand)
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # For 4 Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingersCount=fingers.count(1)
        print(totalFingersCount)

        img[0:200,0:200]=overLayList[totalFingersCount-1]

        cv2.rectangle(img,(20,275),(150,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingersCount),(45,400),cv2.FONT_HERSHEY_PLAIN,8,(0,0,0),20)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Captured Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

