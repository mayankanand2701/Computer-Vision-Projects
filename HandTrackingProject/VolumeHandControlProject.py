import time
import cv2
import numpy as np
import HandTrackingModule as htm
import math

cameraWidth=640
cameraHeight=480

cap=cv2.VideoCapture(0)
cap.set(3,cameraWidth)
cap.set(4,cameraHeight)

pTime=0
cTime=0

detector=htm.HandDetector(detectionCon=0.7)

# Pycaw code for System Volume
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volumeRange=volume.GetVolumeRange()

minVolume=volumeRange[0]
maxVolume=volumeRange[1]

vol=0
volBar=400
volPerctange=0

while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPositions(img,draw=False) # as already we are drawing above
    if len(lmList)!=0:
        #print(lmList[4],lmList[8])
        x1,y1=lmList[4][1],lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx,xy=(x1+x2)//2,(y1+y2)//2

        cv2.circle(img,(x1,y1),12,(255,0,0),cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (cx, xy), 8, (255, 0, 0), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)

        length=math.hypot(x2-x1,y2-y1)
        print(length)

        # Hand Range : 15 - 350
        # Volume Range : -65 - 0
        vol=np.interp(length,[50,250],[minVolume,maxVolume])
        volBar = np.interp(length, [50, 250], [400, 150])
        volPerctange = np.interp(length, [50, 250], [0, 100])

        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img, (cx, xy), 8, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPerctange)}%', (40,450), cv2.FONT_HERSHEY_PLAIN, 3, (0,255, 0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (15, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255, 0), 3)
    cv2.imshow("Captured Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()