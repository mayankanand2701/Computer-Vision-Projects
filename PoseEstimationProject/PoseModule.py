import time
import cv2
import mediapipe as mp

desired_width = 600
desired_height = 600

class poseDetector():
    def __init__(self,mode=False,smooth=True,detectionCon=0.5,trackingCon=0.5):
        self.mode=mode
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackingCon=trackingCon

        self.mpPose=mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,  # Default value
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils


    def findPose(self,img,draw=True):
        img = cv2.resize(img, (desired_width, desired_height))
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return img

    def findPostions(self,img,draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture('Pose Videos/4.mp4')
    detector=poseDetector()

    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        img=detector.findPose(img)
        lmList=detector.findPostions(img)
        if len(lmList)!=0:
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

if __name__=="__main__":
    main()
