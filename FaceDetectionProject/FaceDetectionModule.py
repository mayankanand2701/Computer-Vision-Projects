import cv2
import mediapipe as mp
import time

desired_width = 900
desired_height = 600

class FaceDetector():
    def __init__(self,minDetectionCon=0.5):
        self.minDetectioCon=minDetectionCon

        self.mpFaceDetection=mp.solutions.face_detection
        self.faceDetection=self.mpFaceDetection.FaceDetection(minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findFaces(self,img,draw=True):
        img = cv2.resize(img, (desired_width, desired_height))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(imgRGB)
        boundingBoxes=[]
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img,detection)   # inbuilt code to create rectangle on face
                h,w,c=img.shape
                bboxC=detection.location_data.relative_bounding_box
                bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
                boundingBoxes.append([id,bbox,detection.score])
                # Code for fansy draw
                if draw:
                    # cv2.rectangle(img,bbox,(255,0,255),2)
                    img=self.fancyDraw(img,bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return img,boundingBoxes

    def fancyDraw(self,img,bbox,l=30,t=6):
        x,y,w,h=bbox
        x1,y1=x+w,y+h
        cv2.rectangle(img, bbox, (255, 0, 255), 2)
        # Top left
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom left
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img

def main():
    cap = cv2.VideoCapture('Videos/1.mp4')
    detector=FaceDetector()

    pTime = 0
    cTime = 0
    while True:
        success, img = cap.read()
        img,boundingBoxes=detector.findFaces(img)
        print(boundingBoxes)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()